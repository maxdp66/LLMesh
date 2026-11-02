//! llmesh: pool several ordinary machines to run one LLM together.

mod api;
mod config;
mod discovery;
mod group;
mod llama;
mod models;
mod net;
mod state;
mod sys;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use tracing::{error, info, warn};

use crate::config::{Paths, SavedGroup};
use crate::group::{GroupConfig, GroupSecret, Invite, SignedConfig};
use crate::net::{Mesh, NodeOpts};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DEFAULT_MODEL: &str = "unsloth/Qwen3-8B-GGUF:Q4_K_M";
pub const DEFAULT_CTX: u32 = 8192;

#[derive(Parser, Debug)]
#[command(name = "llmesh", version, about = "Pool several machines' CPUs, GPUs and RAM to run one LLM together")]
struct Cli {
    /// Where keys, groups, llama.cpp and (with this flag) models are stored.
    #[arg(long, global = true, env = "LLMESH_DATA_DIR")]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Option<Cmd>,
    #[command(flatten)]
    start: StartArgs,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run this node in the saved group (or the zero-setup LAN group). This is the default.
    Start(StartArgs),
    /// Manage groups.
    #[command(subcommand)]
    Group(GroupCmd),
    /// Join a group with an invite code, then start.
    Join {
        invite: String,
        #[command(flatten)]
        start: StartArgs,
    },
    /// Print an invite code for the current group.
    Invite {
        /// API port of the running node to ask for a fresh invite.
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// List saved groups.
    Groups,
    /// Leave the current group and go back to the LAN group.
    Leave,
}

#[derive(Subcommand, Debug)]
enum GroupCmd {
    /// Create a group (you become its admin), print an invite, then start.
    Create {
        #[arg(value_name = "NAME")]
        group_name: String,
        #[command(flatten)]
        start: StartArgs,
    },
}

#[derive(Args, Debug, Clone)]
struct StartArgs {
    /// Model to run (admin only): owner/repo:QUANT, owner/repo, owner/repo/file.gguf, a path or an https URL.
    #[arg(long)]
    model: Option<String>,
    /// Port of the local OpenAI-compatible API and dashboard (tries the next ports if taken).
    #[arg(long, default_value_t = 8080, env = "LLMESH_PORT")]
    port: u16,
    /// Only lend this machine's resources; never become the head.
    #[arg(long)]
    worker_only: bool,
    /// Listen on all interfaces instead of 127.0.0.1 (the API has no authentication!).
    #[arg(long)]
    public_api: bool,
    /// Context size (admin only; stored in the group config).
    #[arg(long)]
    ctx: Option<u32>,
    /// Force a llama.cpp backend: metal, cuda, vulkan or cpu.
    #[arg(long, env = "LLMESH_BACKEND")]
    backend: Option<String>,
    /// Use llama.cpp binaries from this folder instead of downloading them.
    #[arg(long, env = "LLMESH_LLAMA_DIR")]
    llama_dir: Option<PathBuf>,
    /// Display name of this node.
    #[arg(long, env = "LLMESH_NAME")]
    name: Option<String>,
    /// Switch to this saved group (name or tag) before starting.
    #[arg(long)]
    group: Option<String>,
    /// Start llama-server with --embeddings (for embedding models).
    #[arg(long)]
    embeddings: bool,
    /// Don't use n0 relays or DNS lookup: direct connections only (LAN, tests).
    /// (Also enabled by LLMESH_NO_RELAY=1.)
    #[arg(long)]
    no_relay: bool,
    /// Don't send or listen for LAN discovery beacons.
    #[arg(long)]
    no_lan: bool,
    /// UDP port for LAN discovery beacons.
    #[arg(long, default_value_t = discovery::DEFAULT_PORT, env = "LLMESH_DISCOVERY_PORT")]
    discovery_port: u16,
    /// Fixed UDP port for the iroh endpoint (default: random).
    #[arg(long, env = "LLMESH_P2P_PORT")]
    p2p_port: Option<u16>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("LLMESH_LOG").unwrap_or_else(|_| "llmesh=info,iroh=warn,noq=warn,warn".into()),
        )
        .with_target(false)
        .init();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(run(cli))
}

async fn run(cli: Cli) -> Result<()> {
    let paths = Paths::new(cli.data_dir.clone())?;
    match cli.cmd {
        None => start(paths, cli.start, false).await,
        Some(Cmd::Start(a)) => start(paths, a, false).await,
        Some(Cmd::Group(GroupCmd::Create { group_name: name, start: a })) => {
            if let Some(m) = &a.model {
                models::ModelRef::parse(m)?;
            }
            let key = paths.node_key()?;
            let secret = GroupSecret::generate();
            let mut cfg = GroupConfig::initial(a.model.as_deref().unwrap_or(DEFAULT_MODEL));
            if let Some(c) = a.ctx {
                cfg.ctx_size = c;
            }
            let g = SavedGroup {
                name: name.clone(),
                secret_hex: hex::encode(secret.0),
                admin: Some(key.public()),
                bootstrap: vec![],
                members: Default::default(),
                config: Some(SignedConfig::sign(&secret, &key, cfg)),
                votes: Default::default(),
            };
            g.save(&paths)?;
            paths.set_current_group(Some(&secret.tag_hex()))?;
            println!("Created group \"{name}\". You are its admin.");
            start(paths, StartArgs { model: None, ctx: None, ..a }, true).await
        }
        Some(Cmd::Join { invite, start: a }) => {
            let inv = Invite::decode(&invite)?;
            let tag = inv.secret.tag_hex();
            let mut g = SavedGroup::from_invite(&inv);
            if let Ok(old) = SavedGroup::load(&paths.group_file(&tag)) {
                g.members = old.members;
                g.config = old.config;
                g.votes = old.votes;
            }
            g.save(&paths)?;
            paths.set_current_group(Some(&tag))?;
            println!("Joined group \"{}\".", inv.name);
            start(paths, a, false).await
        }
        Some(Cmd::Invite { port }) => print_invite(&paths, port).await,
        Some(Cmd::Groups) => {
            let cur = paths.current_group_tag();
            let groups = paths.list_groups();
            println!("{} LAN (default, automatic discovery)", if cur.is_none() { "*" } else { " " });
            for g in groups.iter().filter(|g| !g.is_lan()) {
                let tag = g.secret().tag_hex();
                let online = g.members.len();
                println!("{} {}  [{}]  {} known member(s)", if cur.as_deref() == Some(tag.as_str()) { "*" } else { " " }, g.name, tag, online);
            }
            Ok(())
        }
        Some(Cmd::Leave) => {
            let Some(tag) = paths.current_group_tag() else {
                println!("Not in a group (using the LAN group).");
                return Ok(());
            };
            let name = SavedGroup::load(&paths.group_file(&tag)).map(|g| g.name).unwrap_or(tag.clone());
            let _ = std::fs::remove_file(paths.group_file(&tag));
            paths.set_current_group(None)?;
            println!("Left \"{name}\". This node will use the LAN group next time it starts.");
            Ok(())
        }
    }
}

async fn print_invite(paths: &Paths, port: u16) -> Result<()> {
    // Ask the running node first: its invite lists the members that are online right now.
    let url = format!("http://127.0.0.1:{port}/mesh/api/status");
    if let Ok(r) = reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(3)).build()?.get(&url).send().await {
        if let Ok(v) = r.json::<serde_json::Value>().await {
            if let Some(inv) = v.get("invite").and_then(|i| i.as_str()) {
                println!("{inv}");
                return Ok(());
            }
            if v.pointer("/group/is_lan").and_then(|b| b.as_bool()) == Some(true) {
                bail!("this node is in the LAN group, which needs no invite (create one with `llmesh group create <name>`)");
            }
        }
    }
    let g = paths.active_group()?;
    let Some(admin) = g.admin else { bail!("not in a group; create one with `llmesh group create <name>`") };
    let mut members: Vec<_> = g.members.values().collect();
    members.sort_by_key(|m| std::cmp::Reverse(m.last_seen));
    let boots: Vec<_> = members.into_iter().map(|m| m.addr.clone()).chain(g.bootstrap.iter().cloned()).take(Invite::MAX_BOOTSTRAP).collect();
    eprintln!("(llmesh is not running here, so this invite uses the last known members)");
    println!("{}", Invite { secret: g.secret(), admin, bootstrap: boots, name: g.name.clone() }.encode());
    Ok(())
}

async fn start(paths: Paths, a: StartArgs, show_invite: bool) -> Result<()> {
    if let Some(name) = &a.group {
        let g = paths.find_group(name).with_context(|| format!("no saved group named {name:?} (see `llmesh groups`)"))?;
        let tag = g.secret().tag_hex();
        paths.set_current_group(if g.is_lan() { None } else { Some(&tag) })?;
    }
    if let Some(b) = &a.backend {
        llama::Backend::parse(b)?;
    }
    let group = paths.active_group()?;
    let key = paths.node_key()?;
    let opts = NodeOpts {
        name: a.name.clone().unwrap_or_else(sys::hostname),
        worker_only: a.worker_only,
        no_relay: a.no_relay || std::env::var("LLMESH_NO_RELAY").is_ok_and(|v| !v.is_empty() && v != "0" && v != "false"),
        default_ctx: a.ctx.unwrap_or(DEFAULT_CTX),
        // LLMESH_TEST_STARTED_AT lets the end-to-end tests stage an "older head shows up".
        started_at: std::env::var("LLMESH_TEST_STARTED_AT").ok().and_then(|v| v.parse().ok()).unwrap_or_else(group::now_ms),
    };
    let is_lan = group.is_lan();
    let gname = group.name.clone();
    let mesh = Mesh::bind(key, group, paths.clone(), opts, a.p2p_port).await?;
    let _ = mesh.peer_http.set(api::peer_router(mesh.clone()));

    if let Some(m) = &a.model {
        models::ModelRef::parse(m)?;
        if mesh.can_admin() {
            mesh.update_config(|c| c.active_model = m.clone())?;
        } else {
            warn!("only the admin picks the model; voting for {m} instead");
            mesh.cast_vote(m)?;
        }
    }
    if let Some(ctx) = a.ctx {
        if mesh.can_admin() && mesh.config().is_some_and(|c| c.config.ctx_size != ctx) {
            mesh.update_config(|c| c.ctx_size = ctx)?;
        }
    }
    mesh.spawn_tasks();
    if !a.no_lan {
        discovery::spawn(mesh.clone(), a.discovery_port);
    }
    let port = api::serve(mesh.clone(), a.port, a.public_api).await?;

    // Wait briefly for our addresses so the banner/invite is useful.
    if !mesh.opts.no_relay {
        let _ = tokio::time::timeout(Duration::from_secs(8), mesh.endpoint.online()).await;
    } else {
        for _ in 0..50 {
            if mesh.my_addr().ip_addrs().next().is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    println!();
    println!("  llmesh {VERSION}  node {} ({})", mesh.opts.name, mesh.id().fmt_short());
    println!("  group      {}{}", gname, if is_lan { " (automatic LAN discovery)" } else if mesh.can_admin() { " (you are admin)" } else { "" });
    println!("  dashboard  http://localhost:{port}/");
    println!("  OpenAI API http://localhost:{port}/v1   (any API key and model name work)");
    if a.public_api {
        println!("  WARNING: --public-api exposes the API without authentication on all interfaces");
    }
    if show_invite || (!is_lan && mesh.can_admin()) {
        if let Some(inv) = mesh.invite() {
            println!();
            println!("  Invite others with:");
            println!("    llmesh join {inv}");
        }
    }
    println!();

    // llama.cpp setup can take a while (first-run download), so it runs in the background.
    let m = mesh.clone();
    let lopts = llama::LlamaOpts { backend: a.backend.clone(), llama_dir: a.llama_dir.clone() };
    let hopts = llama::HeadOpts { embeddings: a.embeddings };
    tokio::spawn(async move {
        let mut wait = Duration::from_secs(30);
        let inst = loop {
            match llama::setup(&m.paths, &lopts).await {
                Ok(i) => break i,
                Err(e) => {
                    error!("llama.cpp setup failed: {e:#} (retrying in {}s)", wait.as_secs());
                    m.update_local(|l| l.llama_error = Some(format!("{e:#}")));
                    tokio::time::sleep(wait).await;
                    wait = (wait * 2).min(Duration::from_secs(600));
                }
            }
        };
        if inst.tag != llama::pinned_tag() {
            warn!("llama.cpp is {} but this llmesh pins {}; nodes on other tags are left out of the split", inst.tag, llama::pinned_tag());
        }
        m.update_local(|l| {
            l.llama_tag = inst.tag.clone();
            l.backend = inst.backend.clone();
            l.gpus = inst.gpus.clone();
            l.llama_error = None;
        });
        llama::spawn_rpc_server(m.clone(), inst.clone());
        llama::run_controller(m, inst, hopts).await;
    });

    tokio::signal::ctrl_c().await?;
    info!("shutting down");
    mesh.save_group();
    let _ = tokio::time::timeout(Duration::from_secs(3), mesh.shutdown()).await;
    Ok(())
}
