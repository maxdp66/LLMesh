//! llama.cpp management: download the pinned release, pick a backend, run `rpc-server` on every
//! node and `llama-server` on the head, and keep both alive.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use iroh::EndpointId;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tracing::{debug, info, warn};

use crate::models::{self, ModelRef};
use crate::net::{self, Mesh};
use crate::state::{self, GpuDevice, HeadState, Role};

/// llama.cpp release every node uses, so all nodes speak the same RPC protocol version.
pub const DEFAULT_TAG: &str = "b12017";

pub fn pinned_tag() -> String {
    std::env::var("LLMESH_LLAMA_TAG").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| DEFAULT_TAG.to_string())
}

fn base_url() -> String {
    std::env::var("LLMESH_LLAMA_BASE_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://github.com/ggml-org/llama.cpp/releases/download".into())
        .trim_end_matches('/')
        .to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Metal,
    Cuda,
    Vulkan,
    Cpu,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Cpu => "cpu",
        }
    }
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "metal" => Backend::Metal,
            "cuda" => Backend::Cuda,
            "vulkan" => Backend::Vulkan,
            "cpu" => Backend::Cpu,
            _ => bail!("unknown backend {s:?} (use metal, cuda, vulkan or cpu)"),
        })
    }
    fn is_gpu(&self) -> bool {
        !matches!(self, Backend::Cpu)
    }
}

/// Release asset names for this platform and backend (main archive first).
pub fn assets(tag: &str, os: &str, arch: &str, b: Backend) -> Option<Vec<String>> {
    let a = |v: &str, ext: &str| format!("llama-{tag}-bin-{v}.{ext}");
    Some(match (os, arch, b) {
        ("macos", "aarch64", Backend::Metal) => vec![a("macos-arm64", "tar.gz")],
        ("macos", "x86_64", Backend::Cpu) => vec![a("macos-x64", "tar.gz")],
        ("linux", "x86_64", Backend::Cuda) => {
            vec![a("ubuntu-cuda-12.8-x64", "tar.gz"), format!("cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz")]
        }
        ("linux", "x86_64", Backend::Vulkan) => vec![a("ubuntu-vulkan-x64", "tar.gz")],
        ("linux", "x86_64", Backend::Cpu) => vec![a("ubuntu-x64", "tar.gz")],
        ("linux", "aarch64", Backend::Vulkan) => vec![a("ubuntu-vulkan-arm64", "tar.gz")],
        ("linux", "aarch64", Backend::Cpu) => vec![a("ubuntu-arm64", "tar.gz")],
        ("windows", "x86_64", Backend::Cuda) => vec![a("win-cuda-12.4-x64", "zip"), "cudart-llama-bin-win-cuda-12.4-x64.zip".into()],
        ("windows", "x86_64", Backend::Vulkan) => vec![a("win-vulkan-x64", "zip")],
        ("windows", "x86_64", Backend::Cpu) => vec![a("win-cpu-x64", "zip")],
        ("windows", "aarch64", Backend::Cpu) => vec![a("win-cpu-arm64", "zip")],
        _ => return None,
    })
}

fn on_path(name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(&exe)).find(|p| p.is_file())
}

fn has_vulkan() -> bool {
    if cfg!(windows) {
        let root = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("C:\\Windows"));
        return root.join("System32").join("vulkan-1.dll").is_file();
    }
    [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/lib/x86_64-linux-gnu",
        "/lib/aarch64-linux-gnu",
        "/usr/local/lib",
    ]
    .iter()
    .any(|d| Path::new(d).join("libvulkan.so.1").exists())
}

/// Backends to try on this machine, best first.
pub fn candidates() -> Vec<Backend> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => vec![Backend::Metal],
        ("macos", _) => vec![Backend::Cpu],
        ("linux" | "windows", "x86_64") => {
            let mut v = vec![];
            if on_path("nvidia-smi").is_some() {
                v.push(Backend::Cuda);
            }
            if has_vulkan() {
                v.push(Backend::Vulkan);
            }
            v.push(Backend::Cpu);
            v
        }
        ("linux", "aarch64") => {
            if has_vulkan() {
                vec![Backend::Vulkan, Backend::Cpu]
            } else {
                vec![Backend::Cpu]
            }
        }
        _ => vec![Backend::Cpu],
    }
}

#[derive(Debug, Clone)]
pub struct Install {
    pub bin_dir: PathBuf,
    pub rpc_server: PathBuf,
    pub llama_server: PathBuf,
    pub tag: String,
    pub backend: String,
    pub gpus: Vec<GpuDevice>,
}

impl Install {
    fn command(&self, exe: &Path) -> Command {
        let mut c = Command::new(exe);
        set_lib_env(&mut c, &self.bin_dir);
        c.kill_on_drop(true).stdin(Stdio::null());
        c
    }
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }
}

fn set_lib_env(c: &mut Command, dir: &Path) {
    let var = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let mut parts = vec![dir.to_path_buf()];
    if let Some(old) = std::env::var_os(var) {
        parts.extend(std::env::split_paths(&old));
    }
    if let Ok(joined) = std::env::join_paths(parts) {
        c.env(var, joined);
    }
}

fn find_bin(dir: &Path, name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let ft = match e.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ft.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|f| f == exe.as_str()) {
                return Some(p);
            }
        }
    }
    None
}

fn find_install(dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let rpc = find_bin(dir, "rpc-server")?;
    let srv = find_bin(dir, "llama-server")?;
    Some((rpc, srv))
}

/// Parse `--list-devices` output lines like `  CUDA0: NVIDIA RTX 3060 (12288 MiB, 11000 MiB free)`.
pub fn parse_devices(out: &str) -> Vec<GpuDevice> {
    let mut v = vec![];
    for line in out.lines() {
        let line = line.trim();
        let Some((name, rest)) = line.split_once(": ") else { continue };
        if name.contains(' ') || name.is_empty() {
            continue;
        }
        let Some(open) = rest.rfind('(') else { continue };
        let desc = rest[..open].trim().to_string();
        let inner = rest[open + 1..].trim_end_matches(')');
        let nums: Vec<u64> = inner.split(',').filter_map(|p| p.split_whitespace().next()?.parse().ok()).collect();
        if nums.len() == 2 && inner.contains("MiB") {
            v.push(GpuDevice { name: name.to_string(), description: desc, total: nums[0] << 20, free: nums[1] << 20 });
        }
    }
    v
}

async fn run_capture(mut c: Command, timeout: Duration) -> Result<String> {
    c.stdout(Stdio::piped()).stderr(Stdio::piped());
    let out = tokio::time::timeout(timeout, c.output()).await.map_err(|_| anyhow!("timed out"))??;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !out.status.success() {
        bail!("exit {}: {}", out.status, text.lines().rev().take(5).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" | "));
    }
    Ok(text)
}

/// `llama-server --version` prints `version: 12017 (abcdef)`; that's release tag `b12017`.
async fn detect_tag(bin_dir: &Path, server: &Path) -> Option<String> {
    let mut c = Command::new(server);
    c.arg("--version");
    set_lib_env(&mut c, bin_dir);
    let out = run_capture(c, Duration::from_secs(20)).await.ok()?;
    let line = out.lines().find(|l| l.trim_start().starts_with("version:"))?;
    let n = line.split(':').nth(1)?.split_whitespace().next()?;
    n.parse::<u64>().ok().map(|n| format!("b{n}"))
}

async fn probe(bin_dir: &Path, server: &Path, backend: Backend) -> Result<Vec<GpuDevice>> {
    let mut c = Command::new(server);
    c.arg("--list-devices");
    set_lib_env(&mut c, bin_dir);
    let out = run_capture(c, Duration::from_secs(60)).await?;
    let devs = parse_devices(&out);
    if backend.is_gpu() && devs.is_empty() {
        bail!("no {} devices found", backend.as_str());
    }
    Ok(devs)
}

async fn download(url: &str, dest: &Path) -> Result<()> {
    info!("downloading {url}");
    let r = models::http_client().get(url).send().await?;
    if !r.status().is_success() {
        bail!("HTTP {} for {url}", r.status());
    }
    let total = r.content_length().unwrap_or(0);
    let mut f = tokio::fs::File::create(dest).await?;
    let mut s = r.bytes_stream();
    let (mut done, mut last) = (0u64, Instant::now());
    while let Some(chunk) = s.next().await {
        let chunk = chunk?;
        f.write_all(&chunk).await?;
        done += chunk.len() as u64;
        if last.elapsed() > Duration::from_secs(3) {
            last = Instant::now();
            info!("  {} / {}", crate::sys::human_bytes(done), crate::sys::human_bytes(total));
        }
    }
    f.flush().await?;
    Ok(())
}

pub fn extract(archive: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    let name = archive.to_string_lossy().to_ascii_lowercase();
    if name.ends_with(".zip") {
        let f = std::fs::File::open(archive)?;
        let mut z = zip::ZipArchive::new(f)?;
        z.extract(dest)?;
    } else {
        let f = std::fs::File::open(archive)?;
        let mut t = tar::Archive::new(flate2::read::GzDecoder::new(f));
        t.set_preserve_permissions(true);
        t.set_overwrite(true);
        t.unpack(dest)?;
    }
    Ok(())
}

async fn install_release(root: &Path, tag: &str, b: Backend) -> Result<PathBuf> {
    let names = assets(tag, std::env::consts::OS, std::env::consts::ARCH, b)
        .ok_or_else(|| anyhow!("no {} build of llama.cpp for {}/{}", b.as_str(), std::env::consts::OS, std::env::consts::ARCH))?;
    let dir = root.join(tag).join(b.as_str());
    if dir.join(".complete").exists() {
        return Ok(dir);
    }
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let dl = root.join("downloads");
    std::fs::create_dir_all(&dl)?;
    for n in &names {
        let file = dl.join(n);
        if !file.exists() {
            let tmp = dl.join(format!("{n}.part"));
            download(&format!("{}/{tag}/{n}", base_url()), &tmp).await?;
            std::fs::rename(&tmp, &file)?;
        }
        let (f, d) = (file.clone(), dir.clone());
        tokio::task::spawn_blocking(move || extract(&f, &d)).await??;
        let _ = std::fs::remove_file(&file);
    }
    // cudart archives are flat; put their DLLs/libs next to the binaries.
    if let Some((rpc, _)) = find_install(&dir) {
        let bin = rpc.parent().unwrap().to_path_buf();
        if bin != dir {
            if let Ok(rd) = std::fs::read_dir(&dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_file() && p.file_name().is_some_and(|f| f != ".complete") {
                        let _ = std::fs::rename(&p, bin.join(p.file_name().unwrap()));
                    }
                }
            }
        }
    }
    std::fs::write(dir.join(".complete"), tag)?;
    Ok(dir)
}

#[derive(Debug, Clone, Default)]
pub struct LlamaOpts {
    pub backend: Option<String>,
    pub llama_dir: Option<PathBuf>,
}

async fn finish(rpc: PathBuf, srv: PathBuf, tag: Option<String>, backend: Backend) -> Result<Install> {
    let bin_dir = srv.parent().unwrap().to_path_buf();
    let gpus = probe(&bin_dir, &srv, backend).await?;
    let tag = match tag {
        Some(t) => t,
        None => detect_tag(&bin_dir, &srv).await.unwrap_or_else(|| "unknown".into()),
    };
    Ok(Install { bin_dir, rpc_server: rpc, llama_server: srv, tag, backend: backend.as_str().into(), gpus })
}

/// Find or download llama.cpp and pick a working backend.
pub async fn setup(paths: &crate::config::Paths, opts: &LlamaOpts) -> Result<Install> {
    let tag = pinned_tag();
    let forced = opts.backend.as_deref().map(Backend::parse).transpose()?;

    if let Some(dir) = &opts.llama_dir {
        let (rpc, srv) = find_install(dir).ok_or_else(|| anyhow!("no rpc-server and llama-server under {}", dir.display()))?;
        let bin_dir = srv.parent().unwrap().to_path_buf();
        let detected = detect_tag(&bin_dir, &srv).await;
        let b = forced.unwrap_or_else(|| candidates()[0]);
        let gpus = probe(&bin_dir, &srv, Backend::Cpu).await.unwrap_or_default();
        let backend = if gpus.is_empty() && b.is_gpu() { Backend::Cpu } else { b };
        return Ok(Install { bin_dir, rpc_server: rpc, llama_server: srv, tag: detected.unwrap_or(tag), backend: backend.as_str().into(), gpus });
    }

    // A system install (e.g. Homebrew) of the pinned version is used as-is.
    let system = match (on_path("rpc-server"), on_path("llama-server")) {
        (Some(r), Some(s)) => Some((r, s)),
        _ => None,
    };
    if let Some((r, s)) = &system {
        let bin = s.parent().unwrap().to_path_buf();
        if detect_tag(&bin, s).await.as_deref() == Some(tag.as_str()) {
            let b = forced.unwrap_or(if cfg!(target_os = "macos") { Backend::Metal } else { Backend::Cpu });
            if let Ok(i) = finish(r.clone(), s.clone(), Some(tag.clone()), b).await.or(finish(r.clone(), s.clone(), Some(tag.clone()), Backend::Cpu).await) {
                info!("using llama.cpp {} from PATH ({})", i.tag, bin.display());
                return Ok(i);
            }
        }
    }

    let list = match forced {
        Some(b) => vec![b],
        None => candidates(),
    };
    let root = paths.llama_dir();
    let mut errors = vec![];
    for b in list {
        match install_release(&root, &tag, b).await {
            Ok(dir) => {
                let Some((rpc, srv)) = find_install(&dir) else {
                    errors.push(format!("{}: archive has no rpc-server/llama-server", b.as_str()));
                    continue;
                };
                match finish(rpc, srv, Some(tag.clone()), b).await {
                    Ok(i) => {
                        info!("llama.cpp {tag} ready ({} backend, {} GPU device(s))", b.as_str(), i.gpus.len());
                        return Ok(i);
                    }
                    Err(e) => {
                        warn!("{} backend does not work here: {e:#}; trying the next one", b.as_str());
                        errors.push(format!("{}: {e:#}", b.as_str()));
                    }
                }
            }
            Err(e) => {
                warn!("installing llama.cpp {tag} ({}) failed: {e:#}", b.as_str());
                errors.push(format!("{}: {e:#}", b.as_str()));
            }
        }
    }
    if let Some((r, s)) = system {
        warn!("falling back to llama.cpp from PATH, which is not the pinned {tag}");
        return finish(r, s, None, Backend::Cpu).await;
    }
    bail!("could not set up llama.cpp: {}", errors.join("; "))
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

async fn wait_port(port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

/// Forward a child's output into `sink`, line by line.
fn pipe_output(child: &mut Child, sink: impl Fn(String) + Send + Sync + Clone + 'static) {
    if let Some(out) = child.stdout.take() {
        let s = sink.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                s(l);
            }
        });
    }
    if let Some(err) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                sink(l);
            }
        });
    }
}

/// Keep a local `rpc-server` running on 127.0.0.1 (never exposed on the network).
pub fn spawn_rpc_server(mesh: Arc<Mesh>, inst: Install) {
    tokio::spawn(async move {
        let mut backoff = Duration::from_secs(1);
        loop {
            let res: Result<()> = async {
                let port = free_port()?;
                let threads = crate::sys::cpu_threads().max(2) / 2;
                let mut c = inst.command(&inst.rpc_server);
                c.args(["-H", "127.0.0.1", "-p", &port.to_string(), "-t", &threads.to_string(), "-c"]);
                c.stdout(Stdio::piped()).stderr(Stdio::piped());
                let mut child = c.spawn().with_context(|| format!("starting {}", inst.rpc_server.display()))?;
                pipe_output(&mut child, |l| debug!("rpc-server: {l}"));
                if !wait_port(port, Duration::from_secs(60)).await {
                    let _ = child.kill().await;
                    bail!("rpc-server did not start listening");
                }
                info!("rpc-server listening on 127.0.0.1:{port}");
                mesh.update_local(|l| {
                    l.rpc_port = Some(port);
                    l.rpc_ready = true;
                });
                let started = Instant::now();
                let status = child.wait().await?;
                mesh.update_local(|l| {
                    l.rpc_ready = false;
                    l.rpc_port = None;
                });
                if started.elapsed() > Duration::from_secs(60) {
                    backoff = Duration::from_secs(1);
                }
                bail!("rpc-server exited with {status}")
            }
            .await;
            if let Err(e) = res {
                warn!("{e:#}; restarting in {}s", backoff.as_secs());
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(60));
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Head controller
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Plan {
    model: String,
    ctx: u32,
    workers: Vec<EndpointId>,
}

#[derive(Debug, Clone, Default)]
pub struct HeadOpts {
    pub embeddings: bool,
}

const STABLE_FOR: Duration = Duration::from_secs(5);

/// Decide who the head is, and (when it's us) run llama-server over the current split.
pub async fn run_controller(mesh: Arc<Mesh>, inst: Install, opts: HeadOpts) {
    let me = mesh.id();
    let mut desired: Option<Plan> = None;
    let mut changed_at = Instant::now();
    let mut running: Option<(Plan, tokio::task::JoinHandle<()>)> = None;
    let mut tunnels: HashMap<EndpointId, (u16, tokio::task::JoinHandle<()>)> = HashMap::new();
    loop {
        let nodes = mesh.online_nodes();
        let tag = state::group_tag(nodes.iter()).unwrap_or_else(|| inst.tag.clone());
        let head = state::elect_head(nodes.iter(), &tag);
        let my_ok = nodes.iter().any(|n| n.id == me && n.can_work(&tag));
        let plan = if head == Some(me) {
            let (model, ctx) = mesh.effective_model().unwrap_or_else(|| (crate::DEFAULT_MODEL.to_string(), mesh.opts.default_ctx));
            let mut workers: Vec<EndpointId> = nodes.iter().filter(|n| n.id != me && n.can_work(&tag)).map(|n| n.id).collect();
            workers.sort();
            Some(Plan { model, ctx, workers })
        } else {
            None
        };
        if plan != desired {
            desired = plan.clone();
            changed_at = Instant::now();
        }
        match &desired {
            None => {
                if let Some((_, h)) = running.take() {
                    info!("no longer the head; stopping llama-server");
                    mesh.log("[llmesh] stepping down: another node is now the head");
                    h.abort();
                }
                for (_, (_, h)) in tunnels.drain() {
                    h.abort();
                }
                let model = mesh.effective_model().map(|m| m.0).unwrap_or_default();
                mesh.update_local(|l| {
                    l.role = if my_ok { Role::Worker } else { Role::Idle };
                    l.head_state = HeadState::Idle;
                    l.model = model;
                    l.model_alias.clear();
                    l.split.clear();
                    l.llama_port = None;
                });
            }
            Some(p) => {
                let up_to_date = running.as_ref().is_some_and(|(rp, _)| rp == p);
                if !up_to_date {
                    if running.is_none() {
                        mesh.update_local(|l| {
                            l.role = Role::Head;
                            if matches!(l.head_state, HeadState::Idle) {
                                l.head_state = HeadState::Loading;
                            }
                        });
                    }
                    if changed_at.elapsed() >= STABLE_FOR {
                        if let Some((_, h)) = running.take() {
                            mesh.log("[llmesh] group changed; restarting llama-server with the new split");
                            h.abort();
                        }
                        // Tunnels for the current workers; drop the rest.
                        tunnels.retain(|id, (_, h)| {
                            let keep = p.workers.contains(id);
                            if !keep {
                                h.abort();
                            }
                            keep
                        });
                        let mut ports = BTreeMap::new();
                        for w in &p.workers {
                            if let Some((port, _)) = tunnels.get(w) {
                                ports.insert(*w, *port);
                                continue;
                            }
                            match net::rpc_tunnel(mesh.clone(), *w).await {
                                Ok((port, h)) => {
                                    tunnels.insert(*w, (port, h));
                                    ports.insert(*w, port);
                                }
                                Err(e) => warn!("tunnel to {}: {e:#}", w.fmt_short()),
                            }
                        }
                        info!("starting head: model {} with {} remote worker(s)", p.model, ports.len());
                        let h = tokio::spawn(serve_head(mesh.clone(), inst.clone(), p.clone(), ports, opts.clone()));
                        running = Some((p.clone(), h));
                    }
                }
            }
        }
        // Created after our own updates above, so they don't wake us straight away.
        let wait = mesh.changed.notified();
        tokio::select! {
            _ = wait => {}
            _ = tokio::time::sleep(Duration::from_secs(1)) => {}
        }
    }
}

/// Download the model and keep llama-server running for one plan. Aborting the task kills it.
async fn serve_head(mesh: Arc<Mesh>, inst: Install, plan: Plan, tunnels: BTreeMap<EndpointId, u16>, opts: HeadOpts) {
    let mref = match ModelRef::parse(&plan.model) {
        Ok(r) => r,
        Err(e) => {
            set_error(&mesh, format!("{e:#}"));
            return;
        }
    };
    let alias = mref.alias();
    mesh.update_local(|l| {
        l.model = plan.model.clone();
        l.model_alias = alias.clone();
        l.split = tunnels.keys().copied().collect();
        l.llama_port = None;
    });

    // 1. Get the GGUF onto this machine.
    let mut backoff = Duration::from_secs(2);
    let model_path = loop {
        mesh.update_local(|l| l.head_state = HeadState::Downloading { done: 0, total: 0, file: alias.clone() });
        let m2 = mesh.clone();
        let progress = move |done: u64, total: u64, file: &str| {
            let mut l = m2.local.lock().unwrap();
            l.head_state = HeadState::Downloading { done, total, file: file.to_string() };
        };
        let res = async {
            let resolved = models::resolve(&mref, &mesh.paths).await?;
            models::ensure_local(&resolved, &mesh.paths, Some(&mesh), &progress).await
        }
        .await;
        match res {
            Ok(p) => break p,
            Err(e) => {
                set_error(&mesh, format!("model download failed: {e:#}"));
                mesh.log(format!("[llmesh] model download failed: {e:#}"));
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(120));
            }
        }
    };

    // 2. Run llama-server, restarting with a smaller context if it keeps failing.
    let mut rpc: Vec<String> = tunnels.values().map(|p| format!("127.0.0.1:{p}")).collect();
    if !inst.has_gpu() && !rpc.is_empty() {
        // No local GPU: put the head's own CPU/RAM into the split through its rpc-server too.
        if let Some(p) = mesh.local.lock().unwrap().rpc_port {
            rpc.push(format!("127.0.0.1:{p}"));
        }
    }
    let embeddings = opts.embeddings || mref.looks_like_embedding();
    let mut attempt = 0u32;
    let mut backoff = Duration::from_secs(2);
    loop {
        let ctx = match attempt {
            0 => plan.ctx,
            1 => (plan.ctx / 2).max(2048),
            _ => (plan.ctx / 4).max(2048),
        };
        let started = Instant::now();
        let res = run_llama_server(&mesh, &inst, &model_path, &alias, ctx, &rpc, embeddings).await;
        let msg = match res {
            Ok(status) => format!("llama-server exited ({status})"),
            Err(e) => format!("{e:#}"),
        };
        warn!("{msg}");
        mesh.log(format!("[llmesh] {msg}"));
        set_error(&mesh, msg);
        if started.elapsed() > Duration::from_secs(300) {
            attempt = 0;
            backoff = Duration::from_secs(2);
        } else {
            attempt += 1;
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(60));
        mesh.update_local(|l| l.head_state = HeadState::Loading);
    }
}

fn set_error(mesh: &Mesh, message: String) {
    mesh.update_local(|l| {
        l.head_state = HeadState::Error { message };
        l.llama_port = None;
    });
}

async fn run_llama_server(
    mesh: &Arc<Mesh>,
    inst: &Install,
    model: &Path,
    alias: &str,
    ctx: u32,
    rpc: &[String],
    embeddings: bool,
) -> Result<std::process::ExitStatus> {
    let port = free_port()?;
    let mut c = inst.command(&inst.llama_server);
    c.arg("-m").arg(model);
    c.args(["--host", "127.0.0.1", "--port", &port.to_string(), "--alias", alias, "--jinja", "--ctx-size", &ctx.to_string(), "--no-webui"]);
    if !rpc.is_empty() {
        c.args(["--rpc", &rpc.join(",")]);
    }
    if embeddings {
        c.arg("--embeddings");
    }
    c.stdout(Stdio::piped()).stderr(Stdio::piped());
    let cmdline = format!("llama-server -m {} --alias {alias} --ctx-size {ctx}{}", model.display(), if rpc.is_empty() { String::new() } else { format!(" --rpc {}", rpc.join(",")) });
    info!("{cmdline}");
    mesh.log(format!("[llmesh] {cmdline}"));
    mesh.update_local(|l| l.head_state = HeadState::Loading);
    let mut child = c.spawn().with_context(|| format!("starting {}", inst.llama_server.display()))?;
    let m = mesh.clone();
    pipe_output(&mut child, move |l| m.log(l));

    let client = reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(5)).build()?;
    let url = format!("http://127.0.0.1:{port}/health");
    let mut ready = false;
    loop {
        tokio::select! {
            status = child.wait() => return Ok(status?),
            _ = tokio::time::sleep(Duration::from_millis(500)) => {}
        }
        let ok = matches!(client.get(&url).send().await, Ok(r) if r.status().is_success());
        if ok && !ready {
            ready = true;
            info!("llama-server ready on 127.0.0.1:{port}");
            mesh.log("[llmesh] model loaded; serving requests");
            mesh.update_local(|l| {
                l.head_state = HeadState::Ready;
                l.llama_port = Some(port);
            });
        }
        if ready {
            // Once up, just wait for it to exit.
            return Ok(child.wait().await?);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_names() {
        let t = "b12017";
        assert_eq!(assets(t, "macos", "aarch64", Backend::Metal).unwrap(), vec!["llama-b12017-bin-macos-arm64.tar.gz"]);
        assert_eq!(
            assets(t, "linux", "x86_64", Backend::Cuda).unwrap(),
            vec!["llama-b12017-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b12017-bin-ubuntu-cuda-12.8-x64.tar.gz"]
        );
        assert_eq!(assets(t, "windows", "x86_64", Backend::Cuda).unwrap()[1], "cudart-llama-bin-win-cuda-12.4-x64.zip");
        assert_eq!(assets(t, "windows", "aarch64", Backend::Cpu).unwrap(), vec!["llama-b12017-bin-win-cpu-arm64.zip"]);
        assert!(assets(t, "macos", "aarch64", Backend::Cuda).is_none());
    }

    #[test]
    fn device_parsing() {
        let out = "ggml_cuda_init: found 1 CUDA devices:\nAvailable devices:\n  CUDA0: NVIDIA GeForce RTX 3060 (12288 MiB, 11234 MiB free)\n  Vulkan0: AMD Radeon (8192 MiB, 8000 MiB free)\n";
        let d = parse_devices(out);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].name, "CUDA0");
        assert_eq!(d[0].description, "NVIDIA GeForce RTX 3060");
        assert_eq!(d[0].total, 12288 << 20);
        assert_eq!(d[1].free, 8000 << 20);
        assert!(parse_devices("Available devices:\n  (none)\n").is_empty());
    }
}
