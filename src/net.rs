//! The peer-to-peer layer: iroh endpoint, authenticated connections, gossip and tunnels.
//!
//! Every byte between peers (gossip, llama.cpp RPC, proxied API requests, model files) travels
//! over iroh QUIC connections. Each connection starts with a mutual group-membership proof; no
//! other stream is served until it succeeds.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use iroh::address_lookup::MemoryLookup;
use iroh::endpoint::{presets, Connection, RecvStream, SendStream};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, SecretKey, TransportAddr};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use crate::config::{Paths, SavedGroup, SavedMember};
use crate::group::{self, AuthHello, GroupSecret, Invite, SignedConfig, Vote};
use crate::state::{GpuDevice, HeadState, NodeInfo, Role};

pub const ALPN: &[u8] = b"llmesh/1";
const MAX_FRAME: usize = 8 << 20;
const GOSSIP_EVERY: Duration = Duration::from_secs(2);
/// A peer is considered online if we got gossip from it this recently.
const ONLINE_WINDOW: Duration = Duration::from_secs(9);
/// Members not heard from for this long are not re-announced in gossip.
const ANNOUNCE_WINDOW_MS: u64 = 5 * 60 * 1000;
/// Saved members not heard from for this long are forgotten.
const FORGET_AFTER_MS: u64 = 14 * 24 * 3600 * 1000;

// ---------------------------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------------------------

/// Write a frame: u32 big-endian length, then JSON.
pub async fn write_frame<W: AsyncWrite + Unpin, T: Serialize>(w: &mut W, v: &T) -> Result<()> {
    let body = serde_json::to_vec(v)?;
    if body.len() > MAX_FRAME {
        bail!("frame too large");
    }
    w.write_all(&(body.len() as u32).to_be_bytes()).await?;
    w.write_all(&body).await?;
    w.flush().await?;
    Ok(())
}

pub async fn read_frame<R: AsyncRead + Unpin, T: DeserializeOwned>(r: &mut R) -> Result<T> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_FRAME {
        bail!("frame too large ({len} bytes)");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).await?;
    Ok(serde_json::from_slice(&buf)?)
}

/// Header frame that starts every stream after authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Header {
    Gossip,
    /// Pipe raw bytes to the receiver's local rpc-server.
    Rpc,
    /// An HTTP/1.1 connection carrying OpenAI API requests.
    Http,
    /// Read a cached model file starting at `offset`.
    File { path: String, offset: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileReply {
    pub ok: bool,
    pub size: u64,
    #[serde(default)]
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMember {
    pub addr: EndpointAddr,
    pub name: String,
    pub seen_ago_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gossip {
    pub info: NodeInfo,
    pub addr: EndpointAddr,
    pub members: Vec<GossipMember>,
    pub config: Option<SignedConfig>,
    pub votes: Vec<Vote>,
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NodeOpts {
    pub name: String,
    pub worker_only: bool,
    pub no_relay: bool,
    pub default_ctx: u32,
    pub started_at: u64,
}

/// The parts of this node's NodeInfo that change at runtime.
#[derive(Debug, Clone, Default)]
pub struct LocalStatus {
    pub llama_tag: String,
    pub backend: String,
    pub rpc_ready: bool,
    pub rpc_port: Option<u16>,
    pub gpus: Vec<GpuDevice>,
    pub role: Role,
    pub head_state: HeadState,
    pub model: String,
    pub model_alias: String,
    pub split: Vec<EndpointId>,
    /// Port of the local llama-server when this node is the head.
    pub llama_port: Option<u16>,
    pub llama_error: Option<String>,
}

#[derive(Debug)]
struct Peer {
    conn: Option<(u64, Connection)>,
    info: Option<NodeInfo>,
    addr: Option<EndpointAddr>,
    last_gossip: Option<Instant>,
    next_dial: Instant,
    backoff: Duration,
    dialing: bool,
}

impl Peer {
    fn new() -> Self {
        Peer { conn: None, info: None, addr: None, last_gossip: None, next_dial: Instant::now(), backoff: Duration::from_secs(1), dialing: false }
    }
    fn live_conn(&self) -> Option<&Connection> {
        self.conn.as_ref().map(|(_, c)| c).filter(|c| c.close_reason().is_none())
    }
    fn online(&self) -> bool {
        self.live_conn().is_some() && self.last_gossip.is_some_and(|t| t.elapsed() < ONLINE_WINDOW) && self.info.is_some()
    }
}

/// Snapshot of a peer for the dashboard and the head controller.
#[derive(Debug, Clone, Serialize)]
pub struct PeerView {
    pub id: EndpointId,
    pub info: Option<NodeInfo>,
    pub online: bool,
    pub connected: bool,
    /// "direct", "relayed" or "" when not connected.
    pub path: String,
    pub rtt_ms: Option<f64>,
    pub remote_addr: String,
}

pub struct Mesh {
    pub endpoint: Endpoint,
    pub key: SecretKey,
    pub secret: GroupSecret,
    pub paths: Paths,
    pub opts: NodeOpts,
    pub group: Mutex<SavedGroup>,
    pub local: Mutex<LocalStatus>,
    pub logs: Mutex<VecDeque<String>>,
    /// Bumped whenever anything the dashboard or the head controller cares about changes.
    pub changed: Notify,
    peers: Mutex<HashMap<EndpointId, Peer>>,
    lookup: MemoryLookup,
    conn_seq: AtomicU64,
    group_dirty: std::sync::atomic::AtomicBool,
    /// Router used to serve HTTP requests that arrive from peers through the `http` tunnel.
    pub peer_http: OnceLock<axum::Router>,
    router: OnceLock<Router>,
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh").field("id", &self.endpoint.id()).finish()
    }
}

#[derive(Debug, Clone)]
struct Handler(Arc<Mesh>);

impl ProtocolHandler for Handler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let mesh = self.0.clone();
        let remote = conn.remote_id();
        match tokio::time::timeout(Duration::from_secs(15), mesh.auth_inbound(&conn)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                info!("rejected connection from {}: {e:#}", remote.fmt_short());
                conn.close(1u32.into(), b"auth failed");
                return Ok(());
            }
            Err(_) => {
                conn.close(1u32.into(), b"auth timeout");
                return Ok(());
            }
        }
        mesh.run_connection(conn).await;
        Ok(())
    }
}

impl Mesh {
    pub async fn bind(key: SecretKey, group: SavedGroup, paths: Paths, opts: NodeOpts, bind_port: Option<u16>) -> Result<Arc<Self>> {
        let lookup = MemoryLookup::new();
        let mut b = if opts.no_relay {
            Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled)
        } else {
            Endpoint::builder(presets::N0)
        };
        b = b.secret_key(key.clone()).alpns(vec![ALPN.to_vec()]).address_lookup(lookup.clone());
        if let Some(port) = bind_port {
            b = b
                .bind_addr(SocketAddr::from(([0, 0, 0, 0], port)))
                .map_err(|e| anyhow!("bad bind address: {e}"))?;
        }
        let endpoint = b.bind().await.context("binding iroh endpoint")?;
        let secret = group.secret();
        let mesh = Arc::new(Mesh {
            endpoint: endpoint.clone(),
            key,
            secret,
            paths,
            opts,
            group: Mutex::new(group),
            local: Mutex::new(LocalStatus::default()),
            logs: Mutex::new(VecDeque::new()),
            changed: Notify::new(),
            peers: Mutex::new(HashMap::new()),
            lookup,
            conn_seq: AtomicU64::new(1),
            group_dirty: Default::default(),
            peer_http: OnceLock::new(),
            router: OnceLock::new(),
        });
        let router = Router::builder(endpoint).accept(ALPN, Handler(mesh.clone())).spawn();
        let _ = mesh.router.set(router);

        // Seed peers from the invite bootstrap list and saved members.
        let seeds: Vec<EndpointAddr> = {
            let g = mesh.group.lock().unwrap();
            g.bootstrap.iter().cloned().chain(g.members.values().map(|m| m.addr.clone())).collect()
        };
        for a in seeds {
            mesh.add_candidate(a);
        }
        Ok(mesh)
    }

    pub fn id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub fn my_addr(&self) -> EndpointAddr {
        self.endpoint.addr()
    }


    pub fn admin(&self) -> Option<EndpointId> {
        self.group.lock().unwrap().admin
    }

    /// True if this node may change the group config (holds the admin key, or LAN group).
    pub fn can_admin(&self) -> bool {
        match self.admin() {
            None => true,
            Some(a) => a == self.id(),
        }
    }

    pub fn notify(&self) {
        self.changed.notify_waiters();
    }

    pub fn log(&self, line: impl Into<String>) {
        let line = line.into();
        let mut l = self.logs.lock().unwrap();
        if l.len() >= 400 {
            l.pop_front();
        }
        l.push_back(line);
    }

    pub fn my_info(&self) -> NodeInfo {
        let l = self.local.lock().unwrap();
        NodeInfo {
            id: self.id(),
            name: self.opts.name.clone(),
            version: crate::VERSION.to_string(),
            llama_tag: l.llama_tag.clone(),
            backend: l.backend.clone(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            started_at: self.opts.started_at,
            worker_only: self.opts.worker_only,
            rpc_ready: l.rpc_ready,
            memory: crate::sys::memory(),
            gpus: l.gpus.clone(),
            role: l.role,
            head_state: l.head_state.clone(),
            model: l.model.clone(),
            model_alias: l.model_alias.clone(),
            split: l.split.clone(),
        }
    }

    pub fn update_local(&self, f: impl FnOnce(&mut LocalStatus)) {
        f(&mut self.local.lock().unwrap());
        self.notify();
    }

    // ----- group config / votes -------------------------------------------------------------

    pub fn config(&self) -> Option<SignedConfig> {
        self.group.lock().unwrap().config.clone()
    }


    /// Merge a config received from anywhere. Returns true if it replaced ours.
    pub fn merge_config(&self, c: SignedConfig) -> bool {
        let mut g = self.group.lock().unwrap();
        let admin = g.admin;
        if group::better_config(&self.secret, admin.as_ref(), g.config.as_ref(), &c) {
            info!("group config updated: seq {} model {} ({:?})", c.config.seq, c.config.active_model, c.config.selection);
            g.config = Some(c);
            drop(g);
            self.mark_dirty();
            true
        } else {
            false
        }
    }

    pub fn merge_votes(&self, votes: Vec<Vote>) -> bool {
        let mut g = self.group.lock().unwrap();
        let mut changed = false;
        for v in votes {
            changed |= group::merge_vote(&mut g.votes, &self.secret, v);
        }
        drop(g);
        if changed {
            self.mark_dirty();
        }
        changed
    }

    /// Sign and apply a config change. Errors unless this node may administer the group.
    pub fn update_config(&self, f: impl FnOnce(&mut group::GroupConfig)) -> Result<SignedConfig> {
        if !self.can_admin() {
            bail!("only the group admin can do that (this node does not hold the admin key)");
        }
        let mut cfg = self.config().map(|c| c.config).unwrap_or_else(|| {
            let mut c = group::GroupConfig::initial(crate::DEFAULT_MODEL);
            c.ctx_size = self.opts.default_ctx;
            c
        });
        f(&mut cfg);
        cfg.seq += 1;
        cfg.ts = group::now_ms().max(cfg.ts + 1);
        let signed = SignedConfig::sign(&self.secret, &self.key, cfg);
        if !self.merge_config(signed.clone()) {
            bail!("config was not accepted");
        }
        self.notify();
        Ok(signed)
    }

    pub fn cast_vote(&self, model_ref: &str) -> Result<Vote> {
        let round = self.config().map(|c| c.config.poll.round).unwrap_or(1);
        let v = Vote::sign(&self.secret, &self.key, round, model_ref);
        self.merge_votes(vec![v.clone()]);
        self.notify();
        Ok(v)
    }

    /// The model the group should run right now (config + votes), if known.
    pub fn effective_model(&self) -> Option<(String, u32)> {
        let g = self.group.lock().unwrap();
        let c = g.config.as_ref()?;
        Some((group::effective_model(&c.config, g.votes.values()), c.config.ctx_size))
    }

    fn mark_dirty(&self) {
        self.group_dirty.store(true, Ordering::Relaxed);
        self.notify();
    }

    pub fn save_group(&self) {
        let g = self.group.lock().unwrap().clone();
        if let Err(e) = g.save(&self.paths) {
            warn!("saving group: {e:#}");
        }
    }

    // ----- peers -----------------------------------------------------------------------------

    /// Learn about a possible member (from an invite, a LAN beacon, gossip or the saved list).
    pub fn add_candidate(&self, addr: EndpointAddr) {
        if addr.id == self.id() {
            return;
        }
        if !addr.addrs.is_empty() {
            self.lookup.add_endpoint_info(addr.clone());
        }
        let mut peers = self.peers.lock().unwrap();
        let p = peers.entry(addr.id).or_insert_with(Peer::new);
        match &mut p.addr {
            Some(a) => a.addrs.extend(addr.addrs),
            None => p.addr = Some(addr),
        }
    }

    pub fn peer_views(&self) -> Vec<PeerView> {
        let peers = self.peers.lock().unwrap();
        let mut out: Vec<PeerView> = peers
            .iter()
            .filter(|(_, p)| p.info.is_some() || p.live_conn().is_some())
            .map(|(id, p)| {
                let (mut path, mut rtt, mut remote) = (String::new(), None, String::new());
                if let Some(c) = p.live_conn() {
                    for pa in c.paths().iter() {
                        if pa.is_selected() {
                            path = if pa.is_relay() { "relayed".into() } else { "direct".into() };
                            rtt = Some(pa.rtt().as_secs_f64() * 1000.0);
                            remote = match pa.remote_addr() {
                                TransportAddr::Ip(a) => a.to_string(),
                                TransportAddr::Relay(u) => u.to_string(),
                                other => format!("{other:?}"),
                            };
                        }
                    }
                }
                PeerView { id: *id, info: p.info.clone(), online: p.online(), connected: p.live_conn().is_some(), path, rtt_ms: rtt, remote_addr: remote }
            })
            .collect();
        out.sort_by_key(|p| p.id);
        out
    }

    /// NodeInfo of every online peer plus ourselves.
    pub fn online_nodes(&self) -> Vec<NodeInfo> {
        let mut v: Vec<NodeInfo> = self.peers.lock().unwrap().values().filter(|p| p.online()).filter_map(|p| p.info.clone()).collect();
        v.push(self.my_info());
        v
    }


    fn conn_for(&self, id: &EndpointId) -> Option<Connection> {
        self.peers.lock().unwrap().get(id).and_then(|p| p.live_conn().cloned())
    }

    /// Open a stream to a connected, authenticated peer and send the header.
    pub async fn open_stream(&self, id: &EndpointId, header: &Header) -> Result<(SendStream, RecvStream)> {
        let conn = self.conn_for(id).ok_or_else(|| anyhow!("not connected to {}", id.fmt_short()))?;
        let (mut send, recv) = conn.open_bi().await?;
        write_frame(&mut send, header).await?;
        Ok((send, recv))
    }

    // ----- connection setup --------------------------------------------------------------------

    async fn auth_inbound(&self, conn: &Connection) -> Result<()> {
        let remote = conn.remote_id();
        let (mut send, mut recv) = conn.accept_bi().await?;
        let hello: AuthHello = read_frame(&mut recv).await?;
        hello.verify(&self.secret, &remote, &self.id())?;
        write_frame(&mut send, &AuthHello::new(&self.secret, &self.id(), &remote)).await?;
        send.finish()?;
        Ok(())
    }

    async fn auth_outbound(&self, conn: &Connection) -> Result<()> {
        let remote = conn.remote_id();
        let (mut send, mut recv) = conn.open_bi().await?;
        write_frame(&mut send, &AuthHello::new(&self.secret, &self.id(), &remote)).await?;
        let reply: AuthHello = read_frame(&mut recv).await.context("peer closed the auth stream (wrong group?)")?;
        reply.verify(&self.secret, &remote, &self.id())?;
        Ok(())
    }

    /// Dial a peer (if not already connected) and run the connection in the background.
    pub async fn dial(self: &Arc<Self>, addr: EndpointAddr) -> Result<()> {
        let id = addr.id;
        let conn = tokio::time::timeout(Duration::from_secs(20), self.endpoint.connect(addr, ALPN))
            .await
            .map_err(|_| anyhow!("connect timed out"))??;
        tokio::time::timeout(Duration::from_secs(15), self.auth_outbound(&conn))
            .await
            .map_err(|_| anyhow!("auth timed out"))??;
        debug!("connected to {}", id.fmt_short());
        let me = self.clone();
        tokio::spawn(async move { me.run_connection(conn).await });
        Ok(())
    }

    /// Register an authenticated connection and serve its incoming streams until it closes.
    async fn run_connection(self: &Arc<Self>, conn: Connection) {
        let remote = conn.remote_id();
        let seq = self.conn_seq.fetch_add(1, Ordering::Relaxed);
        {
            let mut peers = self.peers.lock().unwrap();
            let p = peers.entry(remote).or_insert_with(Peer::new);
            p.conn = Some((seq, conn.clone()));
            p.backoff = Duration::from_secs(1);
        }
        info!("peer {} authenticated", remote.fmt_short());
        // Send gossip right away so the peer learns about us without waiting a tick.
        let me = self.clone();
        tokio::spawn(async move { me.push_gossip(remote).await.ok() });
        loop {
            match conn.accept_bi().await {
                Ok((send, recv)) => {
                    let me = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = me.handle_stream(remote, send, recv).await {
                            debug!("stream from {} ended: {e:#}", remote.fmt_short());
                        }
                    });
                }
                Err(e) => {
                    debug!("connection to {} closed: {e}", remote.fmt_short());
                    break;
                }
            }
        }
        let mut peers = self.peers.lock().unwrap();
        if let Some(p) = peers.get_mut(&remote) {
            if p.conn.as_ref().is_some_and(|(s, _)| *s == seq) {
                p.conn = None;
                p.last_gossip = None;
            }
        }
        drop(peers);
        info!("peer {} disconnected", remote.fmt_short());
        self.notify();
    }

    async fn handle_stream(self: &Arc<Self>, remote: EndpointId, send: SendStream, mut recv: RecvStream) -> Result<()> {
        let header: Header = read_frame(&mut recv).await?;
        match header {
            Header::Gossip => {
                let g: Gossip = read_frame(&mut recv).await?;
                drop(send);
                self.on_gossip(remote, g);
            }
            Header::Rpc => {
                let port = self.local.lock().unwrap().rpc_port.ok_or_else(|| anyhow!("rpc-server not running"))?;
                let mut tcp = tokio::net::TcpStream::connect(("127.0.0.1", port)).await?;
                tcp.set_nodelay(true)?;
                let mut s = tokio::io::join(recv, send);
                let _ = tokio::io::copy_bidirectional(&mut s, &mut tcp).await;
            }
            Header::Http => {
                let router = self.peer_http.get().cloned().ok_or_else(|| anyhow!("api not ready"))?;
                let io = hyper_util::rt::TokioIo::new(tokio::io::join(recv, send));
                let svc = hyper_util::service::TowerToHyperService::new(router);
                hyper::server::conn::http1::Builder::new().serve_connection(io, svc).await?;
            }
            Header::File { path, offset } => self.serve_file(send, &path, offset).await?,
        }
        Ok(())
    }

    async fn serve_file(&self, mut send: SendStream, rel: &str, offset: u64) -> Result<()> {
        let file = match crate::models::cache_path_for(&self.paths, rel) {
            Some(f) if f.is_file() => f,
            _ => {
                write_frame(&mut send, &FileReply { ok: false, size: 0, error: "not cached here".into() }).await?;
                send.finish()?;
                return Ok(());
            }
        };
        let mut f = tokio::fs::File::open(&file).await?;
        let size = f.metadata().await?.len();
        write_frame(&mut send, &FileReply { ok: true, size, error: String::new() }).await?;
        if offset < size {
            use tokio::io::AsyncSeekExt;
            f.seek(std::io::SeekFrom::Start(offset)).await?;
            tokio::io::copy(&mut f, &mut send).await?;
        }
        send.finish()?;
        Ok(())
    }

    /// Ask a peer for a cached model file; returns a reader positioned at `offset`.
    pub async fn fetch_file(&self, id: &EndpointId, rel: &str, offset: u64) -> Result<(u64, RecvStream)> {
        let (send, mut recv) = self.open_stream(id, &Header::File { path: rel.to_string(), offset }).await?;
        drop(send);
        let reply: FileReply = read_frame(&mut recv).await?;
        if !reply.ok {
            bail!("{}", reply.error);
        }
        Ok((reply.size, recv))
    }

    // ----- gossip ------------------------------------------------------------------------------

    fn gossip_message(&self) -> Gossip {
        let now = group::now_ms();
        let g = self.group.lock().unwrap();
        let round = g.config.as_ref().map(|c| c.config.poll.round).unwrap_or(0);
        let members = g
            .members
            .values()
            .filter(|m| now.saturating_sub(m.last_seen) < ANNOUNCE_WINDOW_MS)
            .map(|m| GossipMember { addr: m.addr.clone(), name: m.name.clone(), seen_ago_ms: now.saturating_sub(m.last_seen) })
            .collect();
        Gossip {
            info: self.my_info(),
            addr: self.my_addr(),
            members,
            config: g.config.clone(),
            votes: g.votes.values().filter(|v| v.round == round).cloned().collect(),
        }
    }

    async fn push_gossip(&self, to: EndpointId) -> Result<()> {
        let msg = self.gossip_message();
        let (mut send, _recv) = self.open_stream(&to, &Header::Gossip).await?;
        write_frame(&mut send, &msg).await?;
        send.finish()?;
        Ok(())
    }

    fn on_gossip(&self, from: EndpointId, g: Gossip) {
        if g.info.id != from || g.addr.id != from {
            warn!("peer {} sent gossip for someone else; ignoring", from.fmt_short());
            return;
        }
        let now = group::now_ms();
        let mut changed = false;
        {
            let mut peers = self.peers.lock().unwrap();
            let p = peers.entry(from).or_insert_with(Peer::new);
            let was_online = p.online();
            changed |= p.info.as_ref() != Some(&g.info);
            p.info = Some(g.info.clone());
            p.addr = Some(g.addr.clone());
            p.last_gossip = Some(Instant::now());
            changed |= !was_online;
        }
        if !g.addr.addrs.is_empty() {
            self.lookup.add_endpoint_info(g.addr.clone());
        }
        {
            let mut grp = self.group.lock().unwrap();
            grp.members.insert(from, SavedMember { addr: g.addr.clone(), name: g.info.name.clone(), last_seen: now });
            for m in &g.members {
                if m.addr.id == self.id() || m.seen_ago_ms > ANNOUNCE_WINDOW_MS {
                    continue;
                }
                let seen = now.saturating_sub(m.seen_ago_ms);
                let entry = grp.members.get(&m.addr.id);
                if entry.is_none_or(|e| e.last_seen < seen) {
                    grp.members.insert(m.addr.id, SavedMember { addr: m.addr.clone(), name: m.name.clone(), last_seen: seen });
                }
            }
        }
        // Make sure we know how to reach freshly-seen members (they may not be connected to us).
        for m in &g.members {
            if m.seen_ago_ms < 60_000 {
                self.add_candidate(m.addr.clone());
            }
        }
        self.group_dirty.store(true, Ordering::Relaxed);
        if let Some(c) = g.config {
            changed |= self.merge_config(c);
        }
        changed |= self.merge_votes(g.votes);
        if changed {
            self.notify();
        }
    }

    /// Spawn the background loops: gossip, dialing, persistence.
    pub fn spawn_tasks(self: &Arc<Self>) {
        let me = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(GOSSIP_EVERY);
            loop {
                tick.tick().await;
                let targets: Vec<EndpointId> = me.peers.lock().unwrap().iter().filter(|(_, p)| p.live_conn().is_some()).map(|(id, _)| *id).collect();
                for t in targets {
                    let m = me.clone();
                    tokio::spawn(async move {
                        if let Err(e) = tokio::time::timeout(Duration::from_secs(6), m.push_gossip(t)).await {
                            debug!("gossip to {} timed out: {e}", t.fmt_short());
                        }
                    });
                }
                me.notify();
            }
        });

        let me = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(700));
            loop {
                tick.tick().await;
                me.dial_due();
            }
        });

        let me = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tick.tick().await;
                if me.group_dirty.swap(false, Ordering::Relaxed) {
                    let now = group::now_ms();
                    me.group.lock().unwrap().members.retain(|_, m| now.saturating_sub(m.last_seen) < FORGET_AFTER_MS);
                    me.save_group();
                }
            }
        });
    }

    fn dial_due(self: &Arc<Self>) {
        let now = Instant::now();
        let saved_seen: BTreeMap<EndpointId, u64> = self.group.lock().unwrap().members.iter().map(|(k, v)| (*k, v.last_seen)).collect();
        let mut due = vec![];
        {
            let mut peers = self.peers.lock().unwrap();
            for (id, p) in peers.iter_mut() {
                if p.dialing || p.live_conn().is_some() || p.next_dial > now {
                    continue;
                }
                let Some(addr) = p.addr.clone() else { continue };
                p.dialing = true;
                due.push((*id, addr));
            }
        }
        for (id, addr) in due {
            let me = self.clone();
            let stale = saved_seen.get(&id).is_some_and(|s| group::now_ms().saturating_sub(*s) > 10 * 60 * 1000);
            tokio::spawn(async move {
                let res = me.dial(addr).await;
                let mut peers = me.peers.lock().unwrap();
                if let Some(p) = peers.get_mut(&id) {
                    p.dialing = false;
                    if let Err(e) = res {
                        debug!("dial {} failed: {e:#}", id.fmt_short());
                        let cap = if stale { Duration::from_secs(300) } else { Duration::from_secs(30) };
                        p.backoff = (p.backoff * 2).min(cap);
                        p.next_dial = Instant::now() + p.backoff;
                    }
                }
            });
        }
    }

    /// Build an invite with this node and up to three online members as bootstrap peers.
    pub fn invite(&self) -> Option<String> {
        let g = self.group.lock().unwrap();
        let admin = g.admin?;
        let name = g.name.clone();
        drop(g);
        let mut boots = vec![self.my_addr()];
        let peers = self.peers.lock().unwrap();
        for p in peers.values().filter(|p| p.online()) {
            if boots.len() >= Invite::MAX_BOOTSTRAP {
                break;
            }
            if let Some(a) = &p.addr {
                boots.push(a.clone());
            }
        }
        Some(Invite { secret: self.secret, admin, bootstrap: boots, name }.encode())
    }

    pub async fn shutdown(&self) {
        if let Some(r) = self.router.get() {
            let _ = r.shutdown().await;
        }
    }
}

/// Listen on 127.0.0.1:ephemeral and forward each TCP connection to `peer`'s rpc-server.
pub async fn rpc_tunnel(mesh: Arc<Mesh>, peer: EndpointId) -> Result<(u16, tokio::task::JoinHandle<()>)> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let h = tokio::spawn(async move {
        loop {
            let Ok((mut tcp, _)) = listener.accept().await else { break };
            let _ = tcp.set_nodelay(true);
            let mesh = mesh.clone();
            tokio::spawn(async move {
                match mesh.open_stream(&peer, &Header::Rpc).await {
                    Ok((send, recv)) => {
                        let mut s = tokio::io::join(recv, send);
                        let _ = tokio::io::copy_bidirectional(&mut tcp, &mut s).await;
                    }
                    Err(e) => warn!("rpc tunnel to {}: {e:#}", peer.fmt_short()),
                }
            });
        }
    });
    Ok((port, h))
}
