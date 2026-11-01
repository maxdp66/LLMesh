//! The local HTTP server: OpenAI-compatible proxy, dashboard, and `/mesh/api`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hyper_util::rt::TokioIo;
use iroh::EndpointId;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::group::{self, Selection};
use crate::models::{self, ModelRef};
use crate::net::{Header, Mesh};
use crate::state::{self, HeadState, NodeInfo};

const HOP_HEADER: &str = "x-llmesh-hops";
const MAX_HOPS: u32 = 3;
const MAX_BODY: usize = 64 << 20;
const WAIT_FOR_HEAD: Duration = Duration::from_secs(15 * 60);
const DASHBOARD: &str = include_str!("dashboard.html");

type AppState = Arc<Mesh>;

pub fn oai_error(status: StatusCode, message: impl Into<String>, kind: &str, code: Option<&str>) -> Response {
    let body = json!({"error": {"message": message.into(), "type": kind, "param": null, "code": code}});
    (status, Json(body)).into_response()
}

/// Router for requests arriving from peers through the `http` tunnel: only the OpenAI API.
pub fn peer_router(mesh: Arc<Mesh>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/{*rest}", axum::routing::any(proxy_handler))
        .with_state(mesh)
}

pub fn local_router(mesh: Arc<Mesh>) -> Router {
    Router::new()
        .route("/", get(|| async { Html(DASHBOARD) }))
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{*id}", get(get_model))
        .route("/v1/{*rest}", axum::routing::any(proxy_handler))
        .route("/mesh/api/status", get(status))
        .route("/mesh/api/logs", get(logs))
        .route("/mesh/api/hf/search", get(hf_search))
        .route("/mesh/api/hf/files", get(hf_files))
        .route("/mesh/api/model", post(set_model))
        .route("/mesh/api/selection", post(set_selection))
        .route("/mesh/api/vote", post(vote))
        .route("/mesh/api/round", post(new_round))
        .route("/mesh/api/ctx", post(set_ctx))
        .with_state(mesh)
}

/// Bind the API on `port` (or the next free port) and serve forever. Returns the bound port.
pub async fn serve(mesh: Arc<Mesh>, port: u16, public: bool) -> Result<u16> {
    let ip = if public { [0, 0, 0, 0] } else { [127, 0, 0, 1] };
    let mut last_err = None;
    for p in port..port.saturating_add(20) {
        match tokio::net::TcpListener::bind(SocketAddr::from((ip, p))).await {
            Ok(l) => {
                let app = local_router(mesh.clone());
                tokio::spawn(async move {
                    if let Err(e) = axum::serve(l, app.into_make_service_with_connect_info::<SocketAddr>()).await {
                        warn!("api server stopped: {e}");
                    }
                });
                if p != port {
                    warn!("port {port} is taken; API is on port {p} instead");
                }
                return Ok(p);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(anyhow::anyhow!("no free port from {port}: {:?}", last_err))
}

// ---------------------------------------------------------------------------------------------
// Group view helpers
// ---------------------------------------------------------------------------------------------

struct View {
    nodes: Vec<NodeInfo>,
    tag: String,
    head: Option<EndpointId>,
}

fn view(mesh: &Mesh) -> View {
    let nodes = mesh.online_nodes();
    let tag = state::group_tag(nodes.iter()).unwrap_or_else(|| mesh.local.lock().unwrap().llama_tag.clone());
    let head = state::elect_head(nodes.iter(), &tag);
    View { nodes, tag, head }
}

/// The model name clients should see right now.
fn active_alias(mesh: &Mesh) -> Option<String> {
    let v = view(mesh);
    if let Some(h) = v.head {
        if let Some(n) = v.nodes.iter().find(|n| n.id == h) {
            if !n.model_alias.is_empty() {
                return Some(n.model_alias.clone());
            }
        }
    }
    let m = mesh.effective_model().map(|m| m.0).unwrap_or_else(|| crate::DEFAULT_MODEL.to_string());
    ModelRef::parse(&m).ok().map(|r| r.alias())
}

fn capacity(v: &View) -> u64 {
    v.nodes.iter().filter(|n| n.can_work(&v.tag)).map(|n| n.capacity()).sum()
}

// ---------------------------------------------------------------------------------------------
// OpenAI API
// ---------------------------------------------------------------------------------------------

fn model_obj(id: &str) -> Value {
    json!({"id": id, "object": "model", "created": 0, "owned_by": "llmesh"})
}

async fn list_models(State(mesh): State<AppState>) -> Response {
    let data: Vec<Value> = active_alias(&mesh).iter().map(|a| model_obj(a)).collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

async fn get_model(State(mesh): State<AppState>, axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    // Any model name works; report it as available.
    let _ = &mesh;
    Json(model_obj(&id)).into_response()
}

enum Target {
    Local(u16, String),
    Peer(EndpointId, String),
}

/// Find where requests should go, waiting (up to 15 minutes) while the head loads.
#[allow(clippy::result_large_err)]
async fn find_target(mesh: &Arc<Mesh>) -> Result<Target, Response> {
    let deadline = Instant::now() + WAIT_FOR_HEAD;
    let mut last_state = String::new();
    loop {
        let notified = mesh.changed.notified();
        let v = view(mesh);
        match v.head {
            Some(h) if h == mesh.id() => {
                let l = mesh.local.lock().unwrap();
                if let (HeadState::Ready, Some(port)) = (&l.head_state, l.llama_port) {
                    return Ok(Target::Local(port, l.model_alias.clone()));
                }
                last_state = describe(&l.head_state);
            }
            Some(h) => {
                if let Some(n) = v.nodes.iter().find(|n| n.id == h) {
                    if n.head_state == HeadState::Ready {
                        return Ok(Target::Peer(h, n.model_alias.clone()));
                    }
                    last_state = describe(&n.head_state);
                }
            }
            None => last_state = "no node is ready to run the model yet (llama.cpp still starting?)".into(),
        }
        if Instant::now() >= deadline {
            return Err(oai_error(StatusCode::SERVICE_UNAVAILABLE, format!("the group's model is not ready: {last_state}"), "server_error", Some("model_not_ready")));
        }
        tokio::select! {
            _ = notified => {}
            _ = tokio::time::sleep(Duration::from_millis(500)) => {}
        }
    }
}

fn describe(s: &HeadState) -> String {
    match s {
        HeadState::Idle => "idle".into(),
        HeadState::Downloading { done, total, file } => format!("downloading {file} ({} of {})", crate::sys::human_bytes(*done), crate::sys::human_bytes(*total)),
        HeadState::Loading => "loading the model".into(),
        HeadState::Ready => "ready".into(),
        HeadState::Error { message } => format!("error: {message}"),
    }
}

async fn proxy_handler(State(mesh): State<AppState>, req: Request) -> Response {
    match proxy(mesh, req).await {
        Ok(r) => r,
        Err(r) => r,
    }
}

#[allow(clippy::result_large_err)]
async fn proxy(mesh: Arc<Mesh>, req: Request) -> Result<Response, Response> {
    let hops: u32 = req.headers().get(HOP_HEADER).and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).unwrap_or(0);
    if hops >= MAX_HOPS {
        return Err(oai_error(StatusCode::LOOP_DETECTED, "request bounced between nodes; the group is still agreeing on a head, retry shortly", "server_error", Some("routing_loop")));
    }
    let (parts, body) = req.into_parts();
    let body = axum::body::to_bytes(body, MAX_BODY)
        .await
        .map_err(|e| oai_error(StatusCode::PAYLOAD_TOO_LARGE, format!("request body: {e}"), "invalid_request_error", None))?;

    let target = find_target(&mesh).await?;
    let alias = match &target {
        Target::Local(_, a) | Target::Peer(_, a) => a.clone(),
    };
    let body = rewrite_model(&parts.headers, body, &alias);

    let path = parts.uri.path_and_query().map(|p| p.as_str()).unwrap_or("/").to_string();
    let mut b = hyper::Request::builder().method(parts.method.clone()).uri(&path);
    for (k, v) in parts.headers.iter() {
        if matches!(k.as_str(), "host" | "authorization" | "content-length" | "connection" | "transfer-encoding" | "keep-alive" | HOP_HEADER) {
            continue;
        }
        b = b.header(k, v);
    }
    b = b.header(header::HOST, "localhost").header(HOP_HEADER, (hops + 1).to_string()).header(header::CONTENT_LENGTH, body.len());
    let out_req = b
        .body(http_body_util::Full::new(body))
        .map_err(|e| oai_error(StatusCode::BAD_REQUEST, e.to_string(), "invalid_request_error", None))?;

    let resp = match target {
        Target::Local(port, _) => {
            let tcp = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.map_err(|e| upstream_err(format!("llama-server: {e}")))?;
            let _ = tcp.set_nodelay(true);
            send(tcp, out_req).await
        }
        Target::Peer(id, _) => {
            debug!("forwarding {} to head {}", path, id.fmt_short());
            let (send_s, recv_s) = mesh.open_stream(&id, &Header::Http).await.map_err(|e| upstream_err(format!("head unreachable: {e:#}")))?;
            send(tokio::io::join(recv_s, send_s), out_req).await
        }
    }
    .map_err(|e| upstream_err(format!("{e:#}")))?;

    let (rparts, rbody) = resp.into_parts();
    let mut out = Response::builder().status(rparts.status);
    for (k, v) in rparts.headers.iter() {
        if matches!(k.as_str(), "connection" | "transfer-encoding" | "keep-alive") {
            continue;
        }
        out = out.header(k, v);
    }
    if rparts.headers.get(header::CONTENT_TYPE).is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream")) {
        out = out.header("x-accel-buffering", "no");
    }
    // Stream the body through as it arrives (SSE included): no buffering.
    Ok(out.body(Body::new(rbody)).unwrap_or_else(|_| upstream_err("bad upstream response".into())))
}

fn upstream_err(msg: String) -> Response {
    oai_error(StatusCode::BAD_GATEWAY, msg, "server_error", Some("upstream_error"))
}

async fn send<IO>(io: IO, req: hyper::Request<http_body_util::Full<Bytes>>) -> Result<hyper::Response<hyper::body::Incoming>>
where
    IO: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(io)).await?;
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            debug!("upstream connection: {e}");
        }
    });
    Ok(sender.send_request(req).await?)
}

/// Any `model` name works: rewrite it to the alias the head's llama-server uses.
fn rewrite_model(headers: &HeaderMap, body: Bytes, alias: &str) -> Bytes {
    let is_json = headers.get(header::CONTENT_TYPE).is_none_or(|v| v.as_bytes().starts_with(b"application/json"));
    if !is_json || body.is_empty() || alias.is_empty() {
        return body;
    }
    match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(mut m)) => {
            m.insert("model".into(), Value::String(alias.to_string()));
            serde_json::to_vec(&m).map(Bytes::from).unwrap_or(body)
        }
        _ => body,
    }
}

// ---------------------------------------------------------------------------------------------
// /mesh/api
// ---------------------------------------------------------------------------------------------

fn is_local_host(headers: &HeaderMap) -> bool {
    // Guards against DNS rebinding: a web page can't make the browser send `Host: localhost`.
    let Some(h) = headers.get(header::HOST).and_then(|h| h.to_str().ok()) else { return true };
    let host = if h.starts_with('[') { h.split(']').next().unwrap_or("").trim_start_matches('[') } else { h.split(':').next().unwrap_or("") };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

fn local_read(addr: &SocketAddr, headers: &HeaderMap) -> bool {
    addr.ip().is_loopback() && is_local_host(headers)
}

/// Mutating endpoints: loopback only, plus a custom header that a cross-site form can't send.
#[allow(clippy::result_large_err)]
fn check_mutation(addr: &SocketAddr, headers: &HeaderMap) -> Result<(), Response> {
    if !local_read(addr, headers) {
        return Err(oai_error(StatusCode::FORBIDDEN, "only allowed from this machine", "permission_error", None));
    }
    if headers.get("x-llmesh").map(|v| v.as_bytes()) != Some(b"1") {
        return Err(oai_error(StatusCode::FORBIDDEN, "missing X-Llmesh: 1 header", "permission_error", None));
    }
    Ok(())
}

async fn status(State(mesh): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap) -> Response {
    let v = view(&mesh);
    let local = local_read(&addr, &headers);
    let me = mesh.my_info();
    let peers = mesh.peer_views();
    let (cfg, votes, gname, admin) = {
        let g = mesh.group.lock().unwrap();
        (g.config.clone(), g.votes.values().cloned().collect::<Vec<_>>(), g.name.clone(), g.admin)
    };
    let round = cfg.as_ref().map(|c| c.config.poll.round).unwrap_or(1);
    let tally = group::tally(votes.iter(), round);
    let my_vote = votes.iter().find(|x| x.voter_id == mesh.id() && x.round == round).map(|x| x.model_ref.clone());
    let effective = mesh.effective_model().map(|m| m.0).unwrap_or_else(|| crate::DEFAULT_MODEL.to_string());
    let local_status = mesh.local.lock().unwrap().clone();
    Json(json!({
        "id": mesh.id(),
        "version": crate::VERSION,
        "me": me,
        "group": {
            "name": gname,
            "tag": mesh.secret.tag_hex(),
            "is_lan": admin.is_none(),
            "admin": admin,
            "can_admin": mesh.can_admin(),
        },
        "invite": if local { mesh.invite() } else { None },
        "local_view": local,
        "llama_tag": v.tag,
        "head": v.head,
        "peers": peers,
        "config": cfg.map(|c| c.config),
        "effective_model": effective,
        "active_alias": active_alias(&mesh),
        "tally": tally.iter().map(|(m, n)| json!({"model": m, "votes": n})).collect::<Vec<_>>(),
        "round": round,
        "my_vote": my_vote,
        "capacity": capacity(&v),
        "llama_error": local_status.llama_error,
        "addr": mesh.my_addr(),
    }))
    .into_response()
}

async fn logs(State(mesh): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap) -> Response {
    if !local_read(&addr, &headers) {
        return oai_error(StatusCode::FORBIDDEN, "only allowed from this machine", "permission_error", None);
    }
    let lines: Vec<String> = mesh.logs.lock().unwrap().iter().cloned().collect();
    Json(json!({"lines": lines})).into_response()
}

#[derive(Deserialize)]
struct SearchQ {
    q: String,
}

async fn hf_search(ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Query(q): Query<SearchQ>) -> Response {
    if !local_read(&addr, &headers) {
        return oai_error(StatusCode::FORBIDDEN, "only allowed from this machine", "permission_error", None);
    }
    match models::hf_search(&q.q).await {
        Ok(r) => Json(r).into_response(),
        Err(e) => oai_error(StatusCode::BAD_GATEWAY, format!("{e:#}"), "server_error", None),
    }
}

#[derive(Deserialize)]
struct FilesQ {
    repo: String,
}

async fn hf_files(State(mesh): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Query(q): Query<FilesQ>) -> Response {
    if !local_read(&addr, &headers) {
        return oai_error(StatusCode::FORBIDDEN, "only allowed from this machine", "permission_error", None);
    }
    let cap = capacity(&view(&mesh));
    match models::hf_files(&q.repo).await {
        Ok(files) => {
            let groups: Vec<Value> = models::group_quants(&q.repo, &files)
                .into_iter()
                .map(|g| {
                    let fit = models::fit_label(g.size, cap);
                    json!({"quant": g.quant, "model_ref": g.model_ref, "size": g.size, "files": g.files, "fit": fit})
                })
                .collect();
            Json(json!({"repo": q.repo, "capacity": cap, "quants": groups})).into_response()
        }
        Err(e) => oai_error(StatusCode::BAD_GATEWAY, format!("{e:#}"), "server_error", None),
    }
}

#[derive(Deserialize)]
struct ModelBody {
    model: String,
}

fn result_json<T: serde::Serialize>(r: Result<T>) -> Response {
    match r {
        Ok(v) => Json(json!({"ok": true, "result": v})).into_response(),
        Err(e) => oai_error(StatusCode::BAD_REQUEST, format!("{e:#}"), "invalid_request_error", None),
    }
}

async fn set_model(State(mesh): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(b): Json<ModelBody>) -> Response {
    if let Err(r) = check_mutation(&addr, &headers) {
        return r;
    }
    if let Err(e) = ModelRef::parse(&b.model) {
        return oai_error(StatusCode::BAD_REQUEST, format!("{e:#}"), "invalid_request_error", None);
    }
    info!("setting group model to {}", b.model);
    result_json(mesh.update_config(|c| c.active_model = b.model.trim().to_string()).map(|c| c.config))
}

#[derive(Deserialize)]
struct SelectionBody {
    mode: Selection,
}

async fn set_selection(State(mesh): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(b): Json<SelectionBody>) -> Response {
    if let Err(r) = check_mutation(&addr, &headers) {
        return r;
    }
    result_json(mesh.update_config(|c| c.selection = b.mode).map(|c| c.config))
}

async fn vote(State(mesh): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(b): Json<ModelBody>) -> Response {
    if let Err(r) = check_mutation(&addr, &headers) {
        return r;
    }
    if let Err(e) = ModelRef::parse(&b.model) {
        return oai_error(StatusCode::BAD_REQUEST, format!("{e:#}"), "invalid_request_error", None);
    }
    result_json(mesh.cast_vote(b.model.trim()))
}

#[derive(Deserialize)]
struct RoundBody {
    #[serde(default)]
    candidates: Vec<String>,
}

async fn new_round(State(mesh): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(b): Json<RoundBody>) -> Response {
    if let Err(r) = check_mutation(&addr, &headers) {
        return r;
    }
    let cands: Vec<String> = b.candidates.into_iter().map(|c| c.trim().to_string()).filter(|c| ModelRef::parse(c).is_ok()).collect();
    // Keep the model that's running now as the fallback so a new round doesn't cause a switch
    // until votes come in.
    let current = mesh.effective_model().map(|m| m.0);
    result_json(
        mesh.update_config(|c| {
            if let Some(cur) = current {
                c.active_model = cur;
            }
            c.poll.round += 1;
            c.poll.candidates = cands;
        })
        .map(|c| c.config),
    )
}

#[derive(Deserialize)]
struct CtxBody {
    ctx: u32,
}

async fn set_ctx(State(mesh): State<AppState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(b): Json<CtxBody>) -> Response {
    if let Err(r) = check_mutation(&addr, &headers) {
        return r;
    }
    if !(512..=1_048_576).contains(&b.ctx) {
        return oai_error(StatusCode::BAD_REQUEST, "ctx must be between 512 and 1048576", "invalid_request_error", None);
    }
    result_json(mesh.update_config(|c| c.ctx_size = b.ctx).map(|c| c.config))
}
