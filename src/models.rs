//! Model references, Hugging Face search/resolution, and resumable downloads
//! (from Hugging Face, any https URL, or from a peer's cache as a fallback).

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{info, warn};

use crate::net::Mesh;

pub const DEFAULT_QUANT: &str = "Q4_K_M";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    /// `owner/repo`, `owner/repo:QUANT` or `owner/repo/path/file.gguf`.
    Hf { repo: String, quant: Option<String>, file: Option<String> },
    Local(PathBuf),
    Url(String),
}

fn looks_local(s: &str) -> bool {
    s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("../")
        || s.starts_with('~')
        || s.starts_with(".\\")
        || s.starts_with("\\\\")
        || (s.len() > 2 && s.as_bytes()[1] == b':' && (s.as_bytes()[2] == b'\\' || s.as_bytes()[2] == b'/'))
}

impl ModelRef {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            bail!("empty model reference");
        }
        if s.starts_with("https://") || s.starts_with("http://") {
            return Ok(ModelRef::Url(s.to_string()));
        }
        if looks_local(s) || Path::new(s).is_file() {
            let p = if let Some(rest) = s.strip_prefix("~/") {
                dirs::home_dir().map(|h| h.join(rest)).unwrap_or_else(|| PathBuf::from(s))
            } else {
                PathBuf::from(s)
            };
            return Ok(ModelRef::Local(p));
        }
        let (path, quant) = match s.rsplit_once(':') {
            Some((p, q)) if !q.is_empty() && !q.contains('/') => (p, Some(q.to_string())),
            _ => (s, None),
        };
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 2 || parts.iter().any(|p| p.is_empty() || *p == "." || *p == "..") {
            bail!("can't understand model reference {s:?} (expected owner/repo, owner/repo:QUANT, owner/repo/file.gguf, a local path or an https URL)");
        }
        let repo = format!("{}/{}", parts[0], parts[1]);
        if parts.len() > 2 {
            let file = parts[2..].join("/");
            if !file.to_ascii_lowercase().ends_with(".gguf") {
                bail!("{file:?} is not a .gguf file");
            }
            if quant.is_some() {
                bail!("use either owner/repo:QUANT or owner/repo/file.gguf, not both");
            }
            return Ok(ModelRef::Hf { repo, quant: None, file: Some(file) });
        }
        Ok(ModelRef::Hf { repo, quant, file: None })
    }

    /// The name clients see in `/v1/models` and responses.
    pub fn alias(&self) -> String {
        match self {
            ModelRef::Hf { repo, quant, file: None } => format!("{repo}:{}", quant.as_deref().unwrap_or(DEFAULT_QUANT)),
            ModelRef::Hf { repo, file: Some(f), .. } => format!("{repo}/{f}"),
            ModelRef::Local(p) => p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "local".into()),
            ModelRef::Url(u) => u.rsplit('/').next().unwrap_or("model").trim_end_matches(".gguf").to_string(),
        }
    }

    /// Does this look like an embedding model? (llama-server needs `--embeddings` for those.)
    pub fn looks_like_embedding(&self) -> bool {
        let a = self.alias().to_ascii_lowercase();
        ["embed", "bge-", "e5-", "gte-", "minilm"].iter().any(|k| a.contains(k))
    }
}

/// Split GGUF shard info: `name-00001-of-00003.gguf` -> (prefix "name", 1, 3).
pub fn shard_info(file: &str) -> Option<(String, u32, u32)> {
    let lower = file.to_ascii_lowercase();
    let stem = lower.strip_suffix(".gguf")?;
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (prefix, idx) = rest.rsplit_once('-')?;
    if idx.len() != 5 || total.len() != 5 {
        return None;
    }
    let (i, t) = (idx.parse().ok()?, total.parse().ok()?);
    Some((file[..prefix.len()].to_string(), i, t))
}

pub fn shard_names(prefix: &str, total: u32) -> Vec<String> {
    (1..=total).map(|i| format!("{prefix}-{i:05}-of-{total:05}.gguf")).collect()
}

fn is_quant_token(t: &str) -> bool {
    let u = t.to_ascii_uppercase();
    if matches!(u.as_str(), "F16" | "F32" | "BF16" | "FP16" | "FP32") || u.starts_with("MXFP4") {
        return true;
    }
    for p in ["IQ", "TQ", "Q"] {
        if let Some(rest) = u.strip_prefix(p) {
            return rest.chars().next().is_some_and(|c| c.is_ascii_digit());
        }
    }
    false
}

/// The quantization label of a GGUF file name, e.g. `Qwen3-8B-Q4_K_M.gguf` -> `Q4_K_M`.
pub fn quant_of(file: &str) -> String {
    let name = file.rsplit('/').next().unwrap_or(file);
    let base = match shard_info(name) {
        Some((prefix, _, _)) => prefix,
        None => name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF")).unwrap_or(name).to_string(),
    };
    let tokens: Vec<&str> = base.split(['-', '.']).collect();
    for (i, t) in tokens.iter().enumerate().rev() {
        if is_quant_token(t) {
            let q = t.to_ascii_uppercase();
            if i > 0 && tokens[i - 1].eq_ignore_ascii_case("UD") {
                return format!("UD-{q}");
            }
            return q;
        }
    }
    // Fall back to a directory name like `Q4_K_M/` used by some repos.
    if let Some(dir) = file.rsplit('/').nth(1) {
        if is_quant_token(dir) {
            return dir.to_ascii_uppercase();
        }
    }
    "unknown".into()
}

// ---------------------------------------------------------------------------------------------
// Hugging Face API
// ---------------------------------------------------------------------------------------------

pub fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

pub fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(concat!("llmesh/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(20))
        .build()
        .expect("http client")
}

fn with_token(rb: reqwest::RequestBuilder, url: &str) -> reqwest::RequestBuilder {
    match std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()) {
        Some(t) if url.starts_with(&hf_endpoint()) => rb.bearer_auth(t),
        _ => rb,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModel {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
}

pub async fn hf_search(query: &str) -> Result<Vec<HfModel>> {
    let url = format!(
        "{}/api/models?search={}&filter=gguf&sort=downloads&limit=20",
        hf_endpoint(),
        urlencoding::encode(query)
    );
    let r = with_token(http_client().get(&url), &url).timeout(Duration::from_secs(30)).send().await?;
    if !r.status().is_success() {
        bail!("Hugging Face search failed: HTTP {}", r.status());
    }
    Ok(r.json().await?)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfFile {
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantGroup {
    pub quant: String,
    /// Model reference that selects exactly this group.
    pub model_ref: String,
    pub files: Vec<HfFile>,
    pub size: u64,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<TreeLfs>,
}

#[derive(Deserialize)]
struct TreeLfs {
    size: u64,
}

/// All GGUF files in a repo (excluding vision projectors), with sizes.
pub async fn hf_files(repo: &str) -> Result<Vec<HfFile>> {
    let url = format!("{}/api/models/{}/tree/main?recursive=true", hf_endpoint(), repo);
    let r = with_token(http_client().get(&url), &url).timeout(Duration::from_secs(30)).send().await?;
    if r.status() == reqwest::StatusCode::UNAUTHORIZED || r.status() == reqwest::StatusCode::FORBIDDEN {
        bail!("{repo} is gated or private: set HF_TOKEN");
    }
    if !r.status().is_success() {
        bail!("listing {repo} failed: HTTP {}", r.status());
    }
    let entries: Vec<TreeEntry> = r.json().await?;
    Ok(entries
        .into_iter()
        .filter(|e| e.kind == "file" && e.path.to_ascii_lowercase().ends_with(".gguf"))
        .filter(|e| !e.path.to_ascii_lowercase().contains("mmproj"))
        .map(|e| HfFile { size: e.lfs.map(|l| l.size).unwrap_or(e.size), path: e.path })
        .collect())
}

/// Group a repo's files by quantization (split files are combined into one entry).
pub fn group_quants(repo: &str, files: &[HfFile]) -> Vec<QuantGroup> {
    let mut groups: Vec<QuantGroup> = vec![];
    for f in files {
        let q = quant_of(&f.path);
        // Shards of one model share a prefix; non-split files with the same quant label
        // (e.g. two different models in one repo) stay separate.
        let key = match shard_info(&f.path) {
            Some((prefix, _, _)) => prefix,
            None => f.path.clone(),
        };
        if let Some(g) = groups.iter_mut().find(|g| g.files.iter().any(|x| shard_info(&x.path).map(|s| s.0) == Some(key.clone()))) {
            g.files.push(f.clone());
            g.size += f.size;
            continue;
        }
        groups.push(QuantGroup { quant: q.clone(), model_ref: String::new(), files: vec![f.clone()], size: f.size });
    }
    let dup_quant = |q: &str, gs: &[QuantGroup]| gs.iter().filter(|g| g.quant == q).count() > 1;
    let snapshot = groups.clone();
    for g in &mut groups {
        g.files.sort_by(|a, b| a.path.cmp(&b.path));
        g.model_ref = if g.quant != "unknown" && !dup_quant(&g.quant, &snapshot) {
            format!("{repo}:{}", g.quant)
        } else {
            format!("{repo}/{}", g.files[0].path)
        };
    }
    groups.sort_by_key(|g| g.size);
    groups
}

// ---------------------------------------------------------------------------------------------
// Resolution and download
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub url: String,
    /// Path relative to the models cache dir (also the name peers are asked for).
    pub rel: String,
    pub size: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum Resolved {
    Local(PathBuf),
    Remote(Vec<RemoteFile>),
}

fn repo_dir(repo: &str) -> String {
    repo.replace('/', "__")
}

/// Map a peer-supplied relative path into the models cache, refusing anything that escapes it.
pub fn cache_path_for(paths: &crate::config::Paths, rel: &str) -> Option<PathBuf> {
    if rel.is_empty() || rel.len() > 1024 || rel.starts_with('/') || rel.contains('\\') || rel.contains(':') {
        return None;
    }
    let mut p = paths.models.clone();
    for part in rel.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            return None;
        }
        p.push(part);
    }
    if !rel.to_ascii_lowercase().ends_with(".gguf") {
        return None;
    }
    Some(p)
}

/// Turn a reference into the list of files that have to be present locally.
pub async fn resolve(r: &ModelRef, paths: &crate::config::Paths) -> Result<Resolved> {
    let hf = hf_endpoint();
    match r {
        ModelRef::Local(p) => {
            if !p.is_file() {
                bail!("model file {} does not exist on the head node", p.display());
            }
            Ok(Resolved::Local(p.clone()))
        }
        ModelRef::Url(u) => {
            let name = u.split(['?', '#']).next().unwrap_or(u).rsplit('/').next().unwrap_or("model.gguf");
            let name = if name.to_ascii_lowercase().ends_with(".gguf") { name.to_string() } else { format!("{name}.gguf") };
            use sha2::Digest;
            let h = hex::encode(&sha2::Sha256::digest(u.as_bytes())[..6]);
            Ok(Resolved::Remote(vec![RemoteFile { url: u.clone(), rel: format!("url/{h}/{name}"), size: None }]))
        }
        ModelRef::Hf { repo, quant, file } => {
            let listing = hf_files(repo).await;
            let wanted: Vec<HfFile> = match (file, &listing) {
                (Some(f), _) => {
                    let names = match shard_info(f) {
                        Some((prefix, _, total)) => shard_names(&prefix, total),
                        None => vec![f.clone()],
                    };
                    names
                        .into_iter()
                        .map(|n| {
                            let size = listing.as_ref().ok().and_then(|l| l.iter().find(|x| x.path == n)).map(|x| x.size).unwrap_or(0);
                            HfFile { path: n, size }
                        })
                        .collect()
                }
                (None, Ok(list)) => {
                    let q = quant.clone().unwrap_or_else(|| DEFAULT_QUANT.to_string()).to_ascii_uppercase();
                    let groups = group_quants(repo, list);
                    let g = groups
                        .iter()
                        .find(|g| g.quant == q)
                        .or_else(|| groups.iter().find(|g| g.quant.ends_with(&q) || g.quant.contains(&q)))
                        .ok_or_else(|| {
                            let avail: Vec<_> = groups.iter().map(|g| g.quant.clone()).collect();
                            anyhow!("{repo} has no {q} GGUF (available: {})", avail.join(", "))
                        })?;
                    g.files.clone()
                }
                (None, Err(e)) => {
                    // Hugging Face unreachable: use whatever is already cached for this repo/quant.
                    let q = quant.clone().unwrap_or_else(|| DEFAULT_QUANT.to_string()).to_ascii_uppercase();
                    let dir = paths.models.join(repo_dir(repo));
                    let mut found = vec![];
                    if let Ok(rd) = std::fs::read_dir(&dir) {
                        for e in rd.flatten() {
                            let n = e.file_name().to_string_lossy().into_owned();
                            if n.to_ascii_lowercase().ends_with(".gguf") && quant_of(&n) == q {
                                found.push(HfFile { path: n, size: e.metadata().map(|m| m.len()).unwrap_or(0) });
                            }
                        }
                    }
                    if found.is_empty() {
                        return Err(anyhow!("can't list {repo} on Hugging Face ({e:#}) and nothing is cached"));
                    }
                    found.sort_by(|a, b| a.path.cmp(&b.path));
                    found
                }
            };
            Ok(Resolved::Remote(
                wanted
                    .into_iter()
                    .map(|f| RemoteFile {
                        url: format!("{hf}/{repo}/resolve/main/{}", f.path),
                        rel: format!("{}/{}", repo_dir(repo), f.path),
                        size: (f.size > 0).then_some(f.size),
                    })
                    .collect(),
            ))
        }
    }
}

pub type Progress<'a> = &'a (dyn Fn(u64, u64, &str) + Send + Sync);

struct Bar {
    tty: bool,
    last: Instant,
}

impl Bar {
    fn show(&mut self, name: &str, done: u64, total: u64, force: bool) {
        if !force && self.last.elapsed() < Duration::from_millis(if self.tty { 250 } else { 5000 }) {
            return;
        }
        self.last = Instant::now();
        let pct = if total > 0 { done as f64 * 100.0 / total as f64 } else { 0.0 };
        let hb = crate::sys::human_bytes;
        if self.tty {
            let width = 30usize;
            let filled = ((pct / 100.0) * width as f64) as usize;
            eprint!("\r  {name} [{}{}] {pct:5.1}% {}/{}   ", "#".repeat(filled.min(width)), "-".repeat(width - filled.min(width)), hb(done), hb(total));
            if force {
                eprintln!();
            }
        } else {
            info!("downloading {name}: {pct:.1}% ({}/{})", hb(done), hb(total));
        }
    }
}

/// Make sure every file is in the cache. Returns the path of the first file (llama.cpp finds
/// the other shards next to it).
pub async fn ensure_local(resolved: &Resolved, paths: &crate::config::Paths, mesh: Option<&Arc<Mesh>>, progress: Progress<'_>) -> Result<PathBuf> {
    let files = match resolved {
        Resolved::Local(p) => return Ok(p.clone()),
        Resolved::Remote(f) => f,
    };
    let mut first = None;
    for f in files {
        let dest = cache_path_for(paths, &f.rel).ok_or_else(|| anyhow!("bad cache path {}", f.rel))?;
        if first.is_none() {
            first = Some(dest.clone());
        }
        if let Ok(m) = std::fs::metadata(&dest) {
            if f.size.is_none_or(|s| s == m.len()) {
                continue;
            }
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match download_http(f, &dest, progress).await {
            Ok(()) => continue,
            Err(e) => {
                warn!("download of {} failed: {e:#}", f.rel);
                let Some(mesh) = mesh else { return Err(e) };
                download_from_peers(mesh, f, &dest, progress).await.with_context(|| format!("download failed ({e:#}) and no peer has {} cached", f.rel))?;
            }
        }
    }
    first.ok_or_else(|| anyhow!("model has no files"))
}

fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

async fn download_http(f: &RemoteFile, dest: &Path, progress: Progress<'_>) -> Result<()> {
    let part = part_path(dest);
    let name = f.rel.rsplit('/').next().unwrap_or(&f.rel).to_string();
    let client = http_client();
    let mut attempts = 0;
    loop {
        attempts += 1;
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut rb = with_token(client.get(&f.url), &f.url);
        if have > 0 {
            rb = rb.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let res = async {
            let r = rb.send().await?;
            let status = r.status();
            if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && f.size == Some(have) {
                return Ok(());
            }
            if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
                bail!("HTTP {status}: model is gated or private, set HF_TOKEN");
            }
            if !status.is_success() {
                bail!("HTTP {status} for {}", f.url);
            }
            let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
            let start = if resumed { have } else { 0 };
            let total = r.content_length().map(|l| l + start).or(f.size).unwrap_or(0);
            let mut out = tokio::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part).await?;
            let mut done = start;
            let mut bar = Bar { tty: std::io::stderr().is_terminal(), last: Instant::now() - Duration::from_secs(10) };
            let mut stream = r.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                out.write_all(&chunk).await?;
                done += chunk.len() as u64;
                progress(done, total, &name);
                bar.show(&name, done, total, false);
            }
            out.flush().await?;
            bar.show(&name, done, total, true);
            if let Some(s) = f.size {
                if done != s {
                    bail!("download incomplete ({done} of {s} bytes)");
                }
            }
            Ok(())
        }
        .await;
        match res {
            Ok(()) => {
                std::fs::rename(&part, dest)?;
                return Ok(());
            }
            Err(e) if attempts < 4 && !e.to_string().contains("HTTP 4") => {
                warn!("download error ({e:#}), retrying");
                tokio::time::sleep(Duration::from_secs(2 * attempts)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn download_from_peers(mesh: &Arc<Mesh>, f: &RemoteFile, dest: &Path, progress: Progress<'_>) -> Result<()> {
    let part = part_path(dest);
    let name = f.rel.rsplit('/').next().unwrap_or(&f.rel).to_string();
    let peers: Vec<_> = mesh.online_nodes().into_iter().filter(|n| n.id != mesh.id()).map(|n| n.id).collect();
    let mut last_err = anyhow!("no peers online");
    for p in peers {
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let (size, mut recv) = match mesh.fetch_file(&p, &f.rel, have).await {
            Ok(x) => x,
            Err(e) => {
                last_err = e;
                continue;
            }
        };
        info!("fetching {} from peer {}", f.rel, p.fmt_short());
        let mut out = tokio::fs::OpenOptions::new().create(true).append(true).open(&part).await?;
        let mut done = have;
        let mut buf = vec![0u8; 1 << 20];
        let mut bar = Bar { tty: std::io::stderr().is_terminal(), last: Instant::now() - Duration::from_secs(10) };
        loop {
            let n = AsyncReadExt::read(&mut recv, &mut buf).await?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n]).await?;
            done += n as u64;
            progress(done, size, &name);
            bar.show(&name, done, size, false);
        }
        out.flush().await?;
        bar.show(&name, done, size, true);
        if done == size {
            std::fs::rename(&part, dest)?;
            return Ok(());
        }
        last_err = anyhow!("peer sent {done} of {size} bytes");
    }
    Err(last_err)
}

/// Rough check of whether a model fits in the group's pooled memory.
pub fn fit_label(model_bytes: u64, capacity: u64) -> &'static str {
    // Weights plus ~15% for KV cache, compute buffers and per-device overhead.
    let need = model_bytes + model_bytes / 7 + (512 << 20);
    if need <= capacity * 85 / 100 {
        "fits"
    } else if need <= capacity {
        "tight"
    } else {
        "too big"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_refs() {
        assert_eq!(
            ModelRef::parse("unsloth/Qwen3-8B-GGUF:Q4_K_M").unwrap(),
            ModelRef::Hf { repo: "unsloth/Qwen3-8B-GGUF".into(), quant: Some("Q4_K_M".into()), file: None }
        );
        let r = ModelRef::parse("unsloth/Qwen3-8B-GGUF").unwrap();
        assert_eq!(r, ModelRef::Hf { repo: "unsloth/Qwen3-8B-GGUF".into(), quant: None, file: None });
        assert_eq!(r.alias(), "unsloth/Qwen3-8B-GGUF:Q4_K_M");
        assert_eq!(
            ModelRef::parse("bartowski/Llama-3.2-3B-Instruct-GGUF/sub/Llama-3.2-3B-Instruct-Q8_0.gguf").unwrap(),
            ModelRef::Hf { repo: "bartowski/Llama-3.2-3B-Instruct-GGUF".into(), quant: None, file: Some("sub/Llama-3.2-3B-Instruct-Q8_0.gguf".into()) }
        );
        assert!(matches!(ModelRef::parse("/models/x.gguf").unwrap(), ModelRef::Local(_)));
        assert!(matches!(ModelRef::parse("./x.gguf").unwrap(), ModelRef::Local(_)));
        assert!(matches!(ModelRef::parse("C:\\models\\x.gguf").unwrap(), ModelRef::Local(_)));
        assert_eq!(ModelRef::parse("https://example.com/a/b.gguf?download=1").unwrap(), ModelRef::Url("https://example.com/a/b.gguf?download=1".into()));
        assert!(ModelRef::parse("").is_err());
        assert!(ModelRef::parse("justaname").is_err());
        assert!(ModelRef::parse("owner/repo/readme.md").is_err());
        assert!(ModelRef::parse("owner/../etc").is_err());
        assert!(ModelRef::parse("nomic-ai/nomic-embed-text-v1.5-GGUF").unwrap().looks_like_embedding());
    }

    #[test]
    fn quants() {
        assert_eq!(quant_of("Qwen3-8B-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quant_of("Qwen3-8B-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL");
        assert_eq!(quant_of("llama-2-7b.Q5_0.gguf"), "Q5_0");
        assert_eq!(quant_of("model-f16.gguf"), "F16");
        assert_eq!(quant_of("Big-IQ2_XXS-00001-of-00003.gguf"), "IQ2_XXS");
        assert_eq!(quant_of("Q4_K_M/Big-00002-of-00002.gguf"), "Q4_K_M");
        assert_eq!(quant_of("Qwen3-Coder.gguf"), "unknown");
    }

    #[test]
    fn shards() {
        assert_eq!(shard_info("m-Q4_K_M-00001-of-00003.gguf"), Some(("m-Q4_K_M".into(), 1, 3)));
        assert_eq!(shard_info("m-Q4_K_M.gguf"), None);
        assert_eq!(shard_names("m", 2), vec!["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]);
    }

    #[test]
    fn grouping() {
        let files = vec![
            HfFile { path: "M-Q4_K_M.gguf".into(), size: 5 },
            HfFile { path: "Q8_0/M-Q8_0-00002-of-00002.gguf".into(), size: 4 },
            HfFile { path: "Q8_0/M-Q8_0-00001-of-00002.gguf".into(), size: 6 },
        ];
        let g = group_quants("o/r", &files);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].quant, "Q4_K_M");
        assert_eq!(g[0].model_ref, "o/r:Q4_K_M");
        assert_eq!(g[1].quant, "Q8_0");
        assert_eq!(g[1].size, 10);
        assert_eq!(g[1].files[0].path, "Q8_0/M-Q8_0-00001-of-00002.gguf");
    }

    #[test]
    fn fit() {
        let gb = 1u64 << 30;
        assert_eq!(fit_label(4 * gb, 16 * gb), "fits");
        assert_eq!(fit_label(14 * gb, 16 * gb), "too big");
    }

    #[test]
    fn cache_paths_are_confined() {
        let paths = crate::config::Paths { data: "/d".into(), models: "/m".into() };
        assert_eq!(cache_path_for(&paths, "o__r/x.gguf"), Some(PathBuf::from("/m/o__r/x.gguf")));
        assert_eq!(cache_path_for(&paths, "../x.gguf"), None);
        assert_eq!(cache_path_for(&paths, "/etc/x.gguf"), None);
        assert_eq!(cache_path_for(&paths, "o__r/x.txt"), None);
        assert_eq!(cache_path_for(&paths, "a//b.gguf"), None);
    }
}
