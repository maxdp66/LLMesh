//! Per-node status shared over gossip, and head election.

use iroh::EndpointId;
use serde::{Deserialize, Serialize};

use crate::sys::Memory;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuDevice {
    pub name: String,
    pub description: String,
    pub total: u64,
    pub free: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Not (yet) eligible: llama.cpp not ready, or on a different llama.cpp tag.
    #[default]
    Idle,
    /// Contributes its rpc-server to the head.
    Worker,
    /// Runs llama-server and serves the model.
    Head,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum HeadState {
    #[default]
    Idle,
    Downloading { done: u64, total: u64, file: String },
    Loading,
    Ready,
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeInfo {
    pub id: EndpointId,
    pub name: String,
    pub version: String,
    pub llama_tag: String,
    pub backend: String,
    pub os: String,
    pub arch: String,
    /// Unix millis when this llmesh process started (used for election).
    pub started_at: u64,
    pub worker_only: bool,
    pub rpc_ready: bool,
    pub memory: Memory,
    pub gpus: Vec<GpuDevice>,
    pub role: Role,
    pub head_state: HeadState,
    /// Model this node is serving (head) or thinks the group runs.
    pub model: String,
    /// Alias the head's llama-server uses for the model.
    #[serde(default)]
    pub model_alias: String,
    /// Peers whose rpc-server the head is using in its current split.
    #[serde(default)]
    pub split: Vec<EndpointId>,
}

impl NodeInfo {
    /// Can this node take part in the model split with a head on `tag`?
    pub fn can_work(&self, tag: &str) -> bool {
        self.rpc_ready && self.llama_tag == tag
    }

    /// Memory this node can contribute to the model: VRAM if it has GPUs, otherwise RAM.
    pub fn capacity(&self) -> u64 {
        let vram: u64 = self.gpus.iter().map(|g| g.free).sum();
        if vram > 0 { vram } else { self.memory.free }
    }
}

fn tag_num(t: &str) -> u64 {
    t.trim_start_matches('b').parse().unwrap_or(0)
}

/// The llama.cpp tag the group runs on: the most common tag among ready nodes (ties go to the
/// newer release). Nodes on any other tag are left out of the split and flagged.
pub fn group_tag<'a>(nodes: impl IntoIterator<Item = &'a NodeInfo>) -> Option<String> {
    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for n in nodes {
        if n.rpc_ready && !n.llama_tag.is_empty() {
            *counts.entry(n.llama_tag.as_str()).or_default() += 1;
        }
    }
    counts.into_iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| tag_num(a.0).cmp(&tag_num(b.0)))).map(|(t, _)| t.to_string())
}

/// Pick the head: among eligible nodes (same llama tag, rpc ready, not worker-only),
/// the one with the lowest `(started_at, id)`.
pub fn elect_head<'a>(nodes: impl IntoIterator<Item = &'a NodeInfo>, tag: &str) -> Option<EndpointId> {
    nodes
        .into_iter()
        .filter(|n| n.can_work(tag) && !n.worker_only)
        .min_by(|a, b| (a.started_at, a.id).cmp(&(b.started_at, b.id)))
        .map(|n| n.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

    fn node(started: u64, tag: &str) -> NodeInfo {
        NodeInfo {
            id: SecretKey::generate().public(),
            name: "n".into(),
            version: "0".into(),
            llama_tag: tag.into(),
            backend: "cpu".into(),
            os: "linux".into(),
            arch: "x86_64".into(),
            started_at: started,
            worker_only: false,
            rpc_ready: true,
            memory: Memory { total: 8, free: 4 },
            gpus: vec![],
            role: Role::Idle,
            head_state: HeadState::Idle,
            model: String::new(),
            model_alias: String::new(),
            split: vec![],
        }
    }

    #[test]
    fn election() {
        let a = node(100, "b1");
        let b = node(50, "b1");
        let mut c = node(10, "b2"); // oldest, but wrong tag
        let mut d = node(5, "b1");
        d.worker_only = true;
        let mut e = node(1, "b1");
        e.rpc_ready = false;
        let all = [a.clone(), b.clone(), c.clone(), d, e];
        assert_eq!(elect_head(all.iter(), "b1"), Some(b.id));
        c.llama_tag = "b1".into();
        assert_eq!(elect_head([a.clone(), b, c.clone()].iter(), "b1"), Some(c.id));
        // tie on started_at -> lowest id
        let mut x = node(7, "b1");
        let mut y = node(7, "b1");
        if x.id > y.id {
            std::mem::swap(&mut x, &mut y);
        }
        assert_eq!(elect_head([y, x.clone()].iter(), "b1"), Some(x.id));
        assert_eq!(elect_head(std::iter::empty(), "b1"), None);
    }

    #[test]
    fn majority_tag() {
        let n = [node(1, "b100"), node(2, "b200"), node(3, "b100")];
        assert_eq!(group_tag(n.iter()).as_deref(), Some("b100"));
        let n = [node(1, "b100"), node(2, "b200")];
        assert_eq!(group_tag(n.iter()).as_deref(), Some("b200"));
        let mut x = node(1, "b300");
        x.rpc_ready = false;
        assert_eq!(group_tag([x].iter()), None);
    }
}
