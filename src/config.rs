//! Data directories, the node key, and saved group files.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use iroh::{EndpointAddr, PublicKey, SecretKey};
use serde::{Deserialize, Serialize};

use crate::group::{GroupSecret, Invite, SignedConfig, Vote};

#[derive(Debug, Clone)]
pub struct Paths {
    /// Keys, groups, llama.cpp binaries.
    pub data: PathBuf,
    /// Downloaded GGUF models.
    pub models: PathBuf,
}

impl Paths {
    pub fn new(data_dir: Option<PathBuf>) -> Result<Self> {
        let p = match data_dir {
            Some(d) => Paths { models: d.join("models"), data: d },
            None => {
                let data = dirs::data_dir().ok_or_else(|| anyhow!("no data dir on this OS; pass --data-dir"))?.join("llmesh");
                let models = match std::env::var_os("LLMESH_MODELS_DIR") {
                    Some(m) => PathBuf::from(m),
                    None => dirs::cache_dir().unwrap_or_else(|| data.clone()).join("llmesh").join("models"),
                };
                Paths { data, models }
            }
        };
        std::fs::create_dir_all(&p.data).with_context(|| format!("creating {}", p.data.display()))?;
        std::fs::create_dir_all(p.groups_dir())?;
        std::fs::create_dir_all(&p.models)?;
        Ok(p)
    }

    pub fn llama_dir(&self) -> PathBuf {
        self.data.join("llama.cpp")
    }
    pub fn groups_dir(&self) -> PathBuf {
        self.data.join("groups")
    }
    fn key_file(&self) -> PathBuf {
        self.data.join("node.key")
    }
    fn current_file(&self) -> PathBuf {
        self.data.join("current-group")
    }
    pub fn group_file(&self, tag_hex: &str) -> PathBuf {
        self.groups_dir().join(format!("{tag_hex}.json"))
    }

    /// The node's long-term iroh key (its identity and endpoint id).
    pub fn node_key(&self) -> Result<SecretKey> {
        let f = self.key_file();
        if let Ok(s) = std::fs::read_to_string(&f) {
            let raw = hex::decode(s.trim()).context("corrupt node.key")?;
            let arr: [u8; 32] = raw.as_slice().try_into().map_err(|_| anyhow!("corrupt node.key"))?;
            return Ok(SecretKey::from_bytes(&arr));
        }
        let k = SecretKey::generate();
        write_private(&f, hex::encode(k.to_bytes()).as_bytes())?;
        Ok(k)
    }

    pub fn current_group_tag(&self) -> Option<String> {
        std::fs::read_to_string(self.current_file()).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
    }

    pub fn set_current_group(&self, tag: Option<&str>) -> Result<()> {
        match tag {
            Some(t) => std::fs::write(self.current_file(), t)?,
            None => {
                let _ = std::fs::remove_file(self.current_file());
            }
        }
        Ok(())
    }

    pub fn list_groups(&self) -> Vec<SavedGroup> {
        let mut out = vec![];
        if let Ok(rd) = std::fs::read_dir(self.groups_dir()) {
            for e in rd.flatten() {
                if e.path().extension().is_some_and(|x| x == "json") {
                    if let Ok(g) = SavedGroup::load(&e.path()) {
                        out.push(g);
                    }
                }
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Load the active group, or the LAN default group.
    pub fn active_group(&self) -> Result<SavedGroup> {
        if let Some(tag) = self.current_group_tag() {
            let f = self.group_file(&tag);
            if f.exists() {
                return SavedGroup::load(&f);
            }
        }
        let lan = SavedGroup::lan();
        let f = self.group_file(&lan.secret().tag_hex());
        if f.exists() {
            if let Ok(g) = SavedGroup::load(&f) {
                return Ok(g);
            }
        }
        Ok(lan)
    }

    pub fn find_group(&self, name_or_tag: &str) -> Option<SavedGroup> {
        self.list_groups().into_iter().find(|g| g.name == name_or_tag || g.secret().tag_hex() == name_or_tag)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedMember {
    pub addr: EndpointAddr,
    #[serde(default)]
    pub name: String,
    /// Unix millis when we last heard from this member.
    pub last_seen: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedGroup {
    pub name: String,
    pub secret_hex: String,
    /// The admin's public key; `None` for the LAN default group.
    pub admin: Option<PublicKey>,
    #[serde(default)]
    pub bootstrap: Vec<EndpointAddr>,
    #[serde(default)]
    pub members: BTreeMap<PublicKey, SavedMember>,
    #[serde(default)]
    pub config: Option<SignedConfig>,
    #[serde(default)]
    pub votes: BTreeMap<PublicKey, Vote>,
}

impl SavedGroup {
    pub fn lan() -> Self {
        SavedGroup {
            name: "LAN".into(),
            secret_hex: hex::encode(GroupSecret::lan_default().0),
            admin: None,
            bootstrap: vec![],
            members: BTreeMap::new(),
            config: None,
            votes: BTreeMap::new(),
        }
    }

    pub fn from_invite(inv: &Invite) -> Self {
        SavedGroup {
            name: inv.name.clone(),
            secret_hex: hex::encode(inv.secret.0),
            admin: Some(inv.admin),
            bootstrap: inv.bootstrap.clone(),
            members: BTreeMap::new(),
            config: None,
            votes: BTreeMap::new(),
        }
    }

    pub fn secret(&self) -> GroupSecret {
        let raw = hex::decode(&self.secret_hex).unwrap_or_default();
        let mut s = [0u8; 32];
        if raw.len() == 32 {
            s.copy_from_slice(&raw);
        }
        GroupSecret(s)
    }

    pub fn is_lan(&self) -> bool {
        self.admin.is_none()
    }

    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save(&self, paths: &Paths) -> Result<()> {
        let f = paths.group_file(&self.secret().tag_hex());
        let tmp = f.with_extension("json.tmp");
        write_private(&tmp, serde_json::to_string_pretty(self)?.as_bytes())?;
        std::fs::rename(&tmp, &f)?;
        Ok(())
    }
}

/// Write a file readable only by the current user (on Unix).
fn write_private(path: &Path, data: &[u8]) -> Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new().create(true).truncate(true).write(true).mode(0o600).open(path)?;
        f.write_all(data)?;
    }
    #[cfg(not(unix))]
    std::fs::write(path, data)?;
    Ok(())
}
