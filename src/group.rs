//! Groups: invites, peer authentication, the admin-signed group config, votes and tallying.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use anyhow::{anyhow, bail, ensure, Context, Result};
use hmac::{Hmac, Mac};
use iroh::{EndpointAddr, PublicKey, SecretKey, Signature, TransportAddr};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

pub const INVITE_PREFIX: &str = "llmesh1";
const INVITE_VERSION: u8 = 1;
const LAN_SECRET_SEED: &[u8] = b"llmesh-lan-default-group-v1";

/// The 32-byte shared secret of a group. Knowing it is what makes you a member.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GroupSecret(pub [u8; 32]);

impl std::fmt::Debug for GroupSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GroupSecret({})", hex::encode(self.tag()))
    }
}

impl GroupSecret {
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut b = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut b);
        GroupSecret(b)
    }

    /// The fixed secret of the zero-setup LAN group.
    pub fn lan_default() -> Self {
        GroupSecret(Sha256::digest(LAN_SECRET_SEED).into())
    }

    /// `sha256(secret)[..8]`: a public identifier for the group.
    pub fn tag(&self) -> [u8; 8] {
        let d = Sha256::digest(self.0);
        let mut t = [0u8; 8];
        t.copy_from_slice(&d[..8]);
        t
    }

    pub fn tag_hex(&self) -> String {
        hex::encode(self.tag())
    }

    fn mac(&self) -> HmacSha256 {
        HmacSha256::new_from_slice(&self.0).expect("hmac accepts any key length")
    }

    /// `HMAC-SHA256(secret, "llmesh-auth" || sender_id || receiver_id)`.
    ///
    /// Both ids are the TLS-authenticated iroh endpoint ids, so a proof captured on one
    /// connection is useless on any other (it would have to be presented by `sender`).
    pub fn auth_proof(&self, sender: &PublicKey, receiver: &PublicKey) -> [u8; 32] {
        let mut m = self.mac();
        m.update(b"llmesh-auth");
        m.update(sender.as_bytes());
        m.update(receiver.as_bytes());
        m.finalize().into_bytes().into()
    }

    /// Constant-time check of a peer's proof.
    pub fn verify_auth(&self, sender: &PublicKey, receiver: &PublicKey, proof: &[u8]) -> bool {
        let mut m = self.mac();
        m.update(b"llmesh-auth");
        m.update(sender.as_bytes());
        m.update(receiver.as_bytes());
        m.verify_slice(proof).is_ok()
    }

    /// MAC used on LAN beacons so that random UDP noise can't inject addresses.
    pub fn beacon_mac(&self, payload: &[u8]) -> [u8; 16] {
        let mut m = self.mac();
        m.update(b"llmesh-beacon");
        m.update(payload);
        let full: [u8; 32] = m.finalize().into_bytes().into();
        let mut out = [0u8; 16];
        out.copy_from_slice(&full[..16]);
        out
    }
}

/// First frame on the first stream of every connection, in both directions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthHello {
    pub group_tag: String,
    pub proof: String,
    pub version: String,
}

impl AuthHello {
    pub fn new(secret: &GroupSecret, me: &PublicKey, them: &PublicKey) -> Self {
        AuthHello {
            group_tag: secret.tag_hex(),
            proof: hex::encode(secret.auth_proof(me, them)),
            version: crate::VERSION.to_string(),
        }
    }

    /// Verify a hello that `sender` sent to `receiver`.
    pub fn verify(&self, secret: &GroupSecret, sender: &PublicKey, receiver: &PublicKey) -> Result<()> {
        ensure!(self.group_tag == secret.tag_hex(), "peer is in a different group");
        let proof = hex::decode(&self.proof).context("bad proof encoding")?;
        ensure!(secret.verify_auth(sender, receiver, &proof), "bad group proof");
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Invites
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Invite {
    pub secret: GroupSecret,
    pub admin: PublicKey,
    pub bootstrap: Vec<EndpointAddr>,
    pub name: String,
}

struct Writer(Vec<u8>);
impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v)
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_be_bytes())
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b)
    }
    fn str16(&mut self, s: &str) {
        let b = s.as_bytes();
        self.u16(b.len().min(u16::MAX as usize) as u16);
        self.bytes(&b[..b.len().min(u16::MAX as usize)]);
    }
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.0.len() >= n, "invite is truncated");
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }
    fn arr32(&mut self) -> Result<[u8; 32]> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    fn str16(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
}

impl Invite {
    pub const MAX_BOOTSTRAP: usize = 4;
    const MAX_ADDRS_PER_PEER: usize = 8;

    pub fn encode(&self) -> String {
        let mut w = Writer(Vec::with_capacity(256));
        w.u8(INVITE_VERSION);
        w.bytes(&self.secret.0);
        w.bytes(self.admin.as_bytes());
        let boots: Vec<_> = self.bootstrap.iter().take(Self::MAX_BOOTSTRAP).collect();
        w.u8(boots.len() as u8);
        for b in boots {
            w.bytes(b.id.as_bytes());
            // Relay URLs first, then IPv4, then IPv6; capped so invites stay short.
            let mut addrs: Vec<&TransportAddr> = b.addrs.iter().collect();
            addrs.sort_by_key(|a| match a {
                TransportAddr::Relay(_) => 0,
                TransportAddr::Ip(SocketAddr::V4(_)) => 1,
                TransportAddr::Ip(SocketAddr::V6(_)) => 2,
                _ => 3,
            });
            let addrs: Vec<_> = addrs
                .into_iter()
                .filter(|a| matches!(a, TransportAddr::Relay(_) | TransportAddr::Ip(_)))
                .take(Self::MAX_ADDRS_PER_PEER)
                .collect();
            w.u8(addrs.len() as u8);
            for a in addrs {
                match a {
                    TransportAddr::Ip(SocketAddr::V4(s)) => {
                        w.u8(4);
                        w.bytes(&s.ip().octets());
                        w.u16(s.port());
                    }
                    TransportAddr::Ip(SocketAddr::V6(s)) => {
                        w.u8(6);
                        w.bytes(&s.ip().octets());
                        w.u16(s.port());
                    }
                    TransportAddr::Relay(url) => {
                        w.u8(b'r');
                        w.str16(url.as_str());
                    }
                    _ => unreachable!(),
                }
            }
        }
        w.str16(&self.name);
        let body = data_encoding::BASE32_NOPAD.encode(&w.0).to_ascii_lowercase();
        format!("{INVITE_PREFIX}{body}")
    }

    pub fn decode(s: &str) -> Result<Self> {
        let s: String = s.trim().chars().filter(|c| !c.is_whitespace()).collect();
        let body = s
            .strip_prefix(INVITE_PREFIX)
            .ok_or_else(|| anyhow!("not an llmesh invite (should start with {INVITE_PREFIX})"))?;
        let raw = data_encoding::BASE32_NOPAD
            .decode(body.to_ascii_uppercase().as_bytes())
            .context("invite is not valid base32")?;
        let mut r = Reader(&raw);
        let ver = r.u8()?;
        ensure!(ver == INVITE_VERSION, "unsupported invite version {ver}");
        let secret = GroupSecret(r.arr32()?);
        let admin = PublicKey::from_bytes(&r.arr32()?).map_err(|e| anyhow!("bad admin key: {e}"))?;
        let n = r.u8()? as usize;
        ensure!(n <= Self::MAX_BOOTSTRAP, "too many bootstrap peers");
        let mut bootstrap = Vec::with_capacity(n);
        for _ in 0..n {
            let id = PublicKey::from_bytes(&r.arr32()?).map_err(|e| anyhow!("bad peer id: {e}"))?;
            let na = r.u8()? as usize;
            let mut addrs = BTreeSet::new();
            for _ in 0..na {
                match r.u8()? {
                    4 => {
                        let o: [u8; 4] = r.take(4)?.try_into().unwrap();
                        let port = r.u16()?;
                        addrs.insert(TransportAddr::Ip(SocketAddr::new(IpAddr::V4(Ipv4Addr::from(o)), port)));
                    }
                    6 => {
                        let o: [u8; 16] = r.take(16)?.try_into().unwrap();
                        let port = r.u16()?;
                        addrs.insert(TransportAddr::Ip(SocketAddr::new(IpAddr::V6(Ipv6Addr::from(o)), port)));
                    }
                    b'r' => {
                        let url = r.str16()?;
                        addrs.insert(TransportAddr::Relay(url.parse().map_err(|e| anyhow!("bad relay url: {e}"))?));
                    }
                    t => bail!("unknown address type {t} in invite"),
                }
            }
            bootstrap.push(EndpointAddr { id, addrs });
        }
        let name = r.str16()?;
        Ok(Invite { secret, admin, bootstrap, name })
    }
}

// ---------------------------------------------------------------------------------------------
// Signed group config and votes
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Selection {
    #[default]
    Admin,
    Vote,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Poll {
    pub round: u64,
    pub candidates: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupConfig {
    pub seq: u64,
    pub selection: Selection,
    pub active_model: String,
    pub poll: Poll,
    pub ctx_size: u32,
    /// Unix millis when signed. In admin-less (LAN) groups the latest timestamp wins.
    pub ts: u64,
}

impl GroupConfig {
    pub fn initial(model: &str) -> Self {
        GroupConfig {
            seq: 1,
            selection: Selection::Admin,
            active_model: model.to_string(),
            poll: Poll { round: 1, candidates: vec![] },
            ctx_size: crate::DEFAULT_CTX,
            ts: now_ms(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedConfig {
    pub config: GroupConfig,
    pub signer: PublicKey,
    pub sig: String,
}

fn config_msg(tag: &[u8; 8], c: &GroupConfig) -> Vec<u8> {
    let mut m = b"llmesh-config".to_vec();
    m.extend_from_slice(tag);
    m.extend_from_slice(&serde_json::to_vec(c).expect("config serializes"));
    m
}

impl SignedConfig {
    pub fn sign(secret: &GroupSecret, key: &SecretKey, config: GroupConfig) -> Self {
        let sig = key.sign(&config_msg(&secret.tag(), &config));
        SignedConfig { config, signer: key.public(), sig: hex::encode(sig.to_bytes()) }
    }

    /// Check the signature, and that the signer is the admin (if the group has one).
    pub fn verify(&self, secret: &GroupSecret, admin: Option<&PublicKey>) -> Result<()> {
        if let Some(admin) = admin {
            ensure!(&self.signer == admin, "config not signed by the group admin");
        }
        verify_sig(&self.signer, &config_msg(&secret.tag(), &self.config), &self.sig)
    }

    /// Ordering used to pick the winning config: the highest seq in admin groups, the
    /// latest timestamp in admin-less groups. The rest of the tuple only breaks ties.
    pub fn rank(&self, has_admin: bool) -> (u64, u64, [u8; 32]) {
        if has_admin {
            (self.config.seq, self.config.ts, *self.signer.as_bytes())
        } else {
            (self.config.ts, self.config.seq, *self.signer.as_bytes())
        }
    }
}

/// Pick the better of the current and a candidate config. Invalid candidates lose.
pub fn better_config<'a>(
    secret: &GroupSecret,
    admin: Option<&PublicKey>,
    current: Option<&'a SignedConfig>,
    candidate: &'a SignedConfig,
) -> bool {
    if candidate.verify(secret, admin).is_err() {
        return false;
    }
    match current {
        None => true,
        Some(cur) => candidate.rank(admin.is_some()) > cur.rank(admin.is_some()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vote {
    pub round: u64,
    pub model_ref: String,
    pub voter_id: PublicKey,
    pub ts: u64,
    pub sig: String,
}

#[derive(Serialize)]
struct VoteBody<'a> {
    round: u64,
    model_ref: &'a str,
    voter_id: &'a PublicKey,
    ts: u64,
}

fn vote_msg(tag: &[u8; 8], round: u64, model_ref: &str, voter: &PublicKey, ts: u64) -> Vec<u8> {
    let mut m = b"llmesh-vote".to_vec();
    m.extend_from_slice(tag);
    let body = VoteBody { round, model_ref, voter_id: voter, ts };
    m.extend_from_slice(&serde_json::to_vec(&body).expect("vote serializes"));
    m
}

impl Vote {
    pub fn sign(secret: &GroupSecret, key: &SecretKey, round: u64, model_ref: &str) -> Self {
        let voter = key.public();
        let ts = now_ms();
        let sig = key.sign(&vote_msg(&secret.tag(), round, model_ref, &voter, ts));
        Vote { round, model_ref: model_ref.to_string(), voter_id: voter, ts, sig: hex::encode(sig.to_bytes()) }
    }

    pub fn verify(&self, secret: &GroupSecret) -> Result<()> {
        ensure!(!self.model_ref.is_empty() && self.model_ref.len() <= 512, "bad model ref in vote");
        verify_sig(
            &self.voter_id,
            &vote_msg(&secret.tag(), self.round, &self.model_ref, &self.voter_id, self.ts),
            &self.sig,
        )
    }
}

fn verify_sig(pk: &PublicKey, msg: &[u8], sig_hex: &str) -> Result<()> {
    let raw = hex::decode(sig_hex).context("bad signature encoding")?;
    let arr: [u8; 64] = raw.as_slice().try_into().map_err(|_| anyhow!("bad signature length"))?;
    pk.verify(msg, &Signature::from_bytes(&arr)).map_err(|_| anyhow!("bad signature"))
}

/// Keep only the latest valid vote per voter. Returns true if `v` changed the set.
pub fn merge_vote(votes: &mut BTreeMap<PublicKey, Vote>, secret: &GroupSecret, v: Vote) -> bool {
    if v.verify(secret).is_err() {
        return false;
    }
    match votes.get(&v.voter_id) {
        Some(old) if (old.round, old.ts) >= (v.round, v.ts) => false,
        _ => {
            votes.insert(v.voter_id, v);
            true
        }
    }
}

/// Count the latest vote of each voter in `round`. Sorted by count (desc), then model (asc).
pub fn tally<'a>(votes: impl IntoIterator<Item = &'a Vote>, round: u64) -> Vec<(String, usize)> {
    let mut latest: BTreeMap<PublicKey, &Vote> = BTreeMap::new();
    for v in votes {
        if v.round != round {
            continue;
        }
        match latest.get(&v.voter_id) {
            Some(old) if old.ts >= v.ts => {}
            _ => {
                latest.insert(v.voter_id, v);
            }
        }
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for v in latest.values() {
        *counts.entry(v.model_ref.clone()).or_default() += 1;
    }
    let mut out: Vec<(String, usize)> = counts.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// The model the group should be running right now.
pub fn effective_model<'a>(config: &GroupConfig, votes: impl IntoIterator<Item = &'a Vote>) -> String {
    if config.selection == Selection::Vote {
        if let Some((winner, _)) = tally(votes, config.poll.round).into_iter().next() {
            return winner;
        }
    }
    config.active_model.clone()
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> SecretKey {
        SecretKey::generate()
    }

    #[test]
    fn invite_roundtrip() {
        let admin = key();
        let peer = key();
        let addr = EndpointAddr {
            id: peer.public(),
            addrs: [
                TransportAddr::Ip("192.168.1.20:5000".parse().unwrap()),
                TransportAddr::Ip("[fe80::1]:6000".parse().unwrap()),
                TransportAddr::Relay("https://euw1-1.relay.iroh.network./".parse().unwrap()),
            ]
            .into_iter()
            .collect(),
        };
        let inv = Invite { secret: GroupSecret::generate(), admin: admin.public(), bootstrap: vec![addr], name: "Friends 🎮".into() };
        let s = inv.encode();
        assert!(s.starts_with("llmesh1"));
        assert!(s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()));
        let back = Invite::decode(&s).unwrap();
        assert_eq!(back, inv);
        // whitespace/case tolerant
        let back2 = Invite::decode(&format!("  {}\n", s)).unwrap();
        assert_eq!(back2, inv);
    }

    #[test]
    fn invite_rejects_garbage() {
        assert!(Invite::decode("hello").is_err());
        assert!(Invite::decode("llmesh1aaaa").is_err());
        let inv = Invite { secret: GroupSecret::generate(), admin: key().public(), bootstrap: vec![], name: "x".into() };
        let s = inv.encode();
        assert!(Invite::decode(&s[..s.len() - 4]).is_err());
    }

    #[test]
    fn auth_proofs() {
        let s = GroupSecret::generate();
        let a = key().public();
        let b = key().public();
        let hello = AuthHello::new(&s, &a, &b);
        assert!(hello.verify(&s, &a, &b).is_ok());
        // a proof can't be replayed in the other direction or by another peer
        assert!(hello.verify(&s, &b, &a).is_err());
        let c = key().public();
        assert!(hello.verify(&s, &c, &b).is_err());
        // wrong secret
        let other = GroupSecret::generate();
        assert!(hello.verify(&other, &a, &b).is_err());
        let mut forged = AuthHello::new(&other, &a, &b);
        forged.group_tag = s.tag_hex();
        assert!(forged.verify(&s, &a, &b).is_err());
    }

    #[test]
    fn lan_secret_is_stable() {
        assert_eq!(GroupSecret::lan_default(), GroupSecret::lan_default());
        assert_eq!(GroupSecret::lan_default().tag().len(), 8);
    }

    #[test]
    fn config_signatures() {
        let s = GroupSecret::generate();
        let admin = key();
        let mallory = key();
        let c = SignedConfig::sign(&s, &admin, GroupConfig::initial("a/b"));
        assert!(c.verify(&s, Some(&admin.public())).is_ok());
        assert!(c.verify(&s, None).is_ok());
        let m = SignedConfig::sign(&s, &mallory, GroupConfig::initial("evil/model"));
        assert!(m.verify(&s, Some(&admin.public())).is_err());
        // tampering breaks the signature
        let mut t = c.clone();
        t.config.active_model = "x/y".into();
        assert!(t.verify(&s, Some(&admin.public())).is_err());
        // signature is bound to the group
        assert!(c.verify(&GroupSecret::generate(), Some(&admin.public())).is_err());

        // highest seq wins with an admin
        let mut c2cfg = c.config.clone();
        c2cfg.seq = 2;
        c2cfg.ts = 1; // older timestamp, still wins on seq
        let c2 = SignedConfig::sign(&s, &admin, c2cfg);
        assert!(better_config(&s, Some(&admin.public()), Some(&c), &c2));
        assert!(!better_config(&s, Some(&admin.public()), Some(&c2), &c));
        assert!(!better_config(&s, Some(&admin.public()), Some(&c), &m));
        // without an admin, latest timestamp wins
        assert!(better_config(&s, None, Some(&c2), &c));
    }

    #[test]
    fn votes_and_tally() {
        let s = GroupSecret::generate();
        let (a, b, c) = (key(), key(), key());
        let mut votes = BTreeMap::new();
        assert!(merge_vote(&mut votes, &s, Vote::sign(&s, &a, 1, "m/one")));
        assert!(merge_vote(&mut votes, &s, Vote::sign(&s, &b, 1, "m/two")));
        assert!(merge_vote(&mut votes, &s, Vote::sign(&s, &c, 1, "m/two")));
        let t = tally(votes.values(), 1);
        assert_eq!(t, vec![("m/two".to_string(), 2), ("m/one".to_string(), 1)]);

        // a newer vote replaces the old one
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(merge_vote(&mut votes, &s, Vote::sign(&s, &c, 1, "m/one")));
        let t = tally(votes.values(), 1);
        assert_eq!(t, vec![("m/one".to_string(), 2), ("m/two".to_string(), 1)]);

        // forged / wrong-group votes are rejected
        let mut forged = Vote::sign(&s, &a, 1, "m/three");
        forged.voter_id = b.public();
        assert!(!merge_vote(&mut votes, &s, forged));
        assert!(!merge_vote(&mut votes, &s, Vote::sign(&GroupSecret::generate(), &a, 1, "m/x")));

        // ties broken by model string
        let votes2 = [Vote::sign(&s, &a, 2, "z/z"), Vote::sign(&s, &b, 2, "a/a")];
        assert_eq!(tally(votes2.iter(), 2)[0].0, "a/a");
        // other rounds don't count
        assert!(tally(votes2.iter(), 3).is_empty());
    }

    #[test]
    fn effective_model_modes() {
        let s = GroupSecret::generate();
        let (a, b) = (key(), key());
        let mut cfg = GroupConfig::initial("admin/choice");
        let votes = [Vote::sign(&s, &a, 1, "voted/model"), Vote::sign(&s, &b, 1, "voted/model")];
        assert_eq!(effective_model(&cfg, votes.iter()), "admin/choice");
        cfg.selection = Selection::Vote;
        assert_eq!(effective_model(&cfg, votes.iter()), "voted/model");
        cfg.poll.round = 2; // new round, no votes yet -> admin choice
        assert_eq!(effective_model(&cfg, votes.iter()), "admin/choice");
    }
}
