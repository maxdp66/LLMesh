//! Zero-setup LAN discovery with UDP broadcast beacons.
//!
//! Each node broadcasts `{tag, id, addrs, mac}` every few seconds on port 47474. Receivers in the
//! same group add the sender as a connection candidate. The beacon only carries addresses; the
//! real membership check is the authenticated handshake on the iroh connection.

use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use iroh::{EndpointAddr, EndpointId, TransportAddr};
use serde::{Deserialize, Serialize};
use socket2::{Domain, Protocol, Socket, Type};
use tracing::{debug, info, warn};

use crate::net::Mesh;

pub const DEFAULT_PORT: u16 = 47474;

#[derive(Debug, Serialize, Deserialize)]
struct Beacon {
    v: u8,
    tag: String,
    id: EndpointId,
    addrs: Vec<SocketAddr>,
    mac: String,
}

fn payload(tag: &str, id: &EndpointId, addrs: &[SocketAddr]) -> Vec<u8> {
    let mut p = tag.as_bytes().to_vec();
    p.extend_from_slice(id.as_bytes());
    for a in addrs {
        p.extend_from_slice(a.to_string().as_bytes());
        p.push(0);
    }
    p
}

fn socket(port: u16) -> Result<std::net::UdpSocket> {
    let s = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    s.set_reuse_address(true)?;
    #[cfg(all(unix, not(any(target_os = "solaris", target_os = "illumos"))))]
    s.set_reuse_port(true)?;
    s.set_broadcast(true)?;
    s.set_nonblocking(true)?;
    s.bind(&SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port)).into())?;
    Ok(s.into())
}

/// Extra unicast targets for beacons (comma separated `host:port`), mainly for testing on
/// networks where broadcast is filtered.
fn extra_targets() -> Vec<SocketAddr> {
    std::env::var("LLMESH_BEACON_TARGETS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

pub fn spawn(mesh: Arc<Mesh>, port: u16) {
    let sock = match socket(port).and_then(|s| Ok(tokio::net::UdpSocket::from_std(s)?)) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            warn!("LAN discovery disabled (cannot bind UDP {port}: {e})");
            return;
        }
    };
    info!("LAN discovery on UDP port {port}");
    let tag = mesh.secret.tag_hex();

    // Sender
    let (m, s, t) = (mesh.clone(), sock.clone(), tag.clone());
    tokio::spawn(async move {
        let targets: Vec<SocketAddr> = std::iter::once(SocketAddr::from((Ipv4Addr::BROADCAST, port))).chain(extra_targets()).collect();
        loop {
            let addrs: Vec<SocketAddr> = m.my_addr().ip_addrs().copied().collect();
            if !addrs.is_empty() {
                let id = m.id();
                let mac = hex::encode(m.secret.beacon_mac(&payload(&t, &id, &addrs)));
                let b = Beacon { v: 1, tag: t.clone(), id, addrs, mac };
                if let Ok(buf) = serde_json::to_vec(&b) {
                    for target in &targets {
                        if let Err(e) = s.send_to(&buf, target).await {
                            debug!("beacon to {target}: {e}");
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });

    // Receiver
    tokio::spawn(async move {
        let mut buf = vec![0u8; 4096];
        let mut seen: BTreeSet<(EndpointId, Vec<SocketAddr>)> = BTreeSet::new();
        loop {
            let Ok((n, from)) = sock.recv_from(&mut buf).await else {
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            };
            let Ok(b) = serde_json::from_slice::<Beacon>(&buf[..n]) else { continue };
            if b.v != 1 || b.tag != tag || b.id == mesh.id() || b.addrs.len() > 16 {
                continue;
            }
            let expect = hex::encode(mesh.secret.beacon_mac(&payload(&b.tag, &b.id, &b.addrs)));
            if expect != b.mac {
                continue;
            }
            // Prefer the address the packet actually came from, then the advertised ones.
            let mut addrs: BTreeSet<TransportAddr> = b.addrs.iter().map(|a| TransportAddr::Ip(*a)).collect();
            if let Some(port) = b.addrs.iter().find(|a| a.is_ipv4()).map(|a| a.port()) {
                addrs.insert(TransportAddr::Ip(SocketAddr::new(from.ip(), port)));
            }
            if seen.insert((b.id, b.addrs.clone())) {
                info!("LAN beacon from {} at {}", b.id.fmt_short(), from.ip());
            }
            mesh.add_candidate(EndpointAddr { id: b.id, addrs });
        }
    });
}
