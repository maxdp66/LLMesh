//! Host memory information.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Memory {
    pub total: u64,
    pub free: u64,
}

pub fn memory() -> Memory {
    let mut s = sysinfo::System::new();
    s.refresh_memory();
    Memory { total: s.total_memory(), free: s.available_memory() }
}

pub fn cpu_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

pub fn hostname() -> String {
    sysinfo::System::host_name().unwrap_or_else(|| "node".into())
}

pub fn human_bytes(b: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 { format!("{b} B") } else { format!("{v:.1} {}", U[i]) }
}
