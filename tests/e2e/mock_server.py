#!/usr/bin/env python3
"""Local stand-in for Hugging Face and GitHub releases, for the end-to-end tests.

  HF_ENDPOINT=http://127.0.0.1:PORT            /api/models, /api/models/{repo}/tree/main, /{repo}/resolve/main/{file}
  LLMESH_LLAMA_BASE_URL=http://127.0.0.1:PORT/releases   /releases/{tag}/llama-{tag}-bin-ubuntu-x64.tar.gz

The release tarball wraps the fake rpc-server / llama-server scripts in the same layout as the
real llama.cpp archives (top-level llama-{tag}/ folder, executable bits set).
"""
import io, json, os, sys, tarfile, hashlib, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("LLMESH_LLAMA_TAG", "b12017")
port = int(sys.argv[1]) if len(sys.argv) > 1 else 18999
counter = {"resolve": 0, "release": 0}

def blob(name, size):
    h = hashlib.sha256(name.encode()).digest()
    return (h * (size // len(h) + 1))[:size]

REPOS = {
    "test/Tiny-GGUF": {
        "Tiny-Q4_K_M.gguf": blob("q4", 300_000),
        "Tiny-Q8_0.gguf": blob("q8", 500_000),
        "mmproj-Tiny-F16.gguf": blob("mm", 1000),
        "README.md": b"hi",
    },
    "test/Split-GGUF": {
        "Q2_K/Split-Q2_K-00001-of-00002.gguf": blob("s1", 200_000),
        "Q2_K/Split-Q2_K-00002-of-00002.gguf": blob("s2", 150_000),
    },
    "test/Gated-GGUF": {"Gated-Q4_K_M.gguf": blob("g", 1000)},
}

def release_tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        d = tarfile.TarInfo(f"llama-{TAG}"); d.type = tarfile.DIRTYPE; d.mode = 0o755; t.addfile(d)
        # llama.cpp releases since late 2026 ship the RPC server as `ggml-rpc-server` (older ones: `rpc-server`).
        for exe, script in (("ggml-rpc-server", "fake_rpc_server.py"), ("llama-server", "fake_llama_server.py")):
            body = f'#!/bin/sh\nexec "{sys.executable}" "{os.path.join(HERE, script)}" "$@"\n'.encode()
            ti = tarfile.TarInfo(f"llama-{TAG}/{exe}"); ti.size = len(body); ti.mode = 0o755
            t.addfile(ti, io.BytesIO(body))
        lib = b"not really a library"
        ti = tarfile.TarInfo(f"llama-{TAG}/libggml.so.0.0.1"); ti.size = len(lib); ti.mode = 0o644
        t.addfile(ti, io.BytesIO(lib))
        ln = tarfile.TarInfo(f"llama-{TAG}/libggml.so"); ln.type = tarfile.SYMTYPE; ln.linkname = "libggml.so.0.0.1"
        t.addfile(ln)
    return buf.getvalue()

RELEASE = release_tarball()

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *a):
        sys.stderr.write("mock: " + (fmt % a) + "\n")
    def send(self, code, body, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
    def send_range(self, data):
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            if start >= len(data):
                return self.send(416, b"", extra={"Content-Range": f"bytes */{len(data)}"})
            return self.send(206, data[start:], "application/octet-stream", {"Content-Range": f"bytes {start}-{len(data)-1}/{len(data)}"})
        return self.send(200, data, "application/octet-stream")
    def do_GET(self):
        p = self.path.split("?")[0]
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        if p == "/api/models":
            term = ""
            for kv in q.split("&"):
                if kv.startswith("search="):
                    term = kv[7:].lower()
            res = [{"id": r, "downloads": 1000 - i, "likes": i} for i, r in enumerate(REPOS) if term in r.lower()]
            return self.send(200, json.dumps(res).encode())
        if p.startswith("/api/models/") and p.endswith("/tree/main"):
            repo = p[len("/api/models/"):-len("/tree/main")]
            if repo == "test/Gated-GGUF" and not self.headers.get("Authorization"):
                return self.send(401, b'{"error":"gated"}')
            if repo not in REPOS:
                return self.send(404, b'{"error":"not found"}')
            entries = []
            for name, data in REPOS[repo].items():
                e = {"type": "file", "path": name, "size": len(data)}
                if name.endswith(".gguf"):
                    e["size"] = 135; e["lfs"] = {"size": len(data), "oid": "x"}
                entries.append(e)
            return self.send(200, json.dumps(entries).encode())
        if p.startswith("/releases/"):
            name = p.rsplit("/", 1)[1]
            counter["release"] += 1
            if name == f"llama-{TAG}-bin-ubuntu-x64.tar.gz":
                return self.send_range(RELEASE)
            return self.send(404, b"no such asset")
        if p == "/_stats":
            return self.send(200, json.dumps(counter).encode())
        if "/resolve/main/" in p:
            repo, f = p.lstrip("/").split("/resolve/main/", 1)
            if os.environ.get("MOCK_HF_DOWN") or os.path.exists(os.path.join(HERE, ".hf_down")):
                return self.send(503, b"down")
            data = REPOS.get(repo, {}).get(f)
            if data is None:
                return self.send(404, b"not found")
            counter["resolve"] += 1
            return self.send_range(data)
        return self.send(404, b"not found")
    do_HEAD = do_GET

print(f"mock HF + releases on http://127.0.0.1:{port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
