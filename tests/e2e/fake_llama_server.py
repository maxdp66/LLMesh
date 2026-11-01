#!/usr/bin/env python3
"""Stand-in for llama.cpp's llama-server used by the end-to-end tests.

On start it PINGs every --rpc endpoint (through llmesh's tunnels) and remembers which peers
answered. It serves /health (503 while "loading", then 200), OpenAI chat/completions (streamed
and not), completions, embeddings (only with --embeddings), and exits if an RPC peer goes away,
like the real thing does when a worker disappears mid-run.
"""
import json, os, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
if "--version" in args:
    print("version: 0.8.0-dev (build 12017, commit fake)\nbuilt with fake for testing"); sys.exit(0)
if "--list-devices" in args:
    print("Available devices:\n  (none)"); sys.exit(0)

def opt(*names, default=None):
    for n in names:
        if n in args:
            return args[args.index(n) + 1]
    return default

port = int(opt("--port", default="8080"))
host = opt("--host", default="127.0.0.1")
alias = opt("--alias", default="model")
model = opt("-m", "--model", default="")
ctx = opt("-c", "--ctx-size", default="?")
rpc = [x for x in (opt("--rpc", default="") or "").split(",") if x]
embeddings = "--embeddings" in args

if not os.path.isfile(model):
    print(f"error: model file {model} not found", file=sys.stderr); sys.exit(1)
if os.environ.get("FAKE_LLAMA_FAIL_CTX") and int(ctx) > int(os.environ["FAKE_LLAMA_FAIL_CTX"]):
    print(f"error: not enough memory for ctx {ctx}", file=sys.stderr); sys.exit(1)

def ping(ep):
    h, p = ep.rsplit(":", 1)
    with socket.create_connection((h, int(p)), timeout=5) as s:
        s.sendall(b"PING\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(100)
            if not chunk:
                raise IOError("closed")
            data += chunk
        return data.decode().strip().split(" ", 1)[1]

peers = []
for ep in rpc:
    try:
        peers.append(ping(ep))
    except Exception as e:
        print(f"error: failed to connect to RPC server {ep}: {e}", file=sys.stderr); sys.exit(1)
print(f"fake llama-server: model={os.path.basename(model)} alias={alias} ctx={ctx} rpc peers={peers}", flush=True)

state = {"ready": False}
def load():
    time.sleep(float(os.environ.get("FAKE_LOAD_SECONDS", "1.5")))
    state["ready"] = True
    print("fake llama-server: model loaded", flush=True)
threading.Thread(target=load, daemon=True).start()

def watchdog():
    while True:
        time.sleep(2)
        for ep in rpc:
            try:
                ping(ep)
            except Exception as e:
                print(f"error: lost RPC server {ep}: {e}", file=sys.stderr, flush=True); os._exit(1)
threading.Thread(target=watchdog, daemon=True).start()

def reply_text():
    return f"Hello from {alias} on {os.environ.get('LLMESH_NAME','?')}; rpc peers: {','.join(peers) or 'none'}; file: {os.path.basename(model)}"

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send_json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            if state["ready"]:
                self.send_json(200, {"status": "ok"})
            else:
                self.send_json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        elif self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": alias, "object": "model"}]})
        else:
            self.send_json(404, {"error": {"message": "not found"}})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.headers.get("Authorization"):
            self.send_json(500, {"error": {"message": "llmesh should have stripped the API key"}}); return
        if not state["ready"]:
            self.send_json(503, {"error": {"message": "Loading model"}}); return
        hops = self.headers.get("x-llmesh-hops", "")
        if self.path == "/v1/chat/completions":
            text = reply_text() + f"; model field: {body.get('model')}; hops: {hops}"
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                def chunk(s):
                    b = s.encode()
                    self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n"); self.wfile.flush()
                words = text.split(" ")
                for i, w in enumerate(words):
                    ev = {"id": "c1", "object": "chat.completion.chunk", "created": int(time.time()), "model": alias,
                          "choices": [{"index": 0, "delta": {"content": (w if i == 0 else " " + w)}, "finish_reason": None}]}
                    chunk(f"data: {json.dumps(ev)}\n\n")
                    time.sleep(0.05)
                ev = {"id": "c1", "object": "chat.completion.chunk", "created": int(time.time()), "model": alias,
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                chunk(f"data: {json.dumps(ev)}\n\n")
                chunk("data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            else:
                self.send_json(200, {"id": "c1", "object": "chat.completion", "created": int(time.time()), "model": alias,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        elif self.path == "/v1/completions":
            self.send_json(200, {"id": "c2", "object": "text_completion", "created": int(time.time()), "model": alias,
                "choices": [{"index": 0, "text": reply_text(), "finish_reason": "stop"}]})
        elif self.path == "/v1/embeddings":
            if not embeddings:
                self.send_json(501, {"error": {"code": 501, "message": "This server does not support embeddings. Start it with `--embeddings`", "type": "not_supported_error"}}); return
            self.send_json(200, {"object": "list", "model": alias, "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}]})
        else:
            self.send_json(404, {"error": {"message": "not found"}})

ThreadingHTTPServer((host, port), H).serve_forever()
