#!/usr/bin/env python3
"""End-to-end test: three llmesh nodes on one machine with fake llama.cpp binaries.

    cargo build && python3 tests/e2e/run_e2e.py [--keep]

What runs for real: llmesh itself (iroh QUIC between nodes, auth, gossip, election, RPC and
HTTP tunnels, downloads, API proxy). What is faked: llama.cpp (tests/e2e/fake_*.py), Hugging
Face and GitHub releases (tests/e2e/mock_server.py). Relays and n0 DNS are disabled; nodes find
each other through the invite's direct addresses (and LAN beacons in the LAN test).
"""
import base64, json, os, random, shutil, signal, subprocess, sys, tempfile, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.environ.get("LLMESH_BIN", os.path.join(ROOT, "target", "debug", "llmesh"))
MOCK_PORT = 18999
KEEP = "--keep" in sys.argv
WORK = tempfile.mkdtemp(prefix="llmesh-e2e-")
procs = {}
passed = []

def log(msg):
    print(f"[e2e] {msg}", flush=True)

def env_for(name, extra=None):
    e = dict(os.environ)
    e.update({
        "HF_ENDPOINT": f"http://127.0.0.1:{MOCK_PORT}",
        "LLMESH_LLAMA_BASE_URL": f"http://127.0.0.1:{MOCK_PORT}/releases",
        "LLMESH_NO_RELAY": "1",
        "LLMESH_BACKEND": "cpu",
        "LLMESH_NAME": name,
        "FAKE_LOAD_SECONDS": "1",
        "LLMESH_LOG": "llmesh=debug,warn",
        "NO_PROXY": "127.0.0.1,localhost",
    })
    e.update(extra or {})
    return e

def start(name, args, port, extra_env=None):
    d = os.path.join(WORK, name)
    os.makedirs(d, exist_ok=True)
    out = open(os.path.join(WORK, f"{name}.log"), "a")
    cmd = [BIN, "--data-dir", d] + args + ["--port", str(port)]
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env_for(name, extra_env), start_new_session=True)
    procs[name] = (p, port)
    return p

def stop(name):
    p, _ = procs.pop(name)
    os.killpg(p.pid, signal.SIGKILL)  # kill llmesh and its llama children like a crash/power-off
    p.wait()

def logtext(name):
    """The node's own log plus the llama-server output it shows on the dashboard."""
    with open(os.path.join(WORK, f"{name}.log")) as f:
        text = f.read()
    if name in procs:
        try:
            text += "\n".join(http(procs[name][1], "/mesh/api/logs", timeout=5)[1]["lines"])
        except Exception:
            pass
    return text

def http(port, path, body=None, headers=None, timeout=30):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")

def status(name):
    return http(procs[name][1], "/mesh/api/status")[1]

def wait_until(what, fn, timeout=90):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = fn()
            if last:
                return last
        except Exception as e:  # node still starting
            last = e
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what} (last: {last!r})")

def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    passed.append(name)
    log(f"PASS {name}")

def head_info(node):
    s = status(node)
    if not s.get("head"):
        return None
    if s["head"] == s["id"]:
        return s["me"]
    for p in s["peers"]:
        if p["info"] and p["info"]["id"] == s["head"]:
            return p["info"]
    return None

def ready_with(node, model_substr, n_split):
    h = head_info(node)
    return h if (h and h["head_state"]["state"] == "ready" and model_substr in h["model"] and len(h["split"]) == n_split) else None

def chat(port, content="hi", **kw):
    st, r = http(port, "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": content}]}, {"Authorization": "Bearer sk-whatever"}, **kw)
    assert st == 200, (st, r)
    return r["choices"][0]["message"]["content"]

def main():
    from openai import OpenAI
    mock = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_server.py"), str(MOCK_PORT)], stdout=open(os.path.join(WORK, "mock.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
    procs["mock"] = (mock, MOCK_PORT)
    time.sleep(0.7)
    log(f"work dir {WORK}")

    # ---- group create + invite join -------------------------------------------------------
    start("A", ["group", "create", "E2E Group", "--model", "test/Tiny-GGUF", "--no-lan"], 18180, {"FAKE_LLAMA_FAIL_CTX": "4096"})
    invite = wait_until("invite from A", lambda: next((w for w in logtext("A").split() if w.startswith("llmesh1")), None), 20)
    check("invite printed", invite.startswith("llmesh1"))
    time.sleep(1.0)  # B must be older than C for the failover step
    start("B", ["join", invite, "--no-lan"], 18181)
    time.sleep(1.0)
    start("C", ["join", invite, "--no-lan"], 18182)

    h = wait_until("A to be head with B and C in the split", lambda: ready_with("C", "test/Tiny-GGUF", 2), 90)
    check("election picks oldest node", h["name"] == "A")
    sA = status("A")
    check("peers connected directly", all(p["path"] == "direct" for p in sA["peers"] if p["online"]) and sum(p["online"] for p in sA["peers"]) == 2, json.dumps(sA["peers"])[:400])
    check("ctx retry ladder after load failure", "--ctx-size 4096" in logtext("A") and "not enough memory for ctx 8192" in logtext("A"))
    check("release downloaded + extracted", os.path.exists(os.path.join(WORK, "B", "llama.cpp", "b12017", "cpu", ".complete")))

    # ---- API through a non-head node, plain HTTP --------------------------------------------
    text = chat(18182)
    check("chat via non-head node", "on A" in text and "file: Tiny-Q4_K_M.gguf" in text, text)
    peers = text.split("rpc peers: ")[1].split(";")[0].split(",")
    check("model split over all 3 machines (B, C + head's own rpc-server)", sorted(peers) == ["A", "B", "C"], text)
    check("any model name works (rewritten to alias)", "model field: test/Tiny-GGUF:Q4_K_M" in text, text)
    st, models = http(18182, "/v1/models")
    check("/v1/models answered locally", st == 200 and models["data"][0]["id"] == "test/Tiny-GGUF:Q4_K_M", models)
    st, err = http(18182, "/v1/embeddings", {"model": "x", "input": "hi"})
    check("errors in OpenAI format", st == 501 and "message" in err["error"] and "type" in err["error"], (st, err))

    # ---- OpenAI SDK, streaming -------------------------------------------------------------
    client = OpenAI(base_url="http://127.0.0.1:18182/v1", api_key="anything")
    r = client.chat.completions.create(model="whatever", messages=[{"role": "user", "content": "hello"}])
    check("OpenAI SDK non-streaming", "Hello from" in r.choices[0].message.content)
    t0 = time.time(); stamps = []; acc = ""
    for ev in client.chat.completions.create(model="whatever", messages=[{"role": "user", "content": "hello"}], stream=True):
        if ev.choices and ev.choices[0].delta.content:
            stamps.append(time.time() - t0); acc += ev.choices[0].delta.content
    check("OpenAI SDK streaming (SSE not buffered)", len(stamps) > 5 and stamps[-1] - stamps[0] > 0.3 and "Hello from" in acc, f"{len(stamps)} chunks {stamps[:3]}..{stamps[-1:]}")
    rc = client.completions.create(model="x", prompt="once upon")
    check("completions endpoint", "Hello from" in rc.choices[0].text)

    # ---- auth rejection with a wrong secret ---------------------------------------------
    raw = base64.b32decode(invite[len("llmesh1"):].upper() + "=" * (-len(invite[len("llmesh1"):]) % 8))
    forged = raw[:1] + bytes(random.getrandbits(8) for _ in range(32)) + raw[33:]
    forged_inv = "llmesh1" + base64.b32encode(forged).decode().rstrip("=").lower()
    start("D", ["join", forged_inv, "--no-lan"], 18183)
    wait_until("A to reject D", lambda: "rejected connection" in logtext("A"), 30)
    time.sleep(3)
    sD = status("D")
    check("wrong group secret is rejected", not any(p["online"] for p in sD["peers"]) and not any(p["online"] and p["info"]["name"] == "D" for p in status("A")["peers"]))
    stop("D")

    # ---- dashboard API protections -----------------------------------------------------------
    st, _ = http(18180, "/mesh/api/model", {"model": "test/Tiny-GGUF:Q8_0"})
    check("mutation without X-Llmesh header refused", st == 403)
    st, _ = http(18180, "/mesh/api/model", {"model": "test/Tiny-GGUF:Q8_0"}, {"X-Llmesh": "1", "Host": "evil.example"})
    check("DNS-rebinding Host refused", st == 403)
    st, e = http(18181, "/mesh/api/model", {"model": "test/Tiny-GGUF:Q8_0"}, {"X-Llmesh": "1"})
    check("non-admin can't set the model", st == 400 and "admin" in e["error"]["message"], e)
    st, sv = http(18180, "/mesh/api/status", headers={"Host": "evil.example"})
    check("invite (group secret) hidden from non-local status", st == 200 and sv["invite"] is None and status("A")["invite"])

    # ---- admin model change ----------------------------------------------------------------
    st, _ = http(18180, "/mesh/api/model", {"model": "test/Tiny-GGUF:Q8_0"}, {"X-Llmesh": "1"})
    check("admin sets model", st == 200)
    wait_until("Q8_0 to be served", lambda: ready_with("C", "Q8_0", 2), 60)
    check("admin model change propagates", "file: Tiny-Q8_0.gguf" in chat(18182))

    # ---- votes -------------------------------------------------------------------------------
    st, _ = http(18181, "/mesh/api/vote", {"model": "test/Split-GGUF:Q2_K"}, {"X-Llmesh": "1"})
    st2, _ = http(18182, "/mesh/api/vote", {"model": "test/Split-GGUF:Q2_K"}, {"X-Llmesh": "1"})
    http(18180, "/mesh/api/vote", {"model": "test/Tiny-GGUF:Q4_K_M"}, {"X-Llmesh": "1"})
    time.sleep(4)
    check("votes are advisory in admin mode", "Q8_0" in head_info("C")["model"])
    tally = wait_until("tally on A", lambda: status("A")["tally"] if len(status("A")["tally"]) == 2 else None, 20)
    check("tally counts latest vote per member", tally[0] == {"model": "test/Split-GGUF:Q2_K", "votes": 2}, tally)
    st, e = http(18181, "/mesh/api/selection", {"mode": "vote"}, {"X-Llmesh": "1"})
    check("non-admin can't switch selection mode", st == 400)
    st, _ = http(18180, "/mesh/api/selection", {"mode": "vote"}, {"X-Llmesh": "1"})
    wait_until("vote winner to be served", lambda: ready_with("B", "Split-GGUF", 2), 60)
    text = chat(18181)
    check("vote mode runs the winner (split GGUF)", "Split-Q2_K-00001-of-00002.gguf" in text, text)
    check("both shards downloaded", os.path.exists(os.path.join(WORK, "A", "models", "test__Split-GGUF", "Q2_K", "Split-Q2_K-00002-of-00002.gguf")))

    # ---- head failover -----------------------------------------------------------------------
    stop("A")
    wait_until("B to take over", lambda: (lambda h: h and h["name"] == "B" and h["head_state"]["state"] == "ready" and len(h["split"]) == 1)(head_info("C")), 90)
    text = chat(18182)
    check("head failover to next-oldest node", "on B" in text and sorted(text.split("rpc peers: ")[1].split(";")[0].split(",")) == ["B", "C"], text)

    # ---- older head shows up (step-down) + model fetched from a peer while HF is down ---------
    open(os.path.join(HERE, ".hf_down"), "w").close()
    try:
        shutil.rmtree(os.path.join(WORK, "A", "models"))
        start("A", ["--no-lan"], 18180, {"LLMESH_TEST_STARTED_AT": "1"})
        wait_until("A (older) to become head again", lambda: (lambda h: h and h["name"] == "A" and h["head_state"]["state"] == "ready" and len(h["split"]) == 2)(head_info("C")), 120)
        check("newer head steps down for an older one", "stepping down" in logtext("B"))
        check("model fetched from a peer when HF is down", "from peer" in logtext("A") and "Split-Q2_K-00001-of-00002.gguf" in chat(18182))
    finally:
        os.remove(os.path.join(HERE, ".hf_down"))

    # ---- saved group survives restart; invite command ----------------------------------------
    out = subprocess.run([BIN, "--data-dir", os.path.join(WORK, "C"), "invite", "--port", "18182"], capture_output=True, text=True, env=env_for("C"))
    check("`llmesh invite` from a running member", out.stdout.strip().startswith("llmesh1"), out.stderr)
    out = subprocess.run([BIN, "--data-dir", os.path.join(WORK, "C"), "groups"], capture_output=True, text=True)
    check("`llmesh groups` lists the group", "E2E Group" in out.stdout and "*" in out.stdout, out.stdout)

    for n in ["A", "B", "C"]:
        stop(n)

    # ---- LAN group: zero-setup discovery with beacons ----------------------------------------
    lan_env = {"LLMESH_DISCOVERY_PORT": "47999"}
    start("L1", [], 18190, lan_env)
    time.sleep(1)
    start("L2", [], 18191, lan_env)
    try:
        h = wait_until("LAN nodes to find each other and elect a head", lambda: ready_with("L2", "Qwen3-8B", 1) or None, 40)
        check("LAN discovery (UDP beacons) + election", h["name"] == "L1")
    except AssertionError as e:
        # The default model isn't on the mock HF; discovery + election is what we check here.
        s = status("L2")
        ok = any(p["online"] and p["info"]["name"] == "L1" for p in s["peers"]) and s["head"] == status("L1")["id"]
        check("LAN discovery (UDP beacons) + election", ok, str(e))
    stop("L1"); stop("L2")

    log(f"ALL {len(passed)} CHECKS PASSED")

if __name__ == "__main__":
    try:
        main()
        if not KEEP:
            shutil.rmtree(WORK, ignore_errors=True)
    except Exception:
        log("FAILED; logs in " + WORK)
        raise
    finally:
        for n in list(procs):
            try:
                stop(n)
            except Exception:
                pass
