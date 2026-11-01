# llmesh

**Pool a few ordinary computers into one LLM server.** llmesh is a single binary for Windows, Linux and macOS (x64 and arm64). Run it on several machines and they split one model between them, so every machine's CPU, GPU and RAM is used. Then point any OpenAI-compatible tool at `http://localhost:8080/v1`, on any of the machines.

- **Distributed inference** through llama.cpp's RPC backend: the model's layers are spread across every machine in the group.
- **Zero-setup OpenAI API** on every node: chat completions, completions, embeddings and models. Any API key and any `model` name work.
- **Works over the internet:** private groups connect with QUIC hole punching, and fall back to a relay when a direct path isn't possible. Machines on the same LAN find each other automatically.
- **Hugging Face model picker** in the dashboard. It shows each quantization's size and whether it fits in the group's combined memory.
- **Voting:** members vote on which model the group runs, or the admin decides.

## Quick start

Download the binary for your platform from the [releases page](../../releases), then:

```sh
# On every machine on the same network: that's it.
llmesh
```

On first run llmesh downloads the matching llama.cpp release (CUDA, Vulkan, Metal or CPU, picked automatically) and the default model (`unsloth/Qwen3-8B-GGUF:Q4_K_M`, about 5 GB). The dashboard is at <http://localhost:8080/>.

To share machines over the internet, create a group:

```sh
# On the admin's machine
llmesh group create "Friends"
#   ...prints:  llmesh join llmesh1ahb7hrc4...

# On everyone else's machine
llmesh join llmesh1ahb7hrc4...
```

After that, a plain `llmesh` restarts in the saved group and reconnects to whoever is online.

| Command | What it does |
|---|---|
| `llmesh` / `llmesh start` | Run in the saved group, or the LAN group if you haven't joined one |
| `llmesh group create "<name>"` | Create a group (you become admin), print an invite and start |
| `llmesh join <invite>` | Join a group and start |
| `llmesh invite` | Print a fresh invite listing the members online right now |
| `llmesh groups` | List saved groups (`*` marks the active one) |
| `llmesh leave` | Leave the active group and go back to the LAN group |

Useful `start` flags: `--model REF` (admin), `--port 8080`, `--worker-only` (lend resources, never be the head), `--public-api` (listen on all interfaces), `--ctx N`, `--data-dir DIR`, `--backend cuda|vulkan|metal|cpu`, `--llama-dir DIR` (use your own llama.cpp build), `--name NAME`, `--embeddings`, `--no-relay`, `--no-lan`.

Model references can be `owner/repo:QUANT`, `owner/repo` (Q4_K_M by default), `owner/repo/file.gguf`, a local path, or an https URL. Split GGUFs (`-00001-of-0000N.gguf`) are handled automatically.

## Using the API

Every node answers on `http://localhost:8080/v1` and forwards requests to whichever node is running the model.

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anything", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="anything")
for chunk in client.chat.completions.create(
    model="anything",
    messages=[{"role": "user", "content": "Explain QUIC in one paragraph."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

`GET /v1/models` lists the group's active model. Errors use OpenAI's JSON error format. Embeddings need an embedding model: llama-server only serves `/v1/embeddings` when started with `--embeddings`, which llmesh adds automatically when the model name looks like an embedding model (or when the head is started with `--embeddings`).

## How layer splitting works

1. Every node runs llama.cpp's `rpc-server`, bound to `127.0.0.1` only. It exposes the node's GPUs (or its CPU and RAM if it has no GPU) as a remote compute device.
2. The group elects one **head**: the longest-running node that has llama.cpp ready, is on the group's llama.cpp version, and isn't `--worker-only`. Everyone computes the same answer from the gossiped member list, so there's no extra coordination step. When membership has been stable for about 5 seconds, the head downloads the model and starts `llama-server`.
3. For each other member, the head opens a local port that tunnels through an encrypted iroh connection to that member's `rpc-server`. It passes all of them to `llama-server --rpc`. If the head has no GPU of its own, its local `rpc-server` is added too, so its CPU and RAM are used as well.
4. llama.cpp's automatic fitting (`-fit`, on by default) places layers across all those devices according to their free memory. If loading fails, llmesh retries with a smaller context.
5. When someone joins or leaves, the head restarts `llama-server` with the new split. If the head disappears, the next-oldest node takes over, typically within 15 to 20 seconds.

All nodes use the same pinned llama.cpp release (currently `b12017`; override it with `LLMESH_LLAMA_TAG`) because the RPC protocol changes between versions. A node on a different release is left out of the split and flagged in the dashboard. Recent releases ship the RPC server binary as `ggml-rpc-server` (older ones as `rpc-server`); llmesh accepts either name, also under `--llama-dir`.

## Performance: what to expect

- **Tokens per second are limited by latency between nodes.** Generating each token passes activations through every machine in the split, one after another. A wired gigabit LAN works well. Wi-Fi is noticeably slower. Splitting a model over the internet works, but expect a few tokens per second at best.
- **The first load is slow.** The head uploads each worker's share of the weights through the tunnel. `rpc-server` runs with `-c` (cache), so later loads of the same model read the weights from the worker's own disk instead of sending them again.
- **The main benefit is capacity, not speed.** A model that runs on one machine runs faster there. llmesh lets you run models that are bigger than any single machine you own can hold.
- Put the head on your best-connected, fastest machine by starting it first, and use `--worker-only` on machines that should only lend resources.

## Security model

- **Group membership is the security boundary.** An invite contains the group's 32-byte secret, so anyone with the invite can join and use the group's model. Share invites like passwords.
- **Mutual authentication on every connection.** Each node has a long-term iroh key, and every connection is QUIC with TLS bound to that key. The first stream on each connection carries `HMAC-SHA256(secret, "llmesh-auth" ‖ sender id ‖ receiver id)`, in both directions. Because both TLS-authenticated endpoint ids are inside the MAC, a captured proof can't be replayed by anyone else or on any other connection. No other stream is served until authentication succeeds.
- **llama.cpp is never exposed to the network.** `rpc-server` and `llama-server` listen on `127.0.0.1` only. All traffic between machines (RPC, API requests, model files, gossip) goes through the authenticated, encrypted iroh connections.
- **Model choices are signed.** The group config (active model, selection mode, voting round, context size) is signed with the admin's key, and the highest sequence number wins. Votes are signed with each voter's key. In the LAN group there is no admin: any member may sign a model choice and the latest one wins.
- **The local API has no authentication**, like most local LLM servers, and listens on `127.0.0.1` by default. `--public-api` makes it reachable from your network; only use it on networks you trust.
- **Dashboard actions** (changing the model, votes, settings) are accepted only from the local machine, only with an `X-Llmesh: 1` header (so other websites can't forge them), and only when the Host header is `localhost` (guards against DNS rebinding). Admin actions work only on the node that holds the admin key.
- **LAN group caveat:** the LAN group uses a fixed, well-known secret so it works with zero setup. Anyone on your LAN running llmesh can join it. Use a private group on networks you don't control.
- Relays (n0's public iroh relays) only forward end-to-end encrypted QUIC packets; they can't read the traffic.

## Firewall prompts

- **Windows:** the first time you run llmesh, Windows Defender Firewall asks whether to allow it on private and/or public networks. Allow **private networks** for LAN discovery and direct connections. Internet groups still work through hole punching or a relay if you decline. `rpc-server.exe` and `llama-server.exe` only listen on localhost, so they may not need a rule at all.
- **macOS:** you may see "Do you want the application 'llmesh' to accept incoming network connections?". Click **Allow**. If you downloaded the binary with a browser, clear the quarantine flag first with `xattr -d com.apple.quarantine ./llmesh`.
- **Linux:** if you run a firewall, allow UDP port 47474 (LAN discovery) and the QUIC traffic. Use `--p2p-port` to pin the QUIC port so it can be opened.

## Troubleshooting

- **The dashboard shows no other members on the LAN.** Check that UDP broadcast on port 47474 isn't blocked (guest Wi-Fi networks often isolate clients). Use a private group instead: invites work across any network.
- **A peer shows as "relayed".** Hole punching failed, often because of symmetric NAT or carrier-grade NAT. It still works, with more latency.
- **"excluded" next to a node's llama.cpp version.** That node runs a different llama.cpp release. Make sure every machine runs the same llmesh version, and don't point `--llama-dir` at a different build.
- **The model never finishes loading, or keeps restarting.** Open "Head logs" in the dashboard. Common causes: the model doesn't fit (pick a smaller quantization, the picker shows what fits), or a worker dropped out mid-load.
- **CUDA or Vulkan doesn't work.** llmesh falls back to the next backend automatically when `llama-server --list-devices` fails. Force one with `--backend`. For CUDA, make sure the NVIDIA driver is recent enough for CUDA 12.
- **A gated model fails with 401/403.** Accept the license on Hugging Face and set `HF_TOKEN`.
- **Hugging Face is unreachable.** The head fetches model files from other members' caches when it can't download them.
- **Port 8080 is taken.** llmesh tries 8081, 8082 and so on, and prints the port it chose.
- More logs: `LLMESH_LOG=llmesh=debug llmesh`.

Files live in `dirs::data_dir()/llmesh` (keys, groups and llama.cpp) and `dirs::cache_dir()/llmesh/models` (models). On Linux that's `~/.local/share/llmesh` and `~/.cache/llmesh/models`. `--data-dir` puts everything in one folder.

## Environment variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Token for gated or private Hugging Face models |
| `HF_ENDPOINT` | Hugging Face mirror (default `https://huggingface.co`) |
| `LLMESH_LLAMA_TAG` | Override the pinned llama.cpp release |
| `LLMESH_LLAMA_BASE_URL` | Mirror for llama.cpp release downloads |
| `LLMESH_MODELS_DIR` | Model cache location |
| `LLMESH_NO_RELAY=1` | Disable relays and DNS lookup (direct connections only) |
| `LLMESH_LOG` | Log filter, for example `llmesh=debug` |

## Building and testing

```sh
cargo build --release
cargo test                         # unit tests: invites, auth, signatures, tallying, model refs, ...
pip install openai && python3 tests/e2e/run_e2e.py   # 3-node end-to-end test with fake llama.cpp
```

The end-to-end test runs three real llmesh nodes on one machine (relays off, connected through invite addresses), plus a two-node LAN-beacon test. It uses fake `rpc-server` and `llama-server` scripts and a local mock of Hugging Face and the GitHub releases. It covers joining by invite, rejecting a wrong secret, election, step-down when an older head appears, RPC tunnels, API proxying from a non-head node (OpenAI SDK, streaming), admin model changes, votes, head failover, resumable downloads and fetching a model from a peer. **Real llama.cpp inference has not been exercised by these tests:** the fakes check the plumbing, not model execution.

Releases are built by `.github/workflows/release.yml` for Linux, macOS and Windows on x64 and arm64 when a `v*` tag is pushed.

## License

MIT, see [LICENSE](LICENSE).
