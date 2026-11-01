#!/bin/sh
# Quick 2-node smoke test (manual). Usage: tests/e2e/smoke.sh <scratch dir>
S=$1; rm -rf "$S/a" "$S/b"; mkdir -p "$S"
export HF_ENDPOINT=http://127.0.0.1:18999 LLMESH_LLAMA_BASE_URL=http://127.0.0.1:18999/releases LLMESH_NO_RELAY=1 LLMESH_BACKEND=cpu
BIN=$(dirname "$0")/../../target/debug/llmesh
LLMESH_NAME=A $BIN --data-dir "$S/a" group create "Test" --model test/Tiny-GGUF --port 18080 --no-lan > "$S/a.log" 2>&1 &
sleep 3
INV=$(grep -o 'llmesh1[a-z0-9]*' "$S/a.log" | head -1)
LLMESH_NAME=B $BIN --data-dir "$S/b" join "$INV" --port 18081 --no-lan > "$S/b.log" 2>&1 &
