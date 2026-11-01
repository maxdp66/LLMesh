#!/usr/bin/env python3
"""Stand-in for llama.cpp's rpc-server used by the end-to-end tests.

Answers "PING\n" with "PONG <node name>\n" on every connection.
"""
import os, socket, sys, threading

def main():
    args = sys.argv[1:]
    host, port = "127.0.0.1", 50052
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-H", "--host"):
            host = args[i + 1]; i += 1
        elif a in ("-p", "--port"):
            port = int(args[i + 1]); i += 1
        elif a in ("-t", "--threads", "-d", "--device"):
            i += 1
        i += 1
    if host != "127.0.0.1":
        print(f"refusing to bind {host}: llmesh must keep rpc-server on loopback", file=sys.stderr)
        sys.exit(2)
    name = os.environ.get("LLMESH_NAME", f"rpc-{port}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(64)
    print(f"fake rpc-server {name} listening on {host}:{port}", flush=True)

    def handle(c):
        f = c.makefile("rwb")
        try:
            for line in f:
                if line.strip() == b"PING":
                    f.write(f"PONG {name}\n".encode()); f.flush()
        except Exception:
            pass
        finally:
            c.close()

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()

if __name__ == "__main__":
    main()
