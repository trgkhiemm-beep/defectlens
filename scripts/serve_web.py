"""Serve web/ locally with the same cross-origin-isolation headers as the HF Space, so
ONNX Runtime Web can use multi-threaded WASM (SharedArrayBuffer needs COOP + COEP).

  python scripts/serve_web.py            ->  http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HEADERS = {"Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp"}


class IsolatedHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        for k, v in HEADERS.items():
            self.send_header(k, v)
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=Path(__file__).resolve().parents[1] / "web", type=Path)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    handler = functools.partial(IsolatedHandler, directory=str(args.dir))
    print(f"serving {args.dir} with {HEADERS} on http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
