#!/usr/bin/env python3
"""Persistent localhost warm-cache server for ExecuTorch Llama-3.2-1B (model.pte)."""
import argparse
import ctypes
import json
import os
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

try:
    _LIBC = ctypes.CDLL(None)
except Exception:
    _LIBC = None


def _flush_all() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if _LIBC is not None:
        try:
            _LIBC.fflush(None)
        except Exception:
            pass


def _capture_fds(fn):
    """Capture both Python and C++ stdout/stderr (including PyTorchObserver stats)."""
    _flush_all()
    orig_out = os.dup(1)
    orig_err = os.dup(2)
    tmp_out = tempfile.TemporaryFile()
    tmp_err = tempfile.TemporaryFile()
    try:
        os.dup2(tmp_out.fileno(), 1)
        os.dup2(tmp_err.fileno(), 2)
        result = fn()
    finally:
        _flush_all()
        os.dup2(orig_out, 1)
        os.dup2(orig_err, 2)
        os.close(orig_out)
        os.close(orig_err)
    tmp_out.seek(0)
    tmp_err.seek(0)
    cap_out = tmp_out.read().decode("utf-8", errors="replace")
    cap_err = tmp_err.read().decode("utf-8", errors="replace")
    tmp_out.close()
    tmp_err.close()
    return result, cap_out, cap_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.expanduser("~/llama_et"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    model_dir = str(Path(args.model_dir).expanduser())
    t0 = time.perf_counter()
    print(f"[llama_daemon] Loading ExecuTorch model from {model_dir} (one-time)...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = ExecuTorchModelForCausalLM.from_pretrained(model_dir)
    print(f"[llama_daemon] READY in {time.perf_counter() - t0:.1f}s -> http://{args.host}:{args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # keep terminal clean

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = req["prompt"]
            seq_len = int(req.get("seq_len", 400))

            t_req = time.perf_counter()
            out, cap_out, cap_err = _capture_fds(
                lambda: model.text_generation(tokenizer=tok, prompt=prompt, max_seq_len=seq_len)
            )
            dt = time.perf_counter() - t_req

            if out.startswith(prompt):
                gen = out[len(prompt):]
            elif "assistant\n\n" in out:
                gen = out.split("assistant\n\n", 1)[-1]
            else:
                gen = out

            payload = json.dumps({
                "text": prompt + gen.strip(),
                "stdout": cap_out,
                "stderr": cap_err,
            }).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            print(f"[llama_daemon] Served request in {dt:.2f}s (seq_len={seq_len})", flush=True)

    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
