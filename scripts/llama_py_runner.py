#!/usr/bin/env python3
"""ExecuTorch Python runner wrapper for models/llama_et/model.pte (XNNPACK).
Uses warm local daemon (127.0.0.1:8765) if running; falls back to direct load.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _try_daemon(prompt: str, seq_len: int) -> bool:
    url = os.environ.get("SPIKEGUARD_LLAMA_SERVER", "http://127.0.0.1:8765") + "/generate"
    body = json.dumps({"prompt": prompt, "seq_len": seq_len}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
        return False

    if data.get("stderr"):
        sys.stderr.write(data["stderr"])
        sys.stderr.flush()
    if data.get("stdout"):
        sys.stdout.write(data["stdout"])
    print(data["text"])
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seq_len", type=int, default=400)
    args, _ = ap.parse_known_args()

    if _try_daemon(args.prompt, args.seq_len):
        return

    from optimum.executorch import ExecuTorchModelForCausalLM
    from transformers import AutoTokenizer

    model_dir = str(Path(args.model_path).parent)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = ExecuTorchModelForCausalLM.from_pretrained(model_dir)

    out = model.text_generation(tokenizer=tok, prompt=args.prompt, max_seq_len=args.seq_len)

    if out.startswith(args.prompt):
        gen = out[len(args.prompt):]
    elif "assistant\n\n" in out:
        gen = out.split("assistant\n\n", 1)[-1]
    else:
        gen = out

    print(args.prompt + gen.strip())


if __name__ == "__main__":
    main()
