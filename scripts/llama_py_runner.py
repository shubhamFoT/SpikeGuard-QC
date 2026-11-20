#!/usr/bin/env python3
"""ExecuTorch Python runner wrapper for models/llama_et/model.pte (XNNPACK)."""
import argparse
from pathlib import Path

from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seq_len", type=int, default=400)
    args, _ = ap.parse_known_args()

    model_dir = str(Path(args.model_path).parent)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = ExecuTorchModelForCausalLM.from_pretrained(model_dir)

    # optimum-executorch prints the native PyTorchObserver stats line automatically
    out = model.text_generation(tokenizer=tok, prompt=args.prompt, max_seq_len=args.seq_len)

    # Guarantee edge_pipeline_runner.py extracts only the assistant's reply (not the system prompt)
    if out.startswith(args.prompt):
        gen = out[len(args.prompt):]
    elif "assistant\n\n" in out:
        gen = out.split("assistant\n\n", 1)[-1]
    else:
        gen = out

    print(args.prompt + gen.strip())


if __name__ == "__main__":
    main()
