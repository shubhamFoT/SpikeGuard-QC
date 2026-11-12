"""Measure Llama time-to-first-token, decode speed and wall time on the target device.
   python -m scripts.benchmark_llm --runner cmake-out/examples/models/llama/llama_main \
        --model llama3_2_1b.pte --tokenizer tokenizer.model --n 10"""
import argparse
import json
import statistics as st

from advisor.llama_runner import LlamaAdvisor

CASES = [
    dict(category="MACHINE_WEAR_SUSPECT", defect_class="scratches", severity="major", rate=0.4,
         window_n=30, zone="warning", loop_ma=11.9),
    dict(category="PROCESS_OR_MATERIAL_SUSPECT", defect_class="inclusion", severity="major", rate=0.5,
         window_n=30, zone="healthy", loop_ma=4.1),
    dict(category="STOP_AND_INSPECT", defect_class="crazing", severity="critical", rate=0.6,
         window_n=30, zone="critical", loop_ma=19.6),
    dict(category="MACHINE_HEALTH_UNAVAILABLE", defect_class="patches", severity="minor", rate=0.4,
         window_n=30, zone=None, loop_ma=None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="bench_llm.json")
    a = ap.parse_args()
    adv = LlamaAdvisor(a.runner, a.model, a.tokenizer)
    rows = [adv.advise(CASES[i % len(CASES)])[1] for i in range(a.n)]
    ok = [r for r in rows if r["source"] == "llama"]
    res = {"calls": a.n, "llama_ok": len(ok), "json_valid_rate": len(ok) / a.n}
    for k in ("wall_ms", "load_ms", "ttft_ms", "decode_tok_s", "generated_tokens"):
        vals = [r[k] for r in ok if k in r]
        if vals:
            res[k + "_median"] = st.median(vals)
    print(json.dumps(res, indent=2))
    json.dump({"summary": res, "rows": rows}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
