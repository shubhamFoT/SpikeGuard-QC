"""Run a Llama 3.2 Instruct model through ExecuTorch's `llama_main` runner and parse the reply.

Why a subprocess: the runner is the officially documented way to execute a Llama .pte, and
the advisor is event-driven (rare), so paying the model-load cost per event is acceptable. The
load time is measured and reported in the benchmark. Flag names below match the Arm learning
path; run `llama_main --help` for the version you built."""
import json
import re
import subprocess
import time
from typing import Optional

from advisor.prompt import build_prompt

REQUIRED = ("summary", "likely_cause", "action", "urgency")


def template_advice(ctx: dict) -> dict:
    """Deterministic fallback. Used when Llama fails, times out, or returns bad JSON, and as the
    'no-LLM' baseline in the evaluation."""
    cat, dc = ctx["category"], ctx.get("defect_class") or "defect"
    table = {
        "ISOLATED_DEFECT": (f"Single {dc} detected.", "Random local flaw.", "Reject the part and continue.", "low"),
        "PROCESS_OR_MATERIAL_SUSPECT": (f"Repeated {dc} defects.", "Likely material or process settings.",
                                        "Check upstream settings and material batch.", "medium"),
        "MACHINE_WEAR_SUSPECT": (f"Repeated {dc} defects with bearing warning.", "Possible machine wear.",
                                 "Schedule a bearing inspection at the next stop.", "medium"),
        "MACHINE_WARNING_NO_DEFECT_TREND": ("Bearing health in warning.", "Early machine wear.",
                                            "Plan an inspection; parts are still good.", "low"),
        "STOP_AND_INSPECT": ("Bearing health critical.", "Severe bearing damage likely.",
                             "Stop the machine and inspect before restart.", "high"),
        "MACHINE_HEALTH_UNAVAILABLE": ("Machine-health data unavailable.", "Sensor or link fault.",
                                       "Continue on vision only and check the node.", "medium"),
    }
    s, c, a, u = table.get(cat, ("Normal operation.", "-", "No action.", "low"))
    return {"summary": s, "likely_cause": c, "action": a, "urgency": u}


def _first_json_object(text: str) -> Optional[dict]:
    """Find the first balanced {...} that parses as JSON and has the required keys."""
    for m in re.finditer(r"\{", text):
        depth = 0
        for j in range(m.start(), len(text)):
            depth += text[j] == "{"
            depth -= text[j] == "}"
            if depth == 0:
                try:
                    obj = json.loads(text[m.start(): j + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict) and all(k in obj for k in REQUIRED):
                    return obj
                break
    return None


def extract_answer(stdout: str) -> Optional[dict]:
    """Take everything after the LAST 'ANSWER:' (works whether or not the runner echoes the
    prompt), drop log/observer lines, then parse the first valid JSON object."""
    body = stdout.rsplit("ANSWER:", 1)[-1]
    clean = "\n".join(l for l in body.splitlines()
                      if not l.startswith(("PyTorchObserver", "I 0", "W 0", "E 0")) and "executorch:" not in l)
    obj = _first_json_object(clean)
    if obj:
        obj["urgency"] = str(obj["urgency"]).lower() if str(obj["urgency"]).lower() in ("low", "medium", "high") else "medium"
    return obj


def parse_stats(stdout: str) -> dict:
    """ExecuTorch prints a 'PyTorchObserver {json}' line with token counts and timestamps (ms).
    Field names can differ between versions, so everything here is optional."""
    for line in stdout.splitlines():
        if line.startswith("PyTorchObserver"):
            try:
                s = json.loads(line.split("PyTorchObserver", 1)[1])
            except json.JSONDecodeError:
                continue
            out = {}
            gen, t0 = s.get("generated_tokens"), s.get("inference_start_ms")
            t1, tp, tf = s.get("inference_end_ms"), s.get("prompt_eval_end_ms"), s.get("first_token_ms")
            if None not in (s.get("model_load_start_ms"), s.get("model_load_end_ms")):
                out["load_ms"] = s["model_load_end_ms"] - s["model_load_start_ms"]
            if None not in (tf, t0):
                out["ttft_ms"] = tf - t0
            if None not in (gen, t1, tp) and t1 > tp:
                out["decode_tok_s"] = gen / ((t1 - tp) / 1000.0)
            if gen is not None:
                out["generated_tokens"] = gen
            return out
    return {}


class LlamaAdvisor:
    def __init__(self, runner, model, tokenizer, threads=4, seq_len=320, temperature=0.2, timeout_s=180):
        self.cmd = [runner, f"--model_path={model}", f"--tokenizer_path={tokenizer}",
                    f"--cpu_threads={threads}", f"--seq_len={seq_len}", f"--temperature={temperature}"]
        self.timeout_s = timeout_s

    def advise(self, ctx: dict):
        """Returns (advice_dict, meta). Never raises: any failure falls back to the template."""
        t0 = time.time()
        meta = {"source": "llama"}
        try:
            p = subprocess.run(self.cmd + [f"--prompt={build_prompt(ctx)}"], capture_output=True,
                               text=True, timeout=self.timeout_s)
            text = p.stdout + "\n" + p.stderr
            meta.update(parse_stats(text))
            adv = extract_answer(p.stdout)
            if adv is None:
                raise ValueError("no valid JSON in model output")
        except Exception as e:                     # timeout, missing binary, bad JSON ...
            adv, meta = template_advice(ctx), {"source": "template", "reason": str(e)[:120]}
        meta["wall_ms"] = round((time.time() - t0) * 1000, 1)
        return adv, meta
