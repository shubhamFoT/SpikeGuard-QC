"""Build the Llama 3.2 Instruct prompt. The model gets FACTS and must not change the decision."""
import json
from pathlib import Path

SOP = json.loads((Path(__file__).resolve().parents[1] / "knowledge" / "sop.json").read_text())

SYSTEM = (
    "You are a quality-control assistant on a factory line. You are given facts from a vision "
    "inspection and a machine-health monitor. Use ONLY the facts given. Never change the "
    "pass/reject decision. Reply with one JSON object and nothing else, with keys: "
    '"summary" (max 20 words), "likely_cause" (max 20 words), "action" (max 25 words), '
    '"urgency" (low, medium or high).'
)

CATEGORY_HINT = {
    "ISOLATED_DEFECT": "A single defective part. Machine is healthy.",
    "PROCESS_OR_MATERIAL_SUSPECT": "Many recent parts are defective while the machine is healthy.",
    "MACHINE_WEAR_SUSPECT": "Many recent parts are defective and bearing health is in warning.",
    "MACHINE_WARNING_NO_DEFECT_TREND": "Bearing health is in warning but parts are still good.",
    "STOP_AND_INSPECT": "Bearing health is critical.",
    "MACHINE_HEALTH_UNAVAILABLE": "Machine-health data is unavailable; vision only.",
}


def build_user(ctx: dict) -> str:
    """ctx keys: category, defect_class, severity, rate, window_n, zone, loop_ma."""
    sop = SOP.get(ctx.get("defect_class") or "", {})
    lines = [
        f"Situation: {CATEGORY_HINT.get(ctx['category'], ctx['category'])}",
        f"Latest defect type: {ctx.get('defect_class') or 'none'}, severity: {ctx.get('severity')}",
        f"Defect rate in last {ctx.get('window_n')} parts: {ctx.get('rate', 0):.0%}",
        f"Machine health zone: {ctx.get('zone') or 'unavailable'}"
        + (f" ({ctx['loop_ma']:.1f} mA)" if ctx.get("loop_ma") is not None else ""),
    ]
    if sop:
        lines.append("Known causes for this defect: " + "; ".join(sop["likely_causes"]))
        lines.append("Usual first checks: " + "; ".join(sop["first_checks"]))
    return "\n".join(lines)


def build_prompt(ctx: dict) -> str:
    """Llama 3 chat format. No <|begin_of_text|>: the ExecuTorch runner adds BOS itself
    (if outputs look odd, check for a doubled BOS). Ends with 'ANSWER:' so the parser can find
    the start of the reply even if the runner echoes the prompt."""
    return ("<|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + build_user(ctx) + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\nANSWER:")
