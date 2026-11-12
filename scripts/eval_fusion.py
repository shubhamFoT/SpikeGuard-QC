"""Answer the mentors' question: how well does the COMBINED system separate visual defects from
machine-health warnings?

Method (state it in your report): each episode is 40 parts. Part-level 'reject' flags are drawn from
the detector's MEASURED behaviour on defective and clean frames, and machine zones are drawn from the
twin's MEASURED zones on healthy / mild-fault / severe-fault windows. The ground-truth label is the
scenario that generated the episode. So component errors (false alarms, misses) propagate into the
result. It is still a simulation of episodes, not data from a physical line - say so.

   python -m scripts.eval_fusion --recall 0.9 --fpr 0.05        # example numbers, replace with yours
"""
import argparse
import json
import random

from sklearn.metrics import confusion_matrix, accuracy_score

from common.schemas import MachineState
from fusion.rules import (ISOLATED_DEFECT, MACHINE_WEAR, NORMAL, PROCESS_OR_MATERIAL,
                          RollingRate, STOP_AND_INSPECT, fuse)

SCENARIOS = {                       # name: (defect probability per part, zone pool name, accepted categories)
    "normal":           (0.05, "healthy",  {NORMAL, ISOLATED_DEFECT}),
    "process_problem":  (0.50, "healthy",  {PROCESS_OR_MATERIAL}),
    "machine_wear":     (0.50, "warning",  {MACHINE_WEAR}),
    "machine_critical": (0.20, "critical", {STOP_AND_INSPECT}),
}


def episode(name, rng, recall, fpr, zone_pools, n=40):
    p_def, zone_pool, _ = SCENARIOS[name]
    r, last, st = RollingRate(), False, None
    for _ in range(n):
        truly_defective = rng.random() < p_def
        flagged = rng.random() < (recall if truly_defective else fpr)     # detector behaviour
        r.add(flagged)
        last = flagged
        zone = rng.choice(zone_pools[zone_pool])                          # twin behaviour
        st = MachineState(0.0, 4.0, zone, 0.0, 1.0)
    return fuse(r, last, st).category


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall", type=float, default=0.9, help="detector recall on defective parts (measured)")
    ap.add_argument("--fpr", type=float, default=0.05, help="detector false-positive rate on clean parts (measured)")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    # Replace these with the zone frequencies you MEASURE from the twin on healthy/mild/severe windows.
    zone_pools = {"healthy": ["healthy"] * 98 + ["warning"] * 2,
                  "warning": ["warning"] * 80 + ["healthy"] * 10 + ["critical"] * 10,
                  "critical": ["critical"] * 95 + ["warning"] * 5}
    names = list(SCENARIOS)
    y_true, y_pred = [], []
    for name in names:
        for _ in range(a.episodes):
            cat = episode(name, rng, a.recall, a.fpr, zone_pools)
            y_true.append(name)
            y_pred.append(name if cat in SCENARIOS[name][2] else "other:" + cat)
    correct = [t == p for t, p in zip(y_true, y_pred)]
    per = {n: sum(c for t, c in zip(y_true, correct) if t == n) / a.episodes for n in names}
    print(json.dumps({"overall_accuracy": sum(correct) / len(correct), "per_scenario_accuracy": per}, indent=2))


if __name__ == "__main__":
    main()
