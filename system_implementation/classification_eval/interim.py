"""Interim end-to-end matrix from the cases finished so far, using each benchmark's own scoring.

    python -m classification_eval.interim [logs/dgx/e2e]

DEXA has no native score; its row is the device-type P/R/F1 of the agents the MAS assigned a task
(the "+ MAS" row of T1). Partial runs follow dataset order, so they are not yet representative.
"""
import json
import sys
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

from .metrics import AGENT_DEVICE, scores
from .tasks import ROOT, TASKS, t1_dexa_device_types

TOTAL = {"dexa": 100, "homebench": 1000, "simuhome": 600, "homeassistant": 95}
META = {"config", "summary", "qed_leaf", "run", "dataset", "command", "runs", "failure", "comparison"}


def case_rows(base, benchmark):
    rows = []
    for file in sorted((base / benchmark).rglob("*.json")):
        if file.stem in META:
            continue
        try:
            row = json.loads(file.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(row, dict) and "id" in row and {"score", "evaluation", "episode_result", "events"} & set(row):
            rows.append(row)
    return rows


def fmt(value):
    return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    base = Path(argv[0]) if argv else ROOT / "logs" / "dgx" / "e2e"
    print(f"Interim end-to-end results ({base})\n")
    print("| Benchmark | Done | Main metrics | Errors | Mean latency (s) | QED_LEAF |")
    print("|---|---|---|---|---|---|")
    notes = []
    for benchmark in ("dexa", "homebench", "simuhome", "homeassistant"):
        rows = case_rows(base, benchmark)
        found = next((base / benchmark).rglob("qed_leaf.json"), None)
        leaf = json.loads(found.read_text())["summary"]["status"] if found else "-"
        done = f"{len(rows)}/{TOTAL[benchmark]}"
        if not rows:
            print(f"| {benchmark} | {done} | - | - | - | {leaf} |")
            continue
        if benchmark == "dexa":
            gold = {i["id"]: i["labels"] for i in t1_dexa_device_types()}
            rows = [r for r in rows if r["id"] in gold]
            pred = []
            for r in rows:
                result = r.get("episode_result") or next(
                    (e["episode_result"] for e in r.get("events", []) if "episode_result" in e), {})
                pred.append(sorted({AGENT_DEVICE[t.get("agent_id")] for t in (result.get("tasks") or {}).values()
                                    if t.get("agent_id") in AGENT_DEVICE}))
            s = scores([gold[r["id"]] for r in rows], pred, list(TASKS["t1_dexa_device_types"]["labels"]))
            main_metrics = (f"device types: P {fmt(s['precision_micro'])} R {fmt(s['recall_micro'])} "
                            f"F1 {fmt(s['f1_micro'])} (micro), F1 macro {fmt(s['f1_macro'])}, exact set {fmt(s['accuracy'])}")
            errors, latency = sum(r.get("error") is not None for r in rows), sum(r["latency_s"] for r in rows) / len(rows)
        else:
            adapter = import_module(f"external_benchmarks.adapters.{benchmark}")
            s = adapter.summarize(rows)
            if benchmark == "homebench":
                main_metrics = (f"exact match {fmt(s['exact_match'])}, P {fmt(s['precision'])} "
                                f"R {fmt(s['recall'])} F1 {fmt(s['f1'])}")
                by = defaultdict(list)
                for r in rows:
                    kind = r.get("category") or "?"
                    by["multi" if kind.startswith("multi") else kind].append(r)
                notes.append("HomeBench by type:\n" + "\n".join(
                    f"  {kind}: n={t['cases']} exact match {fmt(t['exact_match'])} P {fmt(t['precision'])} "
                    f"R {fmt(t['recall'])} F1 {fmt(t['f1'])}" for kind, t in
                    ((k, adapter.summarize(v)) for k, v in sorted(by.items()))))
            else:
                main_metrics = f"success rate {fmt(s.get('success_rate'))} ({s.get('evaluated_cases')} evaluated)"
            if benchmark == "simuhome":
                failures = Counter("success" if (r.get("evaluation") or {}).get("score") == 1 else
                                   "judge error" if (r.get("evaluation") or {}).get("score") == -1 else
                                   (r.get("evaluation") or {}).get("error_type") or "failed (no error type)" for r in rows)
                notes.append("SimuHome outcomes: " + ", ".join(f"{k}: {v}" for k, v in failures.most_common()))
            errors, latency = s.get("errors"), s.get("mean_latency_s")
        print(f"| {benchmark} | {done} | {main_metrics} | {errors} | {fmt(latency)} | {leaf} |")
    for note in notes:
        print("\n" + note)
    print("\nInterim only: partial runs follow dataset order and are not yet representative.")


if __name__ == "__main__":
    main()
