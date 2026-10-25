"""Compare systems on the extended DEXA benchmark (Akasha's automatic evaluator), run by run.

    python -m classification_eval.compare_dexa mas=logs/dgx/e2e/dexa_extended react=logs/dgx/e2e/react_dexa_extended

Each argument is <label>=<run folder written by run_experiments.py>. The per-case rows and the
summary are the runner's own (run_benchmark.summarize); this script only lines them up: overall,
by suite, by category/dimension, by expected behavior (execute vs. clarify) and by injected
failure mode, plus the paired difference on the shared cases with a 95% bootstrap CI and the
cases that one system passed and the other failed. Writes results/dgx/dexa_extended.{md,json}.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tasks import ROOT

META = {"config", "summary", "qed_leaf", "run", "dataset", "command", "runs", "failure", "comparison"}


def load_run(folder):
    rows = {}
    for file in sorted(Path(folder).rglob("*.json")):
        if file.stem in META:
            continue
        try:
            row = json.loads(file.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(row, dict) and "id" in row and "evaluation" in row:
            rows[row["id"]] = row
    return rows


def score(row):
    s = (row.get("evaluation") or {}).get("score")
    return s if s in (0, 1) else None


def rate(rows):
    scored = [score(r) for r in rows if score(r) is not None]
    return (sum(scored) / len(scored), len(scored)) if scored else (None, 0)


def grouped(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) if r.get(key) is not None else "-")].append(r)
    return {g: rate(v) for g, v in sorted(groups.items())}


def paired(a, b, n=1000, seed=42):
    """Bootstrap CI of success_rate(a) - success_rate(b) over the shared, evaluated cases."""
    ids = sorted(i for i in a if i in b and score(a[i]) is not None and score(b[i]) is not None)
    if not ids:
        return None
    xa = np.array([score(a[i]) for i in ids]); xb = np.array([score(b[i]) for i in ids])
    rng = np.random.default_rng(seed)
    deltas = [(xa[s] - xb[s]).mean() for s in (rng.integers(0, len(ids), len(ids)) for _ in range(n))]
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"n": len(ids), "delta": float((xa - xb).mean()), "ci": [float(low), float(high)],
            "a_only": [i for i in ids if score(a[i]) == 1 and score(b[i]) == 0],
            "b_only": [i for i in ids if score(b[i]) == 1 and score(a[i]) == 0],
            "verdict": "first better" if low > 0 else "second better" if high < 0 else "no significant difference"}


def lean_rejections(rows):
    """Cases whose MAS report shows a batch rejected by the Lean checker (AdmissionError)."""
    return [r["id"] for r in rows if "AdmissionError" in json.dumps((r.get("report") or {}).get("task_results", []))
            or "Lean allocation checker" in str((r.get("report") or {}).get("error") or "")]


def fmt(x):
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="<label>=<run folder>")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dgx")
    args = parser.parse_args(argv)
    runs = {}
    for spec in args.runs:
        label, folder = spec.split("=", 1)
        runs[label] = load_run(folder)
        if not runs[label]:
            sys.exit(f"no evaluated cases under {folder}")
    labels = list(runs)
    lines = ["# Extended DEXA benchmark: systems compared (automatic evaluator)", "",
             "| System | Cases | Evaluated | Success rate | Errors | Mean latency (s) | Lean rejections |",
             "|---|---|---|---|---|---|---|"]
    report = {"runs": {}, "by": {}, "paired": {}}
    for label in labels:
        rows = list(runs[label].values())
        r, n = rate(rows)
        errors = sum(bool((x.get("report") or {}).get("error")) for x in rows)
        latency = sum(x.get("latency_s", 0) for x in rows) / len(rows)
        rej = lean_rejections(rows)
        report["runs"][label] = {"cases": len(rows), "evaluated": n, "success_rate": r, "errors": errors,
                                 "mean_latency_s": latency, "lean_rejections": rej}
        lines.append(f"| {label} | {len(rows)} | {n} | **{fmt(r)}** | {errors} | {latency:.1f} | {len(rej)} |")
    for row in (r for rows in runs.values() for r in rows.values()):  # qed_leaf suite: flatten the safety tags
        safety = row.get("safety") or {}
        row["safety_condition"] = safety.get("condition")
        row["safety_rejects"] = {True: "environment also rejects", False: "checker only"}.get(safety.get("environment_rejects"))
    for key, title in (("suite", "suite"), ("category", "category"), ("dimension", "dimension"),
                       ("expected_behavior", "expected behavior"), ("failure_mode", "injected failure mode"),
                       ("safety_condition", "safety condition (qed_leaf suite)"), ("safety_rejects", "who can refuse (qed_leaf suite)")):
        table = {label: grouped(list(runs[label].values()), key) for label in labels}
        if key.startswith("safety") and all(g == "-" for t in table.values() for g in t):
            continue
        groups = sorted({g for t in table.values() for g in t})
        report["by"][key] = {g: {label: table[label].get(g, (None, 0)) for label in labels} for g in groups}
        lines += ["", f"## By {title}", "", "| " + title + " | n | " + " | ".join(labels) + " |",
                  "|---|---|" + "---|" * len(labels)]
        for g in groups:
            n = max(table[l].get(g, (None, 0))[1] for l in labels)
            lines.append(f"| {g} | {n} | " + " | ".join(fmt(table[l].get(g, (None, 0))[0]) for l in labels) + " |")
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        p = paired(runs[a], runs[b])
        report["paired"][f"{a} - {b}"] = p
        if p:
            lines += ["", f"## Paired comparison: {a} minus {b} (shared evaluated cases)", "",
                      f"- n = {p['n']}, difference in success rate = **{p['delta']:+.3f}**, 95% CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] -> {p['verdict'].replace('first', a).replace('second', b)}",
                      f"- passed only by {a} ({len(p['a_only'])}): {', '.join(p['a_only']) or '-'}",
                      f"- passed only by {b} ({len(p['b_only'])}): {', '.join(p['b_only']) or '-'}"]
    for label in labels:
        rej = report["runs"][label]["lean_rejections"]
        if rej:
            lines += ["", f"Lean-rejected batches in {label}: {', '.join(rej)}"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dexa_extended.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "dexa_extended.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
