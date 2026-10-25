"""Precision, recall and F1 per task and model, bootstrap CIs, and the paired verdict.

    python -m classification_eval.metrics --dexa-mas-logs logs/dgx/e2e/dexa

Reads results/dgx/tasks/<task>.jsonl (gold) and results/dgx/predictions/<task>.<tag>.jsonl, and
writes results/dgx/classification.{md,csv,json}. Multi- and single-label tasks are scored alike
on label indicator matrices; macro averages run over the labels present in gold or predictions
(scikit-learn's default). Accuracy is exact match of the label set. CIs: 1,000 bootstrap
resamples (seed 42); a paired comparison resamples the shared items of both models, and a
model counts as better only if the 95% CI of the macro-F1 difference excludes 0.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .tasks import ROOT, TASKS, read_jsonl

AGENT_DEVICE = {"search_engine": "search", "fridge": "fridge", "tv_display": "display", "audio_system": "audio",
                "clock": "clock", "thermostat": "thermostat", "lighting": "light", "calendar": "calendar"}


def binarize(label_lists, labels):
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(label_lists), len(labels)), dtype=bool)
    for row, names in enumerate(label_lists):
        for name in names:
            if name in index:
                matrix[row, index[name]] = True
    return matrix


def _ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def _counts(Y, P):
    return (Y & P).sum(0), (~Y & P).sum(0), (Y & ~P).sum(0)


def _prf(tp, fp, fn):
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return precision, recall, _ratio(2 * precision * recall, precision + recall)


def f1_macro(Y, P):
    present = Y.any(0) | P.any(0)
    return float(_prf(*_counts(Y, P))[2][present].mean()) if present.any() else 0.0


def scores(gold, pred, labels):
    Y, P = binarize(gold, labels), binarize(pred, labels)
    tp, fp, fn = _counts(Y, P)
    precision, recall, f1 = _prf(tp, fp, fn)
    present = Y.any(0) | P.any(0)
    micro = _prf(np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()]))
    return {"n": len(gold), "accuracy": float((Y == P).all(1).mean()) if len(gold) else 0.0,
            "precision_micro": float(micro[0][0]), "recall_micro": float(micro[1][0]), "f1_micro": float(micro[2][0]),
            "precision_macro": float(precision[present].mean()), "recall_macro": float(recall[present].mean()),
            "f1_macro": float(f1[present].mean()),
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                                  "support": int(Y[:, i].sum())} for i, label in enumerate(labels)}}


def bootstrap(Y, Ps, n=1000, seed=42):
    """Macro-F1 for each prediction matrix over the same resamples of the rows."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(Ps)))
    for b in range(n):
        rows = rng.integers(0, len(Y), len(Y))
        out[b] = [f1_macro(Y[rows], P[rows]) for P in Ps]
    return out


def dexa_mas_predictions(log_dir, out_dir, model="gemma4:31b"):
    """T1 predictions of the end-to-end MAS: device types of the agents that were assigned a task."""
    rows = []
    for file in sorted(Path(log_dir).rglob("*.json")):
        try:
            row = json.loads(file.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(row, dict) or "id" not in row or ("episode_result" not in row and "events" not in row):
            continue
        result = row.get("episode_result") or next(
            (e["episode_result"] for e in row.get("events", []) if "episode_result" in e), {})
        agents = {t.get("agent_id") for t in (result.get("tasks") or {}).values()}
        rows.append({"id": row["id"], "pred": sorted({AGENT_DEVICE[a] for a in agents if a in AGENT_DEVICE}),
                     "problem": None if agents else "no tasks", "latency_s": row.get("latency_s")})
    tag, task = f"{model.replace(':', '-')}-mas", "t1_dexa_device_types"
    with (Path(out_dir) / f"{task}.{tag}.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    (Path(out_dir) / f"{task}.{tag}.meta.json").write_text(json.dumps(
        {"model": model, "mode": "MAS end to end (agents assigned a task)", "n": len(rows), "logs": str(log_dir)}, indent=2))
    return len(rows)


def evaluate(tasks_dir, pred_dir):
    results, comparisons = [], []
    for task, spec in TASKS.items():
        gold_file = Path(tasks_dir) / f"{task}.jsonl"
        if not gold_file.exists():
            continue
        labels = list(spec["labels"])
        gold = {i["id"]: i["labels"] for i in read_jsonl(gold_file) if i["split"] == "test"}
        models = {}
        for file in sorted(Path(pred_dir).glob(f"{task}.*.jsonl")):
            tag = file.name[len(task) + 1:-len(".jsonl")]
            rows = [r for r in read_jsonl(file) if r["id"] in gold]
            meta_file = file.with_name(f"{task}.{tag}.meta.json")
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            models[tag] = ({r["id"]: r for r in rows}, meta)
        for tag, (rows, meta) in models.items():
            ids = sorted(rows)
            Y, P = binarize([gold[i] for i in ids], labels), binarize([rows[i]["pred"] for i in ids], labels)
            result = scores([gold[i] for i in ids], [rows[i]["pred"] for i in ids], labels)
            ci = np.percentile(bootstrap(Y, [P])[:, 0], [2.5, 97.5]) if ids else [0.0, 0.0]
            latencies = [rows[i]["latency_s"] for i in ids if rows[i].get("latency_s") is not None]
            results.append({"task": task, "tag": tag, "model": meta.get("model", tag), "mode": meta.get("mode", ""),
                            **result, "f1_macro_ci": [float(ci[0]), float(ci[1])],
                            "problems": sum(rows[i].get("problem") is not None for i in ids),
                            "mean_latency_s": float(np.mean(latencies)) if latencies else None})
        encoders = [t for t in models if t.startswith("modernbert")]
        llms = [t for t in models if not t.startswith("modernbert")]
        for a in encoders:
            for b in llms:
                ids = sorted(set(models[a][0]) & set(models[b][0]))
                if not ids:
                    continue
                Y = binarize([gold[i] for i in ids], labels)
                Pa = binarize([models[a][0][i]["pred"] for i in ids], labels)
                Pb = binarize([models[b][0][i]["pred"] for i in ids], labels)
                samples = bootstrap(Y, [Pa, Pb])
                delta = samples[:, 0] - samples[:, 1]
                low, high = np.percentile(delta, [2.5, 97.5])
                comparisons.append({"task": task, "modernbert": a, "other": b, "n": len(ids),
                                    "f1_macro_modernbert": f1_macro(Y, Pa), "f1_macro_other": f1_macro(Y, Pb),
                                    "delta_f1_macro": f1_macro(Y, Pa) - f1_macro(Y, Pb), "delta_ci": [float(low), float(high)],
                                    "verdict": "too few shared items" if len(ids) < 20 else "ModernBERT better"
                                    if low > 0 else f"{b} better" if high < 0 else "no significant difference"})
    return results, comparisons


def write_report(results, comparisons, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classification.json").write_text(json.dumps({"results": results, "comparisons": comparisons}, indent=2))
    columns = ["task", "tag", "model", "mode", "n", "accuracy", "precision_micro", "recall_micro", "f1_micro",
               "precision_macro", "recall_macro", "f1_macro", "f1_macro_ci", "problems", "mean_latency_s"]
    with (out / "classification.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    f = lambda v: "–" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["# Classification: Gemma 4 vs ModernBERT", "",
             "| Task | Model | Mode | n | Acc. | P micro | R micro | F1 micro | P macro | R macro | F1 macro | F1 macro 95% CI | Problems | Latency/item (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['task']} | {r['model']} | {r['mode']} | {r['n']} | {f(r['accuracy'])} | {f(r['precision_micro'])} | "
                     f"{f(r['recall_micro'])} | {f(r['f1_micro'])} | {f(r['precision_macro'])} | {f(r['recall_macro'])} | "
                     f"{f(r['f1_macro'])} | {f(r['f1_macro_ci'][0])}–{f(r['f1_macro_ci'][1])} | {r['problems']} | {f(r['mean_latency_s'])} |")
    lines += ["", "## Paired comparison (shared items)", "",
              "| Task | ModernBERT | Other | n | F1 macro ModernBERT | F1 macro other | ΔF1 | 95% CI | Verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in comparisons:
        lines.append(f"| {c['task']} | {c['modernbert']} | {c['other']} | {c['n']} | {f(c['f1_macro_modernbert'])} | "
                     f"{f(c['f1_macro_other'])} | {f(c['delta_f1_macro'])} | {f(c['delta_ci'][0])}–{f(c['delta_ci'][1])} | {c['verdict']} |")
    for r in results:
        lines += ["", f"### {r['task']} – {r['model']} ({r['mode']})", "", "| Label | Precision | Recall | F1 | Support |",
                  "|---|---|---|---|---|"]
        lines += [f"| {label} | {f(v['precision'])} | {f(v['recall'])} | {f(v['f1'])} | {v['support']} |"
                  for label, v in r["per_class"].items()]
    (out / "classification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", type=Path, default=ROOT / "results" / "dgx" / "tasks")
    parser.add_argument("--pred-dir", type=Path, default=ROOT / "results" / "dgx" / "predictions")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dgx")
    parser.add_argument("--dexa-mas-logs", type=Path, help="End-to-end DEXA run folder (adds the Gemma 4 + MAS row of T1)")
    parser.add_argument("--mas-model", default="gemma4:31b")
    args = parser.parse_args(argv)
    if args.dexa_mas_logs:
        print("DEXA MAS predictions:", dexa_mas_predictions(args.dexa_mas_logs, args.pred_dir, args.mas_model))
    results, comparisons = evaluate(args.tasks_dir, args.pred_dir)
    write_report(results, comparisons, args.out)
    for c in comparisons:
        print(f"{c['task']}: {c['modernbert']} vs {c['other']} (n={c['n']}): dF1 {c['delta_f1_macro']:+.3f} "
              f"[{c['delta_ci'][0]:+.3f}, {c['delta_ci'][1]:+.3f}] -> {c['verdict']}")
    print(f"Report: {args.out / 'classification.md'}")


if __name__ == "__main__":
    main()
