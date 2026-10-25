"""Zero-shot classification with an Ollama chat model (Gemma 4) on the same items as ModernBERT.

    OLLAMA_BASE_URL=http://127.0.0.1:11503 python -m classification_eval.gemma \
        --task t1_dexa_device_types --model gemma4:31b --out results/dgx/predictions

Thinking off, temperature 0, seed 42. Predictions are appended per item, so a rerun resumes.
Unparseable answers and labels outside the label set become an empty prediction and are counted.
"""
import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .tasks import ROOT, TASKS, read_jsonl


def messages(task, entry):
    spec = TASKS[task]
    choose = "one or more labels" if spec["multi_label"] else "exactly one label"
    labels = "\n".join(f"- {name}: {description}" for name, description in spec["labels"].items())
    system = (f"You classify smart-home requests. {spec['question']}\nChoose {choose} from:\n{labels}\n"
              'Answer only with JSON: {"labels": ["<label>", ...]}')
    user = (f"Home:\n{entry['context']}\n\n" if entry["context"] else "") + f"Request: {entry['text']}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse(task, content):
    """Known labels from the model's JSON; (labels, problem) where problem is None, 'unparseable' or 'unknown'."""
    spec = TASKS[task]
    try:
        raw = json.loads(content).get("labels", [])
    except (ValueError, AttributeError):
        return [], "unparseable"
    raw = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    names = list(dict.fromkeys(str(label).strip().lower() for label in raw))
    known = [name for name in names if name in spec["labels"]]
    problem = "unknown" if len(known) < len(names) else None
    return (known if spec["multi_label"] else known[:1]), problem


def classify(task, items, model, base_url, out, *, think=False, num_ctx=32768, timeout=600):
    out = Path(out)
    done = {row["id"] for row in read_jsonl(out)} if out.exists() else set()
    with out.open("a", encoding="utf-8") as file:
        for entry in items:
            if entry["id"] in done:
                continue
            body = {"model": model, "messages": messages(task, entry), "format": "json", "stream": False,
                    "think": think, "keep_alive": "30m",
                    "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx, "num_predict": 256}}
            started = time.perf_counter()
            row = {"id": entry["id"], "pred": [], "problem": None, "raw": None}
            try:
                request = Request(base_url.rstrip("/") + "/api/chat", json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})
                with urlopen(request, timeout=timeout) as response:
                    data = json.load(response)
                row["raw"] = data["message"]["content"]
                row["pred"], row["problem"] = parse(task, row["raw"])
                row["tokens"] = [data.get("prompt_eval_count"), data.get("eval_count")]
            except Exception as exc:  # recorded per item; the run continues
                row["problem"] = f"error: {type(exc).__name__}: {exc}"
            row["latency_s"] = time.perf_counter() - started
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--tasks-dir", type=Path, default=ROOT / "results" / "dgx" / "tasks")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ids-file", type=Path, help="Only these item IDs (one per line), e.g. the T2 sample")
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--tag", default=None, help="File tag; default derived from the model name")
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dgx" / "predictions")
    args = parser.parse_args(argv)
    items = [i for i in read_jsonl(args.tasks_dir / f"{args.task}.jsonl") if i["split"] == args.split]
    if args.ids_file:
        wanted = set(args.ids_file.read_text(encoding="utf-8").split())
        items = [i for i in items if i["id"] in wanted]
    tag = args.tag or args.model.replace(":", "-")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.task}.{tag}.meta.json").write_text(json.dumps(
        {"model": args.model, "mode": "zero-shot", "think": False, "base_url": args.base_url,
         "n": len(items), "split": args.split, "ids_file": str(args.ids_file) if args.ids_file else None}, indent=2))
    classify(args.task, items, args.model, args.base_url, args.out / f"{args.task}.{tag}.jsonl")
    rows = read_jsonl(args.out / f"{args.task}.{tag}.jsonl")
    print(f"{args.task} {tag}: {len(rows)} predictions, problems: "
          f"{sum(r['problem'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
