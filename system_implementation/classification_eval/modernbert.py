"""ModernBERT on the classification tasks: zero-shot NLI, fine-tuning, or k-fold cross-validation.

    python -m classification_eval.modernbert --task t1_dexa_device_types    --mode zero-shot
    python -m classification_eval.modernbert --task t4_homeassistant_domain --mode zero-shot
    python -m classification_eval.modernbert --task t2_homebench_validity   --mode finetune
    python -m classification_eval.modernbert --task t3_simuhome_feasibility --mode cv

Zero-shot uses an NLI model (no training data exists for DEXA and Home Assistant). Fine-tuning
trains on the benchmark's train split and selects the epoch with the best macro-F1 on its valid
split. Cross-validation trains k models and pools their held-out predictions. On a V100: fp16,
SDPA attention (no bf16, no FlashAttention-2). Seed 42 everywhere.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tasks import ROOT, TASKS, read_jsonl

ZERO_SHOT_MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
BASE_MODEL = "answerdotai/ModernBERT-large"
HYPOTHESIS = {"t1_dexa_device_types": "This smart-home request involves the {}.",
              "t4_homeassistant_domain": "This smart-home request is about {}."}


def seed_everything(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_info():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "cpu"


def zero_shot(task, items, model_name, threshold=0.5):
    import torch
    from transformers import pipeline
    device, _ = device_info()
    classifier = pipeline("zero-shot-classification", model=model_name, device=0 if device == "cuda" else -1,
                          dtype=torch.float16 if device == "cuda" else None)
    spec = TASKS[task]
    by_description = {description: label for label, description in spec["labels"].items()}
    rows = []
    for entry in items:
        started = time.perf_counter()
        result = classifier(entry["text"], candidate_labels=list(by_description),
                            hypothesis_template=HYPOTHESIS.get(task, "{}."), multi_label=spec["multi_label"])
        scores = {by_description[d]: float(s) for d, s in zip(result["labels"], result["scores"])}
        pred = ([label for label, score in scores.items() if score >= threshold] if spec["multi_label"]
                else [max(scores, key=scores.get)])
        rows.append({"id": entry["id"], "pred": sorted(pred), "problem": None, "scores": scores,
                     "latency_s": time.perf_counter() - started})
    return rows


class Classifier:
    """Fine-tunable sequence classifier for single-label tasks; text and context as a sentence pair."""

    def __init__(self, model_name, labels, max_length, fp16=True):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.labels = torch, list(labels)
        self.device, _ = device_info()
        self.fp16 = fp16 and self.device == "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = dict(num_labels=len(self.labels), id2label=dict(enumerate(self.labels)),
                      label2id={label: i for i, label in enumerate(self.labels)})
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, attn_implementation="sdpa", **kwargs)
        except (ValueError, TypeError):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.max_length = max_length

    def encode(self, items):
        texts, contexts = [i["text"] for i in items], [i["context"] for i in items]
        if any(contexts):
            encoded = self.tokenizer(texts, contexts, truncation="only_second", max_length=self.max_length)
        else:
            encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length)
        return [{k: encoded[k][n] for k in ("input_ids", "attention_mask")} for n in range(len(items))]

    def batches(self, features, labels, batch_size, shuffle, rng=None):
        """Length-bucketed batches: shuffled chunks of 50 batches, sorted by length within a chunk."""
        order = list(range(len(features)))
        if shuffle:
            rng.shuffle(order)
        chunk = batch_size * 50
        for start in range(0, len(order), chunk):
            part = sorted(order[start:start + chunk], key=lambda n: len(features[n]["input_ids"]))
            groups = [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
            if shuffle:
                rng.shuffle(groups)
            for group in groups:
                batch = self.tokenizer.pad([features[n] for n in group], return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if labels is not None:
                    batch["labels"] = self.torch.tensor([labels[n] for n in group], device=self.device)
                yield group, batch

    def train(self, items, epochs, lr, batch_size, accumulation=1, valid=None, seed=42):
        from transformers import get_linear_schedule_with_warmup
        torch, rng = self.torch, random.Random(seed)
        features = self.encode(items)
        labels = [self.labels.index(i["labels"][0]) for i in items]
        steps = epochs * -(-len(features) // (batch_size * accumulation))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)
        best, history = (-1.0, None), []
        for epoch in range(epochs):
            self.model.train()
            for n, (_, batch) in enumerate(self.batches(features, labels, batch_size, True, rng), 1):
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                    loss = self.model(**batch).loss / accumulation
                scaler.scale(loss).backward()
                if n % accumulation == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    schedule.step()
            if valid:
                from .metrics import scores
                rows = self.predict(valid)
                f1 = scores([v["labels"] for v in valid], [r["pred"] for r in rows], self.labels)["f1_macro"]
                history.append({"epoch": epoch + 1, "valid_f1_macro": f1})
                print(f"epoch {epoch + 1}: valid macro-F1 {f1:.4f}", flush=True)
                if f1 > best[0]:
                    best = (f1, {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()})
        if best[1] is not None:
            self.model.load_state_dict(best[1])
        return history

    def predict(self, items, batch_size=64):
        torch = self.torch
        self.model.eval()
        features, rows = self.encode(items), {}
        with torch.no_grad():
            for group, batch in self.batches(features, None, batch_size, False):
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                    probabilities = torch.softmax(self.model(**batch).logits.float(), -1).cpu().numpy()
                per_item = (time.perf_counter() - started) / len(group)
                for n, p in zip(group, probabilities):
                    rows[n] = {"id": items[n]["id"], "pred": [self.labels[int(p.argmax())]], "problem": None,
                               "scores": dict(zip(self.labels, map(float, p))), "latency_s": per_item}
        return [rows[n] for n in range(len(items))]


def length_percentile(model_name, items, percentile=99, cap=8192, sample=2000):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    picked = random.Random(42).sample(items, min(sample, len(items)))
    lengths = [len(tokenizer(i["text"], i["context"] or None)["input_ids"]) for i in picked]
    return min(cap, int(-(-np.percentile(lengths, percentile) // 64) * 64))


def stratified_subsample(items, n, seed=42):
    if not n or n >= len(items):
        return items
    by = defaultdict(list)
    for entry in items:
        by[entry["labels"][0]].append(entry)
    rng, picked = random.Random(seed), []
    for label in sorted(by):
        picked += rng.sample(by[label], round(n * len(by[label]) / len(items)))
    return picked


def folds(items, k=5, seed=42):
    """Stratified folds over (group, label), deterministic."""
    by = defaultdict(list)
    for entry in items:
        by[(entry["group"], entry["labels"][0])].append(entry)
    rng, assignment = random.Random(seed), {}
    for key in sorted(by):
        members = sorted(by[key], key=lambda e: e["id"])
        rng.shuffle(members)
        for n, entry in enumerate(members):
            assignment[entry["id"]] = n % k
    return [[e for e in items if assignment[e["id"]] == f] for f in range(k)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--mode", required=True, choices=("zero-shot", "finetune", "cv"))
    parser.add_argument("--tasks-dir", type=Path, default=ROOT / "results" / "dgx" / "tasks")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dgx" / "predictions")
    parser.add_argument("--model", help=f"default: {ZERO_SHOT_MODEL} (zero-shot) or {BASE_MODEL}")
    parser.add_argument("--tag", help="File tag; default modernbert-<mode>")
    parser.add_argument("--threshold", type=float, default=0.5, help="Zero-shot multi-label threshold")
    parser.add_argument("--epochs", type=int, default=None, help="default: 2 (finetune), 5 (cv)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, help="default: 99th percentile of token lengths (cap 8192)")
    parser.add_argument("--max-train", type=int, default=0, help="Stratified training subsample (0 = all)")
    parser.add_argument("--max-valid", type=int, default=3000, help="Stratified valid subsample for selection")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-test", type=int, default=0, help="Limit test items (smoke tests only)")
    args = parser.parse_args(argv)
    seed_everything()
    spec = TASKS[args.task]
    items = read_jsonl(args.tasks_dir / f"{args.task}.jsonl")
    test = [i for i in items if i["split"] == "test"]
    test = test[:args.max_test] if args.max_test else test
    model_name = args.model or (ZERO_SHOT_MODEL if args.mode == "zero-shot" else BASE_MODEL)
    device, gpu = device_info()
    meta = {"model": model_name, "mode": args.mode, "device": gpu, "seed": 42, "n": len(test)}
    started = time.perf_counter()
    if args.mode == "zero-shot":
        rows = zero_shot(args.task, test, model_name, args.threshold)
        meta.update(threshold=args.threshold if spec["multi_label"] else None,
                    hypothesis=HYPOTHESIS.get(args.task, "{}."))
    else:
        if spec["multi_label"]:
            raise SystemExit("Fine-tuning is implemented for single-label tasks only.")
        epochs = args.epochs or (2 if args.mode == "finetune" else 5)
        labels = list(spec["labels"])
        if args.mode == "finetune":
            train = stratified_subsample([i for i in items if i["split"] == "train"], args.max_train)
            valid = stratified_subsample([i for i in items if i["split"] == "valid"], args.max_valid)
            max_length = args.max_length or length_percentile(model_name, train)
            print(f"train {len(train)}, valid {len(valid)}, test {len(test)}, max_length {max_length}", flush=True)
            classifier = Classifier(model_name, labels, max_length)
            history = classifier.train(train, epochs, args.lr, args.batch_size, args.accumulation, valid=valid)
            rows = classifier.predict(test)
            meta.update(train_n=len(train), valid_n=len(valid), history=history)
        else:
            max_length = args.max_length or length_percentile(model_name, test)
            rows, history = [], []
            for f, held_out in enumerate(folds(test, args.folds)):
                train = [i for i in test if i not in held_out]
                print(f"fold {f + 1}/{args.folds}: train {len(train)}, held out {len(held_out)}", flush=True)
                seed_everything()
                classifier = Classifier(model_name, labels, max_length)
                classifier.train(train, epochs, args.lr, args.batch_size, args.accumulation)
                rows += classifier.predict(held_out)
                del classifier
            meta.update(folds=args.folds)
        meta.update(epochs=epochs, lr=args.lr, batch_size=args.batch_size, accumulation=args.accumulation,
                    max_length=max_length)
    meta["wall_s"] = time.perf_counter() - started
    tag = args.tag or f"modernbert-{args.mode}"
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"{args.task}.{tag}.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    (args.out / f"{args.task}.{tag}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{args.task} {tag}: {len(rows)} predictions in {meta['wall_s']:.0f}s on {gpu}")


if __name__ == "__main__":
    main()
