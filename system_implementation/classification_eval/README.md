# Gemma 4 vs ModernBERT: one classification task per benchmark

ModernBERT is an encoder: it classifies text but cannot generate plans, JSON or tool calls, so it
cannot run the benchmarks end to end. To compare it with Gemma 4 anyway, each benchmark gets a
classification task with gold labels taken from the benchmark's own annotations. Both models
answer the same items; precision, recall and F1 decide.

| Task | Items (test) | Labels | Gold | ModernBERT | Gemma 4 |
|---|---|---|---|---|---|
| `t1_dexa_device_types` | DEXA, 60 original + 40 extension | 8 device types, multi-label | device types of required ∪ acceptable intents ("a or b" counts as both) | zero-shot NLI (`MoritzLaurer/ModernBERT-large-zeroshot-v2.0`, threshold 0.5) | zero-shot; also **Gemma 4 + MAS**: device types of the agents the end-to-end run assigned a task |
| `t2_homebench_validity` | HomeBench single-instruction test cases (12,952) | normal / unexist_device / unexist_attribute | HomeBench `type` | fine-tuned `answerdotai/ModernBERT-large` on the train split (102,819), epoch chosen on valid | zero-shot on a 1,500-item sample (500 per class, seed 42) |
| `t3_simuhome_feasibility` | SimuHome episodes (600) | feasible / infeasible | `meta.case` | 5-fold stratified cross-validation (query type × label) | zero-shot |
| `t4_homeassistant_domain` | Home Assistant Assist cases (95) | 8 device categories, one per case | task category | zero-shot NLI | zero-shot |

T2 and T3 give both models the home's devices as context (HomeBench: each device's operations;
SimuHome: user location and each room's devices with their Matter clusters).

Gemma 4 runs through Ollama with thinking off, temperature 0 and seed 42, and answers with
`{"labels": [...]}`. Unparseable answers or labels outside the label set count as an empty
prediction and are reported in the `Problems` column.

## Metrics

Accuracy (exact label set), precision/recall/F1 micro and macro, and per-class values; a 95%
bootstrap CI for macro-F1 (1,000 resamples, seed 42). Each ModernBERT result is compared with
each other model on their shared items by a paired bootstrap of the macro-F1 difference; a model
is called better only if that CI excludes 0 (and at least 20 items are shared).

## Run (DGX-2)

```bash
cd ~/dgx_eval/repo/system_implementation
uv venv --python 3.13 .venv-clf
uv pip install --python .venv-clf/bin/python torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-clf/bin/python -r classification_eval/requirements.txt

.venv-bench/bin/python -m classification_eval.tasks                          # T1-T3 -> results/dgx/tasks
.venv-homeassistant/bin/python -m classification_eval.tasks --tasks t4

export CUDA_VISIBLE_DEVICES=3 OLLAMA_BASE_URL=http://127.0.0.1:11503
for t in t1_dexa_device_types t3_simuhome_feasibility t4_homeassistant_domain; do
  .venv-bench/bin/python -m classification_eval.gemma --task $t
done
.venv-bench/bin/python -m classification_eval.gemma --task t2_homebench_validity --ids-file results/dgx/t2_sample_ids.txt

.venv-clf/bin/python -m classification_eval.modernbert --task t1_dexa_device_types --mode zero-shot
.venv-clf/bin/python -m classification_eval.modernbert --task t4_homeassistant_domain --mode zero-shot
.venv-clf/bin/python -m classification_eval.modernbert --task t3_simuhome_feasibility --mode cv
.venv-clf/bin/python -m classification_eval.modernbert --task t2_homebench_validity --mode finetune

.venv-bench/bin/python -m classification_eval.metrics --dexa-mas-logs logs/dgx/e2e/dexa
```

The report is `results/dgx/classification.md` (also `.csv`, `.json`). `results/dgx/tasks/t2_homebench_validity.jsonl`
(about 430 MB) is git-ignored; it is rebuilt from HomeBench by `classification_eval.tasks`.
