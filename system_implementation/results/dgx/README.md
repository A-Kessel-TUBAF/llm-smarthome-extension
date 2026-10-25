# DGX-2 results: Gemma 4 and ModernBERT on four smart-home benchmarks

Runs of 23–24 October 2026 on the DGX-2 (Tesla V100-SXM3-32GB), branch `QED_LEAF`.

- **Gemma 4:** `gemma4:31b` (Ollama, Q4_K_M), thinking off, temperature 0, seed 42.
- **End-to-end system:** the full MAS with the Lean checker admitting every command batch
  (`checker = "lean"`, `experiment_config.dgx.toml`). The QED_LEAF audit of the checker
  **passed** (24/24 checks, both negative controls caught) and is recorded with every run.
- **ModernBERT:** `answerdotai/ModernBERT-large` fine-tuned or cross-validated, and
  `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` for zero-shot NLI. ModernBERT is an encoder and
  cannot generate plans or tool calls, so it is compared with Gemma 4 only on one classification
  task per benchmark (`classification_eval/README.md`).

## 1. End to end: Gemma 4 31B + MAS + Lean admission

| Benchmark | Cases | Main result | Errors | LLM calls | Tokens in / out | Mean latency |
|---|---|---|---|---|---|---|
| DEXA (original edition, 60 cases) | 60 | device types P **0.938**, R 0.487, F1 0.641 (micro); required-intent recall 0.549 | 10 | 99 | 0.32 M / 0.01 M | 39 s |
| HomeBench (stratified sample) | 1,000 | exact match **0.649**, P 0.668, R 0.742, F1 **0.703** | 49 | 1,262 | 5.98 M / 0.18 M | 37 s |
| SimuHome | 600 | success rate **0.417** | 227 | 3,167 | 19.0 M / 0.34 M | 332 s |
| Home Assistant (Assist) | 95 | success rate **0.842** | 9 | 311 | 1.77 M / 0.01 M | 107 s |

DEXA has no automatic task score. Its row gives the device types of the agents the MAS assigned a
task, against the device types of all required and acceptable intents. Required-intent recall
counts only the required intents.

**HomeBench by instruction type**

| Type | n | Exact match | P | R | F1 |
|---|---|---|---|---|---|
| normal | 356 | 0.848 | 0.791 | 0.916 | 0.849 |
| missing device | 242 | 0.905 | 0.818 | 0.909 | 0.861 |
| missing attribute | 148 | 0.696 | 0.656 | 0.696 | 0.675 |
| multi-step | 254 | 0.098 | 0.615 | 0.679 | 0.645 |

**SimuHome by category** (50 episodes each; success = native evaluation incl. 3-vote judge)

| Query type | Feasible | Infeasible |
|---|---|---|
| QT1 (state questions) | 0.82 | 0.80 |
| QT2 | 0.42 | 0.34 |
| QT3 | 0.42 | 0.80 |
| QT4-1 | 0.26 | 0.26 |
| QT4-2 | 0.44 | **0.02** |
| QT4-3 | 0.40 | **0.02** |

In QT4-2 and QT4-3 infeasible, 49 of 50 answers each were judged unfaithful: the system
does not recognize these infeasible workflow requests.

**Home Assistant by domain:**

| Domain | n | Success |
|---|---|---|
| cover, fan, todo, valve | 11, 6, 9, 9 | 1.00 each |
| media player | 21 | 0.86 |
| vacuum | 17 | 0.71 |
| light | 13 | 0.69 |
| lock | 9 | 0.67 |

## 2. Classification: Gemma 4 vs ModernBERT

| Task | Model | Mode | n | Acc. | P / R / F1 micro | P / R / F1 macro | F1 macro 95% CI | s / item |
|---|---|---|---|---|---|---|---|---|
| T1 DEXA device types | Gemma 4 + MAS | end to end | 60 | 0.267 | 0.938 / 0.487 / 0.641 | 0.948 / 0.497 / 0.638 | 0.552–0.706 | 39.1 |
| T1 | Gemma 4 | zero-shot | 60 | 0.550 | 0.940 / 0.812 / 0.871 | 0.940 / 0.834 / **0.872** | 0.825–0.906 | 1.79 |
| T1 | ModernBERT-large NLI | zero-shot | 60 | 0.267 | 0.892 / 0.481 / 0.624 | 0.909 / 0.510 / 0.611 | 0.524–0.681 | 0.19 |
| T2 HomeBench validity | Gemma 4 | zero-shot | 1,500 | 0.873 | 0.873 / 0.873 / 0.873 | 0.890 / 0.873 / 0.872 | 0.856–0.888 | 8.66 |
| T2 | ModernBERT-large | fine-tuned (same 1,500) | 1,500 | 0.972 | 0.972 / 0.972 / 0.972 | 0.974 / 0.972 / **0.972** | – | 0.026 |
| T2 | ModernBERT-large | fine-tuned (full test) | 12,952 | 0.946 | 0.946 / 0.946 / 0.946 | 0.929 / 0.961 / 0.940 | 0.936–0.944 | 0.026 |
| T3 SimuHome feasibility | Gemma 4 | zero-shot | 600 | 0.592 | 0.592 / 0.592 / 0.592 | 0.607 / 0.592 / 0.577 | 0.537–0.616 | 4.26 |
| T3 | ModernBERT-large | 5-fold CV | 600 | 0.788 | 0.788 / 0.788 / 0.788 | 0.794 / 0.788 / **0.787** | 0.755–0.820 | 0.010 |
| T4 Home Assistant domain | Gemma 4 | zero-shot | 95 | 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / **1.000** | 1.000–1.000 | 1.40 |
| T4 | ModernBERT-large NLI | zero-shot | 95 | 0.937 | 0.937 / 0.937 / 0.937 | 0.972 / 0.956 / 0.958 | 0.922–0.986 | 0.15 |

The T1 "Gemma 4 + MAS" row is the end-to-end system: 4 of its 60 cases assigned no task at all.

**Paired comparison** (paired bootstrap of the macro-F1 difference on shared items; a model counts
as better only if the 95% CI excludes 0):

| Task | ModernBERT | vs | n | ΔF1 macro (ModernBERT − other) | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| T1 | zero-shot | Gemma 4 zero-shot | 60 | −0.261 | −0.345 to −0.196 | **Gemma 4 better** |
| T1 | zero-shot | Gemma 4 + MAS | 60 | −0.027 | −0.112 to +0.054 | no significant difference |
| T2 | fine-tuned | Gemma 4 zero-shot | 1,500 | +0.100 | +0.082 to +0.119 | **ModernBERT better** |
| T3 | 5-fold CV | Gemma 4 zero-shot | 600 | +0.211 | +0.154 to +0.268 | **ModernBERT better** |
| T4 | zero-shot | Gemma 4 zero-shot | 95 | −0.042 | −0.078 to −0.014 | **Gemma 4 better** |

**Per-class F1:**

| Task | Model | Per-class F1 |
|---|---|---|
| T1 | Gemma 4 | search 0.77, fridge 0.92, display 0.90, audio 0.85, clock 0.93, thermostat 0.73, light 0.90, calendar 0.97 |
| T1 | ModernBERT | search 0.07, fridge 0.76, display 0.60, audio 0.59, clock 0.74, thermostat 0.58, light 0.65, calendar 0.89 |
| T1 | Gemma 4 + MAS | search 0.54, fridge 0.91, display 0.60, audio 0.48, clock 0.64, thermostat 0.67, light 0.56, calendar 0.71 |
| T2 (same 1,500) | ModernBERT | normal 0.956, missing device 1.000, missing attribute 0.960 |
| T2 (same 1,500) | Gemma 4 | normal 0.852, missing device 0.969, missing attribute 0.794 |
| T3 | ModernBERT | feasible 0.80, infeasible 0.77 |
| T3 | Gemma 4 | feasible 0.66, infeasible 0.50 |
| T4 | ModernBERT | 1.00 except media player 0.88, vacuum 0.79 |
| T4 | Gemma 4 | 1.00 for all |

## 3. Does ModernBERT beat Gemma 4?

- **End to end: no.** ModernBERT cannot plan or call tools; only Gemma 4 can run the benchmarks.
- **Classification with in-domain training labels: yes.** Fine-tuned on HomeBench, ModernBERT
  beats zero-shot Gemma 4 by +0.10 macro-F1 (T2). Cross-validated on SimuHome, it wins by +0.21 (T3).
- **Zero-shot against zero-shot: no.** Gemma 4 wins on DEXA (+0.26) and Home Assistant (+0.04).
- **Speed and cost:** ModernBERT is about 9× faster zero-shot and about 330–430× faster when
  fine-tuned (0.01–0.19 s against 1.4–8.7 s per item).
- **The bottleneck is MAS planning, not the model's understanding.** Gemma 4 alone names DEXA's
  device types well (macro-F1 0.872, required-intent recall 0.92), but the end-to-end MAS assigns
  tasks to far fewer devices (F1 0.638, required-intent recall 0.55). This matches the "incomplete
  intent decomposition" failure pattern of the DEXA paper.

## 4. Caveats

- **One run per setting:** temperature 0, seed 42; Gemma 4 quantized to Q4_K_M; thinking off.
- **SimuHome judge.** Gemma 4 31B judges Gemma 4's own answers (3 votes, 49,152-token context;
  1,008 calls, 0 errors, 0 truncated prompts). The project default judge (gemma2, 8,192 tokens)
  cannot read the ~28,000-token judge prompts. A first run with it scored every answer B; it was
  discarded.
- **Home Assistant** is the run after commit `b1a8ece`. Before that fix, 81 of 95 cases failed on
  the routing argument `service_id` (success 0.168). That earlier run is kept aside, not used here.
- **DEXA edition.** T1 and the DEXA end-to-end row use the benchmark's unchanged original edition
  (`benchmark_data.json`, 60 cases). The extended edition (`extension_tasks.json`, 127 cases with its own
  automatic evaluator, upstream `3700e9d`) was not yet run; 40 earlier extension cases that were run are
  not part of either edition's gold set and are excluded here.
- **DEXA device-type gold** counts every required and acceptable intent ("a or b" as both). A
  system that legitimately picks one interpretation of an ambiguous request loses recall, hence the
  extra required-intent recall.
- **T2 sizes differ.** Gemma 4 answered a stratified 1,500-item sample (500 per class). The
  verdict uses those shared items; ModernBERT's full-test row is for reference.
- **T3 data is small.** Cross-validation trains on 480 episodes per fold.

## Files

| File | Content |
|---|---|
| `classification.{md,csv,json}` | Section 2, with per-class values |
| `end_to_end.txt` | Section 1 summary |
| `e2e/<benchmark>.jsonl` | One compact record per case: id, category, score/evaluation, error, agents, latency, model usage |
| `e2e/*_summary_*.json` | Native benchmark summaries (SimuHome per category) |
| `predictions/` | All classification predictions, with model metadata |
| `tasks/` | Classification items (T2 is rebuilt by `classification_eval.tasks`; 430 MB, not in git) |
| `homebench_sample_ids.txt`, `t2_sample_ids.txt` | The stratified samples (seed 42) |
