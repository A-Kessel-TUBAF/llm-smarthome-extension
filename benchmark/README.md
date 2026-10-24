# DEXA extended benchmark

The revised edition contains **127 scenarios**: **104 core** and **23 robustness**.
It is the complete dataset in `extension_tasks.json`, with annotations in `complexity_annotations.json`, including the revised cases
inherited from the original edition. `benchmark_data.json` is the **unchanged
historical 60-case reference** and does not require the new episode interface.
`moderate_15`, `ext_fail_08`, `simple_13` (reading light) and `complex_08` (cozy movie night) are absent from the revised edition and its annotations. `ext_amb_09` replaces the latter subjective scene assumptions with a unique audible-player scenario.

## Run

From `system_implementation/`, install the existing requirements and configure the
model in `experiment_config.toml`. ReAct/DS-IA and the SAGE parent can use
`external_benchmarks/requirements_react.txt`; upstream SAGE still needs its existing
separate checkout/environment. The extended benchmark requires no judge model.

```bash
# Same cases, environment and evaluator for all four participants
python run_experiments.py --benchmark dexa --system mas --output-dir logs/dexa/mas
python run_experiments.py --benchmark dexa --system react --output-dir logs/dexa/react
python run_experiments.py --benchmark dexa --system dsia --output-dir logs/dexa/dsia
python run_experiments.py --benchmark dexa --system sage --output-dir logs/dexa/sage

# Select the unperturbed core or the robustness suite
python run_experiments.py --benchmark dexa --system mas --suite core --output-dir logs/dexa/core
python run_experiments.py --benchmark dexa --system mas --suite robustness --output-dir logs/dexa/robustness
python run_experiments.py --benchmark dexa --system react --failure-mode transient --output-dir logs/dexa/transient

# One case, repeated experiment, or resume a compatible existing run
python run_experiments.py --benchmark dexa --system mas --case-id robust_power_cut --output-dir logs/dexa/power
python run_experiments.py --benchmark dexa --system mas --repetitions 3 --suite robustness --output-dir logs/dexa/repeated
python run_experiments.py --benchmark dexa --system mas --suite robustness --output-dir logs/dexa/robustness --resume

# Historical reference: direct run_request, historical inputs, no new interface
python run_benchmark.py all --dataset original --system mas --output-dir logs/dexa/original
```

`--dataset extended` is the DEXA default. The older names `extension` and `all`
are compatibility aliases for this complete revised edition, not additional copies
of the historical cases. `--category`, `--case-id`, `--dimension`, `--suite` and
`--failure-mode` select cases before execution. `--failure-mode none` includes the
five unperturbed robustness controls; `--suite core` excludes those controls.
`python run_experiments.py` now defaults to all configured systems on extended
DEXA first, followed by SimuHome, Home Assistant and HomeBench. All repetitions
of one benchmark finish before the next begins. `--benchmark dexa` selects only
this benchmark. `--benchmark all` applies DEXA suite/dimension/failure filters only
to its DEXA runs. The historical `--dataset original` is a single-benchmark option.

## Boundary and ownership

The existing `external_benchmarks/common.py` owns `Episode` and `run_episode`.
An episode supplies the request, public context, JSON tool schemas and `invoke`.
Each system wrapper translates this boundary into its native input/output format.
It must return `answer` (a string) and an execution status, and may return
`llm_calls` or other internal diagnostics. Missing model telemetry is unknown,
not zero. All device actions must use `episode.invoke(name, arguments)`.

`environment.py` owns the home, devices/services, fixed catalogues, clock,
state transitions, fault injection and recorder. It imports no participant code.
The participant receives no gold goals, failure labels or reference annotations.
Calendar and fridge contents are learned through their tools. MAS translates public
schemas into its native cards/contracts and uses its existing controller; the
wrapper adds no planning or repair loop. ReAct, DS-IA and SAGE keep their existing
architectures. The wrappers contain no benchmark score calculation.

Each case starts a fresh `Session`. All beginning values come from the versioned
`initial_home()` profile plus that case's explicit `scenario` room selection,
state overrides, time and public preferences. This includes guest/kids/study/dining
rooms where required, dimmable lights, thermostats, fridge expiry dates and to-do
information. Temperatures describe thermostat setpoints, not physical heating.
Media and information are synthetic catalogue entries, not real streaming/web results.

`evaluation.py` alone computes the extended benchmark's scores from the request's
reference predicates, recorded calls, user-facing answer and full observed state.
Independent external benchmarks retain their native environments/evaluators.

## Automatic evaluation

Every extended case receives the same result structure, including inherited cases.
There is no manual Complete/Partial/Failed step and no LLM judge. Each result has
an automatic binary `evaluation.score` and the individual checks explaining it.

- Required state predicates must hold. Intermediate states are permitted only
  for goals explicitly annotated as sequential/temporary.
- Information goals require an actual recorded observation. Selected operation
  arguments, durations, dates and media/content terms must match the published
  reference rules. The recipe-dependent timer checks the cooking time of the
  selected, named catalogue recipe rather than requiring one fixed duration.
  Inventory-based selection rules also check the known recipe ingredients against
  the actual starting inventory; meal-plan variants require the declared number
  of distinct named recipes.
- Result dependencies require successful prerequisites before dependent calls.
  Order-only dependencies allow continuation after a failed prerequisite attempt.
  Computational answer goals can combine independent reads in either order.
- A complete-state comparison checks every field outside `allowed_changes`.
  Wrong values inside requested fields fail the corresponding goal. Media/display content must still match in the final state unless the goal is explicitly temporary. The trace catches forbidden
  writes even if the final state is later restored.
- Existing calendar entries and list items must be preserved. Duplicate timers,
  alarms/events and excess list writes fail, including duplicates with different labels.
- Scheduled effects run on the virtual clock after the participant returns.
  Their execution time, operation and arguments are checked. Acting immediately
  instead of waiting for an explicitly future action fails. No real-time waiting
  or hidden action scheduling is performed on a participant's behalf.
- Ambiguity 0 means an explicit task; 1 means omitted settings/targets are uniquely resolved by public context; 2 requires a focused clarification for unresolved parts. All 26 level-2 cases require a question about unresolved goals; ten also require the clear independent goals to execute. Only the unresolved parts and dependent actions must remain untouched. No user reply is simulated or awaited.
- Music playback never requires a separate music-search call. `simple_06` requires playback for one hour followed by stopping, with no extra countdown timer. `complex_17` likewise requires timed effects without separate countdowns. Necessary power preparation is allowed; unrelated volume, lighting and temperature changes are not.
- Conditional variants cover each outcome, including all four home/rain combinations, eggs/flour combinations and calendar/ingredient/light-status alternatives. `inactive_branch_calls` rejects executing a skipped branch, even when a call leaves the state unchanged.
- The unique audible-player cases accept muting, stopping or switching off the actual sound source. Other devices and temperatures must stay unchanged. Equivalent ways to fulfill a specified goal are not unresolved ambiguity.
- The public context includes default device routing, saved numerical preferences and a search location. Explicit targets and unique observable referents take precedence over defaults. Each annotation records its rationale in `ambiguity_reason`.
- Permanent failures require truthful user-facing failure reporting, preserved
  independent goals and no dependent writes using unavailable information.
  Transient/power-cut scenarios require actual recovery of the requested goals.

Text/content checks are intentionally **lexical**, using `all_of`, `any_of`,
`none_of` and optional pattern-count rules stored per case. They support the listed
English/German forms. They do not prove arbitrary semantic equivalence or fully
understand negation, recipe quality, entertainment suitability, or unsupported
paraphrases. Keep fixture/canonical names in displayed content when possible.
The score means "passed the published deterministic checks", not a general proof
of natural-language correctness. No hidden judge or manual fallback is used.

## Metrics

Mandatory: total participant wall time; every tool call/argument/result/error;
actual injections; initial/final state; per-goal fulfillment; applicable dependency,
conflict, clarification and schedule checks; unwanted changes and duplicate writes;
failure reporting and recovery when a fault was exercised. `response_match` is
reported only where lexical answer requirements exist. Not-applicable fields are null.
`intent_recall` counts required executable user goals, excluding technical helper steps; it is null when no such goal is executable. It can be below 1 in a correctly handled
permanent-failure scenario; the overall scenario score follows its recovery rules.

Optional: logical LLM calls, transport attempts/retries/errors and backend token
counts. Incomplete token telemetry is null and never silently totaled as complete.
System diagnostics remain in the raw `report`. **Summed model-call time, Lean
acceptance rate and MAS-internal execution batches are not standard metrics.**
The summed model-call time may remain inside participant-owned raw diagnostics.
Tool counts include unsuccessful system attempts; scheduled calls are counted
separately. Simulator evaluator read-back is not an agent action.

`summary.json` groups by suite, original task category/dimension, failure mode and
paired task ID. Evaluation errors leave the score null, keep coverage explicit and
cause a nonzero CLI exit. Resume refuses changed sources/settings and does not
repeat already completed participant episodes. A configured but never triggered
fault is `not_exercised`, gives no recovery credit, and cannot yield a robustness pass.

## Complexity and checks

See `EXTENSION.md` for the reviewed complexity/ambiguity rules and the implementation
audit. Print all vectors with `python ../benchmark/complexity.py --json`.

Offline integration checks, from the project root:

```bash
python -m unittest discover -s benchmark -p 'test_*.py' -v
```

They exercise the actual MAS controller and ReAct/DS-IA loops using scripted model
responses, plus SAGE's parent/worker protocol with a test worker. They do not claim
an end-to-end run of upstream SAGE or live language-model experiments.
