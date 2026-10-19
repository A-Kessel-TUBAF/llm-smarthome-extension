# Benchmark extension (DRAFT)

This is a proposal for the "Expand the benchmark" item of the DEXA extension plan, for review by
the team. It adds, without changing `benchmark_data.json`:

* a computable definition of the task complexity vector C(task) (`complexity.py`), with
  annotations of the 60 original cases (`complexity_annotations.json`),
* 40 additional test cases, 8 for each dimension the extension should measure: dependency
  structure, ambiguity, scale, concurrency and failure recovery (`extension_tasks.json`),
* additional metrics for comparing systems.

All new cases use the 8 existing device types, so the current system can run them unchanged.

```bash
cd benchmark
python complexity.py          # C(task) of all 100 cases, and averages per category / dimension
python complexity.py --json   # the vectors as JSON
```

## 1. Complexity vector C(task)

```
C(task) = (intents, devices, graph_depth, graph_width, ambiguity,
           temporal_constraints, spatial_constraints, conflicts)
```

All features are computed on the *maximal interpretation* of a case: every listed intent,
required and acceptable.

| Feature | Definition |
|---|---|
| `intents` | number of listed intents (`required_intents` + `acceptable_intents`) |
| `devices` | number of distinct device types among them |
| `graph_depth` | longest chain in the dependency graph, counted in tasks (1 = no dependencies) |
| `graph_width` | most tasks on one level of the dependency graph, i.e. the most tasks that can run in parallel |
| `ambiguity` | number of acceptable intents (0 = explicit request) |
| `temporal_constraints` | number of time constraints: clock times, dates, durations, deadlines, required order ("then") |
| `spatial_constraints` | number of location constraints: rooms, "all the lights", "every room", "nearby" |
| `conflicts` | number of intent pairs that compete for the same device in the same place with incompatible settings or outputs |

**Dependency graph.** The nodes are the intents, plus helper devices that are not intents but
provide information (e.g. "Audio obtains music recommendations from search device" when search is
not one of the intents). An edge `a → b` means `b` needs the result of `a` (information
dependency) or must run after `a` because of it (action dependency, including conditions such as
"if it rains"). The level of a node is 1 if it has no prerequisites, otherwise one more than its
highest prerequisite; `graph_depth` is the highest level and `graph_width` the largest level.

**Annotations.** The features that cannot be read off the data (dependencies, constraints,
conflicts) are annotated per case:

```json
"moderate_03": {"deps": [[1, 0], [0, 2]], "temporal": [], "spatial": [], "conflicts": []}
```

Intent indices refer to `required_intents` followed by `acceptable_intents`; a device name in a
dependency is a helper device. For the original cases, `deps` follow the listed collaboration
conditions and notes; obvious dependencies that a case does not list are kept apart as
`inferred_deps` (3 cases), so that they can be reviewed. The new cases carry their annotation
in an `annotation` field.

## 2. Additional test cases

`extension_tasks.json` has the same format as `benchmark_data.json`, plus:

* `dimension`: what the case is designed to measure,
* `annotation`: as above,
* for concurrency: `expected_concurrency` with the groups of intents that may run in `parallel`
  and the pairs that must be serialized or clarified (`serialize_or_clarify`),
* for failure recovery: `injected_failure` (which intent fails, and how) and
  `expected_recovery` (which intents must still complete, which failure must be reported, and
  which dependent intents must be replanned instead of executed with missing information).

`category` follows the original rule (1–2 intents simple, 3 moderate, 4+ complex).

| Dimension | Cases | What varies |
|---|---|---|
| dependency structure | `ext_dep_01`–`08` | chains, fan-out, fan-in, conditional branches (depth up to 3) |
| ambiguity | `ext_amb_01`–`08` | vague requests ("Turn it down", "I can't sleep") with 2–4 acceptable intents |
| scale | `ext_scale_01`–`08` | 5–8 intents, up to 7 device types and several rooms |
| concurrency | `ext_conc_01`–`08` | 4 cases with contradictory requests for the same device, 4 with fully or partly parallel tasks |
| failure recovery | `ext_fail_01`–`08` | an injected failure, with and without dependent tasks |

Averages of C(task) per group (from `complexity.py`):

| group | n | intents | devices | depth | width | ambiguity | temporal | spatial | conflicts |
|---|---|---|---|---|---|---|---|---|---|
| simple | 20 | 1.9 | 1.8 | 1.4 | 1.5 | 0.6 | 0.1 | 0.2 | 0.0 |
| moderate | 20 | 2.7 | 2.6 | 1.7 | 2.2 | 0.7 | 0.6 | 0.1 | 0.1 |
| complex | 20 | 3.5 | 3.2 | 1.9 | 2.8 | 0.4 | 1.4 | 0.2 | 0.0 |
| dependency structure | 8 | 3.4 | 3.4 | 2.6 | 1.6 | 0.0 | 1.5 | 0.5 | 0.0 |
| ambiguity | 8 | 3.5 | 3.4 | 1.1 | 3.4 | 3.4 | 0.4 | 0.1 | 0.0 |
| scale | 8 | 6.6 | 5.4 | 2.0 | 5.4 | 0.0 | 2.2 | 2.9 | 0.0 |
| concurrency | 8 | 2.9 | 2.0 | 1.4 | 2.6 | 0.0 | 0.6 | 1.0 | 0.5 |
| failure recovery | 8 | 2.5 | 2.5 | 1.8 | 2.0 | 0.0 | 1.0 | 0.6 | 0.0 |

Each dimension raises the feature it targets, which the original cases hardly cover: dependency
depth, ambiguity, scale, and conflicts.

**Injected failures.** The benchmark runner (not yet implemented) makes the named device agent
fail in one of three ways:

* `unavailable`: the agent returns an error,
* `no_effect`: the agent reports success, but the observed state does not change (tests
  observed-state validation),
* `state_changed`: after planning, the device's state is changed from outside (tests
  state-versioned authorization).

## 3. Additional metrics

The current evaluation rates each case Complete / Partial / Failed by hand. Proposed additional
metrics, per case and averaged per category and dimension:

| Metric | Definition | Needs |
|---|---|---|
| Intent recall | share of required intents carried out | logs (as now) |
| Over-activation | intents carried out that are not listed (ambiguous cases) | logs |
| Dependency correctness | share of annotated dependencies respected: the dependent task ran after its prerequisite and used its result | logs with task order and inputs |
| Schedule length | number of sequential rounds of agent calls; compare with `graph_depth` (best possible) and `intents` (fully sequential) | logs with timing |
| Conflict handling | share of `serialize_or_clarify` pairs not executed at the same time (asked, serialized or one chosen explicitly), and share of `parallel` groups run concurrently | logs with timing |
| Failure detection | the injected failure is reported to the user | logs |
| Recovery locality | share of `must_complete` intents completed after the failure | logs |
| Recovery correctness | no `must_replan` intent executed with missing information | logs |
| Recovery cost | agent calls after the failure, compared with replanning everything | logs |
| Plan validity | share of planned batches accepted by the Lean checker (`system_implementation/lean`, called by `check_allocation`), and the rejection reasons | structured plans |
| Cost | LLM calls, tokens, wall-clock time | logs |

Most of these need structured logs (task, agent, start and end time, inputs received from other
agents) instead of the current free-text logs; the runner should write them.

## Open questions for the team

* Are the feature definitions the intended ones (e.g. should `intents` count acceptable intents,
  and should vague phrases such as "soon" count as temporal constraints)?
* The annotations of the 60 original cases are a first pass and should be checked, in particular
  the three `inferred_deps`.
* Should conditional dependencies ("if it rains") be a separate feature?
