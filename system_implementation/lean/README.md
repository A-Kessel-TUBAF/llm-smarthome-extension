# Lean allocation checker

A batch of device commands planned by the LLM is executed only if this checker accepts it. The
checker is written in Lean 4, and its decisions are proven correct.

```bash
cd system_implementation/lean
lake build    # builds the checker, checks all proofs and runs the axiom audit (~10 s)
```

The Python system uses it with `create_initial_state(checker="lean")` (see `check_allocation`
in `smart_home_langgraph.py`).

## Admission (when may a batch run?)

| File | Content |
|---|---|
| `Types.lean` | data model: catalogue, task graph, snapshot of the observed state, batch |
| `Spec.lean` | `AllocationValid`: when a batch may be admitted (capabilities, argument bounds, dependencies with valid evidence, preconditions, deadline, load budget, worker capacity, pairwise independent footprints, and preconditions that no other command or running operation may change) |
| `Check.lean` | `checkAllocation`: the executable checker |
| `Soundness.lean`, `Completeness.lean` | `checkAllocation_iff`: the checker accepts exactly the valid batches |
| `Examples.lean` | accepted and rejected example batches, checked by the kernel |
| `Codec.lean`, `Main.lean` | the JSON interface used from Python |
| `TypedValues.lean` | typed argument values at the JSON interface (below) |
| `Audit.lean` | fails the build if a theorem uses an axiom beyond Lean's standard ones |

**Typed argument values.** The catalogue may declare the type of each argument of an
operation: `{"number": scale}`, `"bool"` or `{"enum": [...]}`. The checker then decodes the
agents' values itself (`21.5` with scale 10 becomes `215`, `true` becomes `1`, `"heat"` its
position in the list). `decodeTyped_iff` proves this decoding exact: it succeeds exactly for
values of the declared type, so a wrongly typed argument makes the request invalid instead of
being turned into some number. Operations without declared types are decoded as before.

**Stable preconditions.** A command's preconditions are checked against the snapshot, so they
must not change before it starts. The keys an operation's preconditions are about therefore count
as reads of its footprint (`opFootprint`), and `TaskPreStable` requires that no other command of
the batch and no running operation writes a key a task precondition is about. For example, a
batch that dims a light while another task requires it to be bright is rejected
(`preDisturbed_invalid` in `Examples.lean`).

## Execution (what happens while admitted batches run?)

The specification says when a batch may be admitted. These modules prove what that guarantees
when it runs. They model the devices abstractly: a `Behaviour` gives the state after a command,
and it only has to respect the catalogue's footprints (`Respects`): a command changes only the
keys it writes, and what it writes depends only on the keys it reads. Nothing is assumed about
the values written.

| File | Main results |
|---|---|
| `Execution.lean` | **Parallel execution is safe.** `admitted_order_irrelevant`: every execution order of an admitted batch gives the same result as running all its commands at once. `admitted_reads_stable`: each command starts with the original values of what it reads. `admitted_preconditions_hold`: each command's preconditions still hold when it starts. |
| `Failures.lean` | **Failures stay contained.** A failed command may leave any values on the keys it writes. `admitted_failures_isolated`: every key no failed command writes ends as predicted, so the other commands achieve exactly their effect. **Observed-state validation is exact.** `validated_iff_predicted`: checking each command's outcome on its own is the same as checking the whole batch; `admitted_successful_validated`: a command that succeeded always passes, whatever else failed. |
| `Repair.lean` | **Local repair is sound.** Over a plan of several rounds, skipping only the commands that read a key written by a failed or skipped command, or whose task depends on a failed or skipped task, keeps the rest on plan. `repair_sound`: every command that still runs starts with its predicted reads, and every key outside the tainted ones ends as predicted; `repair_no_failures`: without failures nothing is skipped. |
| `Invariants.lean` | **Safety limits hold over time.** If operations set attributes to their (bounded) arguments, as the registry's `equals_arg` postconditions say, `reachable_within`: every state reachable by running admitted batches stays within the limits, e.g. brightness within 0–100. |
| `ExecutionExamples.lean` | the results applied to the batches of `Examples.lean` with a concrete device behaviour |

All theorems are proved without `sorry` and use only Lean's standard axioms (`Audit.lean`).

**Limits of the model.** The execution results hold for device behaviours that respect the
declared footprints; a device that writes keys it does not declare is outside the model (its
deviations are what observed-state validation detects). The safety limits concern the system's
own commands; changes made by people are not modelled.
