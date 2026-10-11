# Formal verification of the allocation checker (Lean 4)

This folder contains a machine-checked proof that the **allocation checker** only accepts task
batches that satisfy the **allocation specification**. It is the first step of the "Integrate
Lean" item of the DEXA extension plan.

## What is proved

An *agent card* declares, per device, the operations it offers: parameters with ranges, the
state variables each operation reads and writes, and pre-/postconditions
(`SmartHome/Cards.lean`). A *batch* is a list of planner tasks meant to run together.

`SmartHome/Allocation.lean` defines three things:

| Name | Kind | Meaning |
|---|---|---|
| `AllocSpec` | specification (`Prop`) | what "well-allocated" means; this is what a human reviews |
| `check` | checker (`Bool`) | executable code that accepts or rejects a batch |
| `check_sound` | theorem | `check cards state batch = true → AllocSpec cards state batch` |

A batch is well-allocated when

1. every task names a device that has a card, and an operation on that card,
2. every parameter of the operation is supplied, with a value of its type (booleans,
   integers inside their range, identifiers of the right kind such as `RoomID`),
3. the operation's preconditions hold in the observed state, and
4. no two tasks of the batch write the same state variable (write–write conflict freedom).

The theorem is proved without `sorry` and uses only Lean's standard axioms.

`SmartHome/Examples.lean` contains the lighting card from the extension plan, a thermostat
card, and batches that are accepted or rejected (unknown device, out-of-range argument,
missing argument, wrongly typed argument, identifier of the wrong kind, write conflict,
failed precondition). The `#guard` lines are checked at
build time.

## Building

Install [elan](https://github.com/leanprover/elan) (the Lean version manager), then:

```bash
cd formal
lake build                  # compiles definitions, checks all proofs (~5 s, no Mathlib needed)
lake exe allocation-check   # runs the checker on the example batches
```

The Lean version is pinned in `lean-toolchain`; `elan` downloads it on first use.

## Scope and next steps

* Values are typed: booleans, integers and identifiers of a kind (e.g. `RoomID`). Identifiers
  are checked by kind only; any room name is accepted. A list of valid rooms is a next step.
* State variables are not indexed by room: `set_brightness` writes `light.brightness` whatever
  the `room` argument is. So dimming the bedroom and the living room in one batch is
  rejected as a write conflict. Room-indexed variables (e.g. `light[room].brightness`) would
  fix this.
* An argument supplied twice is checked only at its first occurrence. Rejecting duplicate or
  unknown arguments is a next step.
* Conflicts are write–write only. Read–write conflicts (one task reads what another writes)
  are a natural next step.
* `check_sound` is *soundness* (accepted ⇒ well-allocated). *Completeness*
  (well-allocated ⇒ accepted) is not yet proved.
* Postconditions are recorded on cards but not yet used; they belong to observed-state
  validation after execution.
* Not yet connected to the Python system: agent cards and batches would be exported
  (e.g. as JSON) and passed to `allocation-check` before a batch is executed.
