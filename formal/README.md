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
3. every state variable the operation writes is determined (its room is known from the card or
   from the task's arguments),
4. the operation's preconditions hold in the observed state, and
5. no two tasks of the batch write the same state variable (write–write conflict freedom).

A *state variable* is one property of one device, optionally in one room, e.g. the temperature
setting of the thermostat in the bedroom. Cards refer to variables of their own device only and
may leave the room to a task argument, so a task can never write another device's state.

The theorem is proved without `sorry` and uses only Lean's standard axioms.

`SmartHome/Examples.lean` contains a thermostat card, the lighting card from the extension
plan, and batches that are accepted (two devices; the same device in two rooms) or rejected
(write conflict, out-of-range or wrongly typed argument, missing argument, identifier of the
wrong kind, unknown device, a room without that device, a device that is offline). The
`#guard` lines and the `example` are checked at build time.

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
* Cards can only refer to their own device's variables, for reading as well as writing. Reading
  another device's state (e.g. a thermostat reading a window sensor) is a next step.
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
