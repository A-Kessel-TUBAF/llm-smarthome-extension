/-
  Finite data model for the smart-home allocation checker.

  Every identifier is a natural number. The Python integration layer maps the
  string identifiers of devices, attributes, operations and tasks to these
  numbers deterministically (see `src/dexa_verified/lean_client.py`).
  Values are bounded integers; enumerated modes are encoded as integers by the
  trusted catalogue.

  This module imports only Lean's `Init` prelude.
-/
namespace AllocationChecker

/-- A state key: an attribute of a device (or of a virtual device such as the
    household load meter). -/
structure StateKey where
  dev  : Nat
  attr : Nat
  deriving DecidableEq, Repr, Inhabited

/-- Lifecycle status of a task-graph node. -/
inductive Status where
  | pending
  | running
  | verified
  | failed
  | invalidated
  deriving DecidableEq, Repr, Inhabited

/-- Comparison operators of the finite precondition language. -/
inductive Cmp where
  | eq
  | ne
  | le
  | ge
  deriving DecidableEq, Repr, Inhabited

/-- Semantics of a comparison: `Cmp.Holds c observed expected`. -/
def Cmp.Holds : Cmp → Int → Int → Prop
  | .eq, a, b => a = b
  | .ne, a, b => a ≠ b
  | .le, a, b => a ≤ b
  | .ge, a, b => a ≥ b

/-- A precondition on an explicit state key. -/
structure Precondition where
  key : StateKey
  cmp : Cmp
  val : Int
  deriving DecidableEq, Repr

/-- An intrinsic operation precondition on an attribute of the target device. -/
structure AttrPre where
  attr : Nat
  cmp  : Cmp
  val  : Int
  deriving DecidableEq, Repr

/-- Instantiate an intrinsic precondition for a concrete device. -/
def AttrPre.at (p : AttrPre) (dev : Nat) : Precondition :=
  { key := { dev := dev, attr := p.attr }, cmp := p.cmp, val := p.val }

/-- Inclusive integer bounds of one operation argument. -/
structure ArgBound where
  lo : Int
  hi : Int
  deriving DecidableEq, Repr

/-- Source of a task argument in the task contract: a literal, or the verified
    output of a dependency plus an offset. -/
inductive ArgSrc where
  | lit (v : Int)
  | ref (task : Nat) (offset : Int)
  deriving DecidableEq, Repr

/-- Trusted catalogue entry for one operation. Footprints are given relative to
    the target device and are instantiated by `opFootprint`. -/
structure OpSpec where
  id          : Nat
  cap         : Nat
  bounds      : List ArgBound
  reads       : List Nat
  writes      : List Nat
  excl        : List Nat
  sharedReads : List StateKey
  pre         : List AttrPre
  /-- Participates in the shared household load policy. -/
  coupled     : Bool
  loadBase    : Int
  loadPerUnit : Int
  deriving Repr

structure Device where
  id   : Nat
  caps : List Nat
  ops  : List Nat
  deriving Repr

structure Catalogue where
  revision   : Nat
  ops        : List OpSpec
  devices    : List Device
  /-- Device attribute that stores the device's current modeled load. -/
  loadAttr   : Nat
  /-- State key of the household load meter (sum of modeled device loads). -/
  meterKey   : StateKey
  loadBudget : Int
  deriving Repr

structure Task where
  id       : Nat
  status   : Status
  op       : Nat
  device   : Nat
  args     : List ArgSrc
  deps     : List Nat
  /-- Topological rank witness: every dependency has a strictly smaller rank. -/
  rank     : Nat
  pre      : List Precondition
  deadline : Nat
  deriving Repr

structure TaskGraph where
  revision : Nat
  tasks    : List Task
  deriving Repr

/-- One observed state entry with its per-key version. -/
structure Entry where
  key : StateKey
  val : Int
  ver : Nat
  deriving DecidableEq, Repr

/-- An evidence record about the outcome of a task. `basis` lists the state-key
    versions against which the outcome was observed. -/
structure Evidence where
  task       : Nat
  verified   : Bool
  validUntil : Nat
  output     : Option Int
  basis      : List (StateKey × Nat)
  deriving Repr

/-- Read, write and exclusive-resource keys of an operation instance. -/
structure Footprint where
  reads  : List StateKey
  writes : List StateKey
  excl   : List StateKey
  deriving Repr

structure Snapshot where
  now       : Nat
  workerCap : Nat
  state     : List Entry
  evidence  : List Evidence
  /-- Footprints reserved by operations that are still running. -/
  active    : List Footprint
  deriving Repr

/-- A concrete device command proposed for one task. -/
structure Command where
  task   : Nat
  device : Nat
  op     : Nat
  args   : List Int
  deriving DecidableEq, Repr

structure AllocationBatch where
  graphRev     : Nat
  catalogueRev : Nat
  commands     : List Command
  deriving Repr

/-- Footprint of an operation on a concrete target device, derived only from
    the trusted catalogue. Operations coupled by the household load policy also
    read and write the meter key and the device's load attribute, which
    serialises them conservatively. -/
def opFootprint (cat : Catalogue) (o : OpSpec) (dev : Nat) : Footprint :=
  let onDev := fun (a : Nat) => ({ dev := dev, attr := a } : StateKey)
  let policy : List StateKey :=
    if o.coupled then [cat.meterKey, { dev := dev, attr := cat.loadAttr }] else []
  { reads  := o.reads.map onDev ++ o.sharedReads ++ policy
    writes := o.writes.map onDev ++ policy
    excl   := o.excl.map onDev }

/-- Projected modeled load of the target device after the operation. -/
def projectedLoad (o : OpSpec) (args : List Int) : Int :=
  o.loadBase + o.loadPerUnit * args.headD 0

end AllocationChecker
