/-
  Declarative specification of a valid allocation batch.

  `AllocationValid` is stated with membership, quantifiers and `List.Nodup` /
  `List.Pairwise`. It does not mention the executable checker in `Check.lean`.
-/
import AllocationChecker.Types

namespace AllocationChecker

/-! ### Reference relations -/

def OpOf (cat : Catalogue) (opId : Nat) (o : OpSpec) : Prop :=
  o ∈ cat.ops ∧ o.id = opId

def DeviceOf (cat : Catalogue) (devId : Nat) (d : Device) : Prop :=
  d ∈ cat.devices ∧ d.id = devId

def TaskOf (g : TaskGraph) (tid : Nat) (t : Task) : Prop :=
  t ∈ g.tasks ∧ t.id = tid

def EntryOf (s : Snapshot) (k : StateKey) (e : Entry) : Prop :=
  e ∈ s.state ∧ e.key = k

/-! ### Well-formedness: known, unambiguous references and a rank witness -/

/-- Operation and device identifiers of the catalogue are unambiguous. -/
def CatalogueWF (cat : Catalogue) : Prop :=
  (cat.ops.map (·.id)).Nodup ∧ (cat.devices.map (·.id)).Nodup

/-- Each state key occurs at most once in the snapshot. -/
def SnapshotWF (s : Snapshot) : Prop :=
  (s.state.map (·.key)).Nodup

/-- Task identifiers are unambiguous, every dependency refers to a task of the
    graph, and the supplied ranks strictly increase along dependency edges
    (which certifies acyclicity, see `Soundness.lean`). -/
def GraphWF (g : TaskGraph) : Prop :=
  (g.tasks.map (·.id)).Nodup ∧
  ∀ t ∈ g.tasks, ∀ d ∈ t.deps, ∃ u, TaskOf g d u ∧ u.rank < t.rank

/-! ### Per-command conditions -/

/-- Only pending or explicitly invalidated tasks may be scheduled. -/
def Eligible (st : Status) : Prop :=
  st = .pending ∨ st = .invalidated

/-- A precondition holds in the snapshot: the key is observed and its value
    satisfies the comparison. A missing key never satisfies a precondition. -/
def PreHolds (s : Snapshot) (p : Precondition) : Prop :=
  ∃ e, EntryOf s p.key e ∧ p.cmp.Holds e.val p.val

/-- Evidence `ev` is verified, unexpired at `s.now`, about task `d`, and every
    state version it was observed against is still current. -/
def EvidenceValid (s : Snapshot) (ev : Evidence) (d : Nat) : Prop :=
  ev.task = d ∧ ev.verified = true ∧ s.now < ev.validUntil ∧
  ∀ kv ∈ ev.basis, ∃ e, EntryOf s kv.1 e ∧ e.ver = kv.2

/-- A dependency is satisfied: its task is marked verified in the graph and a
    still-valid evidence record supports that status. -/
def DepSatisfied (s : Snapshot) (g : TaskGraph) (d : Nat) : Prop :=
  (∃ u, TaskOf g d u ∧ u.status = .verified) ∧
  ∃ ev, ev ∈ s.evidence ∧ EvidenceValid s ev d

/-- A concrete argument value is resolved from the task contract: either the
    literal itself, or the verified output of a declared dependency plus the
    contract offset. -/
def ArgResolved (s : Snapshot) (t : Task) : ArgSrc → Int → Prop
  | .lit w, v => v = w
  | .ref d off, v =>
      d ∈ t.deps ∧ ∃ ev, ev ∈ s.evidence ∧ EvidenceValid s ev d ∧ ev.output = some (v - off)

def InBounds (b : ArgBound) (v : Int) : Prop :=
  b.lo ≤ v ∧ v ≤ b.hi

/-- Household load policy for coupled operations: the observed meter total,
    minus the device's current load, plus its projected load stays within the
    catalogue budget. Coupled operations are serialised by their footprints. -/
def PolicyHolds (cat : Catalogue) (s : Snapshot) (o : OpSpec) (c : Command) : Prop :=
  o.coupled = true →
    ∃ m cur, EntryOf s cat.meterKey m ∧ EntryOf s { dev := c.device, attr := cat.loadAttr } cur ∧
      m.val - cur.val + projectedLoad o c.args ≤ cat.loadBudget

/-- All conditions on one command, given the task, operation and device it
    refers to. -/
structure CommandFits (cat : Catalogue) (s : Snapshot) (g : TaskGraph) (c : Command)
    (t : Task) (o : OpSpec) (d : Device) : Prop where
  eligible   : Eligible t.status
  opMatches  : t.op = c.op
  devMatches : t.device = c.device
  capability : o.cap ∈ d.caps
  supported  : o.id ∈ d.ops
  arity      : c.args.length = o.bounds.length
  contractArity : t.args.length = o.bounds.length
  bounds     : ∀ p ∈ o.bounds.zip c.args, InBounds p.1 p.2
  resolved   : ∀ p ∈ t.args.zip c.args, ArgResolved s t p.1 p.2
  deps       : ∀ dep ∈ t.deps, DepSatisfied s g dep
  taskPre    : ∀ p ∈ t.pre, PreHolds s p
  opPre      : ∀ p ∈ o.pre, PreHolds s (p.at c.device)
  deadline   : s.now < t.deadline
  policy     : PolicyHolds cat s o c

/-- A command refers to a known task, operation and device, and fits them. -/
def CommandValid (cat : Catalogue) (s : Snapshot) (g : TaskGraph) (c : Command) : Prop :=
  ∃ t o d, TaskOf g c.task t ∧ OpOf cat c.op o ∧ DeviceOf cat c.device d ∧
    CommandFits cat s g c t o d

/-! ### Resource independence -/

def KeysDisjoint (a b : List StateKey) : Prop :=
  ∀ k ∈ a, k ∉ b

/-- Conservative independence of two footprints:
    `W₁ ∩ (R₂ ∪ W₂) = ∅`, `W₂ ∩ R₁ = ∅` and `X₁ ∩ X₂ = ∅`.
    Simultaneous reads are permitted. -/
def Indep (f g : Footprint) : Prop :=
  KeysDisjoint f.writes (g.reads ++ g.writes) ∧
  KeysDisjoint g.writes f.reads ∧
  KeysDisjoint f.excl g.excl

/-- `f` is the trusted footprint of command `c`. -/
def CmdFootprint (cat : Catalogue) (c : Command) (f : Footprint) : Prop :=
  ∃ o, OpOf cat c.op o ∧ f = opFootprint cat o c.device

/-- Distinct commands of the batch are pairwise independent, and every command
    is independent of the reservations of running operations. -/
def ResourceIndependent (cat : Catalogue) (s : Snapshot) (cs : List Command) : Prop :=
  cs.Pairwise (fun a b => ∀ fa fb, CmdFootprint cat a fa → CmdFootprint cat b fb → Indep fa fb) ∧
  ∀ c ∈ cs, ∀ fc, CmdFootprint cat c fc → ∀ f ∈ s.active, Indep fc f

/-! ### Valid allocation batch -/

/-- Declarative correctness of an allocation batch at a snapshot. -/
structure AllocationValid (s : Snapshot) (cat : Catalogue) (g : TaskGraph)
    (b : AllocationBatch) : Prop where
  catalogueWF : CatalogueWF cat
  snapshotWF  : SnapshotWF s
  graphWF     : GraphWF g
  graphRev    : b.graphRev = g.revision
  catalogueRev : b.catalogueRev = cat.revision
  workerCap   : b.commands.length ≤ s.workerCap
  uniqueTasks : (b.commands.map (·.task)).Nodup
  commands    : ∀ c ∈ b.commands, CommandValid cat s g c
  independent : ResourceIndependent cat s b.commands

end AllocationChecker
