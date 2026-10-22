/-
  Executable Boolean allocation checker.

  This module does not import `Spec.lean`; the connection between the two is
  established only by the theorems in `Soundness.lean`.
-/
import AllocationChecker.Types

namespace AllocationChecker

/-! ### Lookups -/

def lookupOp (cat : Catalogue) (opId : Nat) : Option OpSpec :=
  cat.ops.find? (fun o => o.id == opId)

def lookupDevice (cat : Catalogue) (devId : Nat) : Option Device :=
  cat.devices.find? (fun d => d.id == devId)

def lookupTask (g : TaskGraph) (tid : Nat) : Option Task :=
  g.tasks.find? (fun t => t.id == tid)

def lookupEntry (s : Snapshot) (k : StateKey) : Option Entry :=
  s.state.find? (fun e => e.key == k)

/-! ### Generic Boolean helpers -/

/-- Duplicate-free check. -/
def nodupB [DecidableEq α] : List α → Bool
  | [] => true
  | x :: xs => !decide (x ∈ xs) && nodupB xs

/-- Pairwise check over all ordered pairs `(xᵢ, xⱼ)` with `i < j`. -/
def pairwiseB (r : α → α → Bool) : List α → Bool
  | [] => true
  | x :: xs => xs.all (r x) && pairwiseB r xs

def disjointB (a b : List StateKey) : Bool :=
  a.all (fun k => !decide (k ∈ b))

/-! ### Well-formedness -/

def checkCatalogueWF (cat : Catalogue) : Bool :=
  nodupB (cat.ops.map (·.id)) && nodupB (cat.devices.map (·.id))

def checkSnapshotWF (s : Snapshot) : Bool :=
  nodupB (s.state.map (·.key))

def rankOk (g : TaskGraph) (t : Task) (d : Nat) : Bool :=
  match lookupTask g d with
  | some u => decide (u.rank < t.rank)
  | none => false

def checkGraphWF (g : TaskGraph) : Bool :=
  nodupB (g.tasks.map (·.id)) && g.tasks.all (fun t => t.deps.all (rankOk g t))

/-! ### Per-command checks -/

def eligibleB : Status → Bool
  | .pending => true
  | .invalidated => true
  | _ => false

def cmpB : Cmp → Int → Int → Bool
  | .eq, a, b => decide (a = b)
  | .ne, a, b => decide (a ≠ b)
  | .le, a, b => decide (a ≤ b)
  | .ge, a, b => decide (a ≥ b)

def preHoldsB (s : Snapshot) (p : Precondition) : Bool :=
  match lookupEntry s p.key with
  | some e => cmpB p.cmp e.val p.val
  | none => false

def basisCurrentB (s : Snapshot) (kv : StateKey × Nat) : Bool :=
  match lookupEntry s kv.1 with
  | some e => e.ver == kv.2
  | none => false

def evidenceValidB (s : Snapshot) (ev : Evidence) (d : Nat) : Bool :=
  ev.task == d && ev.verified && decide (s.now < ev.validUntil) &&
    ev.basis.all (basisCurrentB s)

def depSatisfiedB (s : Snapshot) (g : TaskGraph) (d : Nat) : Bool :=
  (match lookupTask g d with
   | some u => u.status == .verified
   | none => false) &&
  s.evidence.any (fun ev => evidenceValidB s ev d)

def argResolvedB (s : Snapshot) (t : Task) : ArgSrc → Int → Bool
  | .lit w, v => v == w
  | .ref d off, v =>
      decide (d ∈ t.deps) &&
        s.evidence.any (fun ev => evidenceValidB s ev d && ev.output == some (v - off))

def inBoundsB (b : ArgBound) (v : Int) : Bool :=
  decide (b.lo ≤ v) && decide (v ≤ b.hi)

def policyB (cat : Catalogue) (s : Snapshot) (o : OpSpec) (c : Command) : Bool :=
  if o.coupled then
    match lookupEntry s cat.meterKey, lookupEntry s { dev := c.device, attr := cat.loadAttr } with
    | some m, some cur => decide (m.val - cur.val + projectedLoad o c.args ≤ cat.loadBudget)
    | _, _ => false
  else true

def commandFitsB (cat : Catalogue) (s : Snapshot) (g : TaskGraph) (c : Command)
    (t : Task) (o : OpSpec) (d : Device) : Bool :=
  eligibleB t.status &&
  t.op == c.op &&
  t.device == c.device &&
  decide (o.cap ∈ d.caps) &&
  decide (o.id ∈ d.ops) &&
  c.args.length == o.bounds.length &&
  t.args.length == o.bounds.length &&
  (o.bounds.zip c.args).all (fun p => inBoundsB p.1 p.2) &&
  (t.args.zip c.args).all (fun p => argResolvedB s t p.1 p.2) &&
  t.deps.all (depSatisfiedB s g) &&
  t.pre.all (preHoldsB s) &&
  o.pre.all (fun p => preHoldsB s (p.at c.device)) &&
  decide (s.now < t.deadline) &&
  policyB cat s o c

def checkCommand (cat : Catalogue) (s : Snapshot) (g : TaskGraph) (c : Command) : Bool :=
  match lookupTask g c.task, lookupOp cat c.op, lookupDevice cat c.device with
  | some t, some o, some d => commandFitsB cat s g c t o d
  | _, _, _ => false

/-! ### Resource independence -/

def indepB (f g : Footprint) : Bool :=
  disjointB f.writes (g.reads ++ g.writes) && disjointB g.writes f.reads &&
    disjointB f.excl g.excl

def cmdFootprint? (cat : Catalogue) (c : Command) : Option Footprint :=
  (lookupOp cat c.op).map (fun o => opFootprint cat o c.device)

def pairIndepB (cat : Catalogue) (a b : Command) : Bool :=
  match cmdFootprint? cat a, cmdFootprint? cat b with
  | some fa, some fb => indepB fa fb
  | _, _ => false

def activeIndepB (cat : Catalogue) (s : Snapshot) (c : Command) : Bool :=
  match cmdFootprint? cat c with
  | some fc => s.active.all (indepB fc)
  | none => false

def checkIndependence (cat : Catalogue) (s : Snapshot) (cs : List Command) : Bool :=
  pairwiseB (pairIndepB cat) cs && cs.all (activeIndepB cat s)

/-! ### Stable preconditions -/

/-- The keys the task preconditions of command `c` are about. -/
def taskPreKeys (g : TaskGraph) (c : Command) : List StateKey :=
  match lookupTask g c.task with
  | some t => t.pre.map (·.key)
  | none => []

def keepsTaskPreB (cat : Catalogue) (g : TaskGraph) (a b : Command) : Bool :=
  match cmdFootprint? cat a with
  | some fa => disjointB fa.writes (taskPreKeys g b)
  | none => false

def checkTaskPreStable (cat : Catalogue) (s : Snapshot) (g : TaskGraph) (cs : List Command) :
    Bool :=
  pairwiseB (fun a b => keepsTaskPreB cat g a b && keepsTaskPreB cat g b a) cs &&
    cs.all (fun c => s.active.all (fun f => disjointB f.writes (taskPreKeys g c)))

/-! ### Top-level checker -/

/-- The allocation checker covered by `checkAllocation_sound`. -/
def checkAllocation (s : Snapshot) (cat : Catalogue) (g : TaskGraph)
    (b : AllocationBatch) : Bool :=
  checkCatalogueWF cat &&
  checkSnapshotWF s &&
  checkGraphWF g &&
  b.graphRev == g.revision &&
  b.catalogueRev == cat.revision &&
  decide (b.commands.length ≤ s.workerCap) &&
  nodupB (b.commands.map (·.task)) &&
  b.commands.all (checkCommand cat s g) &&
  checkIndependence cat s b.commands &&
  checkTaskPreStable cat s g b.commands

end AllocationChecker
