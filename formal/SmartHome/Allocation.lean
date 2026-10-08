import SmartHome.Cards

/-!
# Allocation specification, checker and soundness

Three layers:

* `AllocSpec` — the *specification*: a mathematical statement of what it means for a batch
  to be well-allocated. This is what a human reviews.
* `check` — the *checker*: executable code that accepts or rejects a batch. This is what the
  system runs before executing a batch.
* `check_sound` — the *theorem*: every batch accepted by `check` satisfies `AllocSpec`.

A batch is well-allocated when
1. every task names a device that has a card and an operation on that card,
2. every parameter of the operation is supplied, with a value inside its range,
3. the operation's preconditions hold in the observed state, and
4. no two tasks of the batch write the same state variable.
-/

namespace SmartHome

/-! ## Lookups -/

/-- The card of device `d`, if there is one. -/
def findCard (cards : List AgentCard) (d : DeviceId) : Option AgentCard :=
  cards.find? (fun c ↦ c.device == d)

/-- The operation named `n` on card `c`, if there is one. -/
def findOp (c : AgentCard) (n : OpName) : Option Operation :=
  c.ops.find? (fun o ↦ o.name == n)

/-- The value of argument `p` in `args`, if supplied. -/
def argVal (args : List (String × Int)) (p : String) : Option Int :=
  (args.find? (fun a ↦ a.1 == p)).map (·.2)

/-- The value of state variable `v` in state `s`, if known. -/
def stateVal (s : State) (v : StateVar) : Option Int :=
  (s.find? (fun a ↦ a.1 == v)).map (·.2)

/-- The operation a task refers to, if its device and operation exist. -/
def resolve (cards : List AgentCard) (t : Task) : Option Operation :=
  (findCard cards t.device).bind (fun c ↦ findOp c t.op)

/-- The state variables written by task `t` (empty if the task cannot be resolved). -/
def writesOf (cards : List AgentCard) (t : Task) : List StateVar :=
  match resolve cards t with
  | some op => op.writes
  | none => []

/-! ## Specification -/

/-- Every parameter of `op` is supplied by `args` with a value inside its range. -/
def ArgsOK (op : Operation) (args : List (String × Int)) : Prop :=
  ∀ p ∈ op.params, ∃ v, argVal args p.name = some v ∧ p.lo ≤ v ∧ v ≤ p.hi

/-- Condition `c` holds in state `s` under arguments `args`. -/
def CondHolds (s : State) (args : List (String × Int)) : Cond → Prop
  | .eqConst var c => stateVal s var = some c
  | .eqParam var p => ∃ v, argVal args p = some v ∧ stateVal s var = some v

/-- Task `t` is admissible: its device and operation exist, its arguments are in range and
the operation's preconditions hold in the observed state. -/
def TaskOK (cards : List AgentCard) (s : State) (t : Task) : Prop :=
  ∃ op, resolve cards t = some op ∧ ArgsOK op t.args ∧ ∀ c ∈ op.pre, CondHolds s t.args c

/-- Two lists of state variables have no element in common. -/
def Disjoint (a b : List StateVar) : Prop := ∀ x ∈ a, x ∉ b

/-- The allocation specification: every task is admissible and no two tasks of the batch write
the same state variable. -/
def AllocSpec (cards : List AgentCard) (s : State) (b : Batch) : Prop :=
  (∀ t ∈ b, TaskOK cards s t) ∧
    List.Pairwise (fun t u ↦ Disjoint (writesOf cards t) (writesOf cards u)) b

/-! ## Checker -/

/-- Executable version of `ArgsOK`. -/
def argsOK (op : Operation) (args : List (String × Int)) : Bool :=
  op.params.all fun p ↦
    match argVal args p.name with
    | some v => decide (p.lo ≤ v) && decide (v ≤ p.hi)
    | none => false

/-- Executable version of `CondHolds`. -/
def condHolds (s : State) (args : List (String × Int)) : Cond → Bool
  | .eqConst var c => decide (stateVal s var = some c)
  | .eqParam var p =>
    match argVal args p with
    | some v => decide (stateVal s var = some v)
    | none => false

/-- Executable version of `TaskOK`. -/
def taskOK (cards : List AgentCard) (s : State) (t : Task) : Bool :=
  match resolve cards t with
  | some op => argsOK op t.args && op.pre.all (condHolds s t.args)
  | none => false

/-- Executable version of `Disjoint`. -/
def disjoint (a b : List StateVar) : Bool :=
  a.all fun x ↦ decide (x ∉ b)

/-- No two tasks of the batch write the same state variable. -/
def conflictFree (cards : List AgentCard) : Batch → Bool
  | [] => true
  | t :: ts =>
    ts.all (fun u ↦ disjoint (writesOf cards t) (writesOf cards u)) && conflictFree cards ts

/-- The allocation checker: accepts a batch iff every task is admissible and the batch is
conflict-free. -/
def check (cards : List AgentCard) (s : State) (b : Batch) : Bool :=
  b.all (taskOK cards s) && conflictFree cards b

/-! ## Soundness -/

theorem argsOK_sound {op : Operation} {args : List (String × Int)} (h : argsOK op args = true) :
    ArgsOK op args := by
  intro p hp
  have hp' := List.all_eq_true.mp h p hp
  cases hv : argVal args p.name with
  | none => simp [hv] at hp'
  | some v =>
    simp only [hv, Bool.and_eq_true, decide_eq_true_eq] at hp'
    exact ⟨v, rfl, hp'.1, hp'.2⟩

theorem condHolds_sound {s : State} {args : List (String × Int)} {c : Cond}
    (h : condHolds s args c = true) : CondHolds s args c := by
  cases c with
  | eqConst var k =>
    simp only [condHolds, decide_eq_true_eq] at h
    simp only [CondHolds]
    exact h
  | eqParam var p =>
    simp only [CondHolds]
    cases hv : argVal args p with
    | none => simp [condHolds, hv] at h
    | some v =>
      simp only [condHolds, hv, decide_eq_true_eq] at h
      exact ⟨v, rfl, h⟩

theorem taskOK_sound {cards : List AgentCard} {s : State} {t : Task}
    (h : taskOK cards s t = true) : TaskOK cards s t := by
  unfold TaskOK
  cases hr : resolve cards t with
  | none => simp [taskOK, hr] at h
  | some op =>
    simp only [taskOK, hr, Bool.and_eq_true] at h
    exact ⟨op, rfl, argsOK_sound h.1, fun c hc ↦ condHolds_sound (List.all_eq_true.mp h.2 c hc)⟩

theorem disjoint_sound {a b : List StateVar} (h : disjoint a b = true) : Disjoint a b := by
  intro x hx
  exact of_decide_eq_true (List.all_eq_true.mp h x hx)

theorem conflictFree_sound {cards : List AgentCard} : ∀ {b : Batch},
    conflictFree cards b = true →
      List.Pairwise (fun t u ↦ Disjoint (writesOf cards t) (writesOf cards u)) b
  | [], _ => List.Pairwise.nil
  | t :: ts, h => by
    simp only [conflictFree, Bool.and_eq_true] at h
    exact List.Pairwise.cons (fun u hu ↦ disjoint_sound (List.all_eq_true.mp h.1 u hu))
      (conflictFree_sound h.2)

/-- **Soundness of the allocation checker**: every batch accepted by `check` satisfies the
allocation specification. -/
theorem check_sound {cards : List AgentCard} {s : State} {b : Batch}
    (h : check cards s b = true) : AllocSpec cards s b := by
  unfold AllocSpec
  simp only [check, Bool.and_eq_true] at h
  exact ⟨fun t ht ↦ taskOK_sound (List.all_eq_true.mp h.1 t ht), conflictFree_sound h.2⟩

end SmartHome
