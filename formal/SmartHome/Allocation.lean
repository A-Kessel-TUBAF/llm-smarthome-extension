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
2. every parameter of the operation is supplied, with a value of its type (integers: inside
   their range),
3. every state variable the operation writes is determined, i.e. its room is known from the
   card or from the task's arguments,
4. the operation's preconditions hold in the observed state, and
5. no two tasks of the batch write the same state variable.

Cards refer to state variables of their own device only (see `resolveVar`), so a task can never
write another device's state.
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
def argVal (args : List (String × Value)) (p : String) : Option Value :=
  (args.find? (fun a ↦ a.1 == p)).map (·.2)

/-- The value of state variable `v` in state `s`, if known. -/
def stateVal (s : State) (v : StateVar) : Option Value :=
  (s.find? (fun a ↦ a.1 == v)).map (·.2)

/-- The operation a task refers to, if its device and operation exist. -/
def resolve (cards : List AgentCard) (t : Task) : Option Operation :=
  (findCard cards t.device).bind (fun c ↦ findOp c t.op)

/-- The room that reference `r` denotes under arguments `args`: `some none` for the whole home,
`some (some room)` for a room, `none` if a room argument is missing or not a `RoomID`. -/
def resolveRoom (args : List (String × Value)) : RoomRef → Option (Option RoomId)
  | .home => some none
  | .fixed r => some (some r)
  | .param p =>
    match argVal args p with
    | some (.id k r) => if k = "RoomID" then some (some r) else none
    | _ => none

/-- The state variable of device `d` that the card reference `v` denotes under arguments `args`,
if its room can be determined. -/
def resolveVar (d : DeviceId) (args : List (String × Value)) (v : VarRef) : Option StateVar :=
  (resolveRoom args v.room).map fun r ↦ { device := d, room := r, prop := v.prop }

/-- The state variables written by task `t` (empty if the task cannot be resolved). -/
def writesOf (cards : List AgentCard) (t : Task) : List StateVar :=
  match resolve cards t with
  | some op => op.writes.filterMap (resolveVar t.device t.args)
  | none => []

/-! ## Specification -/

/-- Value `v` has parameter type `t`: a boolean for `bool`, an integer inside `[lo, hi]` for
`int lo hi`, and an identifier of the same kind for `id kind`. -/
def HasType : Value → ParamType → Prop
  | .bool _, .bool => True
  | .int n, .int lo hi => lo ≤ n ∧ n ≤ hi
  | .id k _, .id k' => k = k'
  | _, _ => False

/-- Every parameter of `op` is supplied by `args` with a value of its type. -/
def ArgsOK (op : Operation) (args : List (String × Value)) : Prop :=
  ∀ p ∈ op.params, ∃ v, argVal args p.name = some v ∧ HasType v p.ty

/-- Every state variable that `op` writes is determined for device `d` under arguments `args`. -/
def WritesResolve (d : DeviceId) (args : List (String × Value)) (op : Operation) : Prop :=
  ∀ w ∈ op.writes, ∃ x, resolveVar d args w = some x

/-- Condition `c` holds in state `s`, for device `d` under arguments `args`. -/
def CondHolds (s : State) (d : DeviceId) (args : List (String × Value)) : Cond → Prop
  | .eqConst var c => ∃ x, resolveVar d args var = some x ∧ stateVal s x = some c
  | .eqParam var p =>
    ∃ x v, resolveVar d args var = some x ∧ argVal args p = some v ∧ stateVal s x = some v

/-- Task `t` is admissible: its device and operation exist, its arguments are well-typed, the
variables it writes are determined, and the operation's preconditions hold in the observed
state. -/
def TaskOK (cards : List AgentCard) (s : State) (t : Task) : Prop :=
  ∃ op, resolve cards t = some op ∧ ArgsOK op t.args ∧ WritesResolve t.device t.args op ∧
    ∀ c ∈ op.pre, CondHolds s t.device t.args c

/-- Two lists of state variables have no element in common. -/
def Disjoint (a b : List StateVar) : Prop := ∀ x ∈ a, x ∉ b

/-- The allocation specification: every task is admissible and no two tasks of the batch write
the same state variable. -/
def AllocSpec (cards : List AgentCard) (s : State) (b : Batch) : Prop :=
  (∀ t ∈ b, TaskOK cards s t) ∧
    List.Pairwise (fun t u ↦ Disjoint (writesOf cards t) (writesOf cards u)) b

/-! ## Checker -/

/-- Executable version of `HasType`. -/
def hasType : Value → ParamType → Bool
  | .bool _, .bool => true
  | .int n, .int lo hi => decide (lo ≤ n) && decide (n ≤ hi)
  | .id k _, .id k' => decide (k = k')
  | _, _ => false

/-- Executable version of `ArgsOK`. -/
def argsOK (op : Operation) (args : List (String × Value)) : Bool :=
  op.params.all fun p ↦
    match argVal args p.name with
    | some v => hasType v p.ty
    | none => false

/-- Executable version of `WritesResolve`. -/
def writesResolve (d : DeviceId) (args : List (String × Value)) (op : Operation) : Bool :=
  op.writes.all fun w ↦ (resolveVar d args w).isSome

/-- Executable version of `CondHolds`. -/
def condHolds (s : State) (d : DeviceId) (args : List (String × Value)) : Cond → Bool
  | .eqConst var c =>
    match resolveVar d args var with
    | some x => decide (stateVal s x = some c)
    | none => false
  | .eqParam var p =>
    match resolveVar d args var, argVal args p with
    | some x, some v => decide (stateVal s x = some v)
    | _, _ => false

/-- Executable version of `TaskOK`. -/
def taskOK (cards : List AgentCard) (s : State) (t : Task) : Bool :=
  match resolve cards t with
  | some op =>
    argsOK op t.args && writesResolve t.device t.args op &&
      op.pre.all (condHolds s t.device t.args)
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

theorem hasType_sound {v : Value} {t : ParamType} (h : hasType v t = true) : HasType v t := by
  cases v <;> cases t <;> simp_all [hasType, HasType]

theorem argsOK_sound {op : Operation} {args : List (String × Value)}
    (h : argsOK op args = true) : ArgsOK op args := by
  intro p hp
  have hp' := List.all_eq_true.mp h p hp
  cases hv : argVal args p.name with
  | none => simp [hv] at hp'
  | some v =>
    simp only [hv] at hp'
    exact ⟨v, rfl, hasType_sound hp'⟩

theorem writesResolve_sound {d : DeviceId} {args : List (String × Value)} {op : Operation}
    (h : writesResolve d args op = true) : WritesResolve d args op := by
  intro w hw
  exact Option.isSome_iff_exists.mp (List.all_eq_true.mp h w hw)

theorem condHolds_sound {s : State} {d : DeviceId} {args : List (String × Value)} {c : Cond}
    (h : condHolds s d args c = true) : CondHolds s d args c := by
  cases c with
  | eqConst var k =>
    cases hx : resolveVar d args var with
    | none => simp [condHolds, hx] at h
    | some x =>
      simp only [condHolds, hx, decide_eq_true_eq] at h
      exact ⟨x, hx, h⟩
  | eqParam var p =>
    cases hx : resolveVar d args var with
    | none => simp [condHolds, hx] at h
    | some x =>
      cases hv : argVal args p with
      | none => simp [condHolds, hx, hv] at h
      | some v =>
        simp only [condHolds, hx, hv, decide_eq_true_eq] at h
        exact ⟨x, v, hx, hv, h⟩

theorem taskOK_sound {cards : List AgentCard} {s : State} {t : Task}
    (h : taskOK cards s t = true) : TaskOK cards s t := by
  unfold TaskOK
  cases hr : resolve cards t with
  | none => simp [taskOK, hr] at h
  | some op =>
    simp only [taskOK, hr, Bool.and_eq_true] at h
    exact ⟨op, rfl, argsOK_sound h.1.1, writesResolve_sound h.1.2,
      fun c hc ↦ condHolds_sound (List.all_eq_true.mp h.2 c hc)⟩

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
