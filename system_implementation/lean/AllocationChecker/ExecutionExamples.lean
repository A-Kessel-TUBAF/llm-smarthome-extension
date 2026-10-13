/-
  The execution theorems applied to the examples of `Examples.lean`, with a concrete device
  behaviour: a command sets every key it writes to its first argument. This shows that the
  assumption `Respects` is satisfiable, and how the theorems are used.
-/
import AllocationChecker.Repair
import AllocationChecker.Invariants
import AllocationChecker.Examples

namespace AllocationChecker.ExecutionExamples

open AllocationChecker AllocationChecker.Examples

/-- A simple device behaviour: every key the command writes takes its first argument. -/
def setToFirstArg (cat : Catalogue) : Behaviour := fun c v k =>
  if k ∈ (footprint cat c).writes then some (c.args.headD 0) else v k

theorem setToFirstArg_respects (cat : Catalogue) : Respects cat (setToFirstArg cat) where
  frame _ _ _ hk := by simp [setToFirstArg, hk]
  onlyReads _ _ _ _ _ hk := by simp [setToFirstArg, hk]

/-- No command of the example batch writes a key another one's preconditions are about. -/
theorem withDependent_undisturbed :
    PreconditionsUndisturbed cat graph withDependent.commands := by decide

/-- Dimming the light, setting the volume and heating give the same result in every order. -/
theorem withDependent_order_irrelevant {order : List Command}
    (hp : order.Perm withDependent.commands) :
    runSeq (setToFirstArg cat) snap.valuation order =
      runPar cat (setToFirstArg cat) snap.valuation withDependent.commands :=
  admitted_order_irrelevant withDependent_valid (setToFirstArg_respects cat) hp _

/-- In every order, each of the three commands starts with its preconditions met. -/
theorem withDependent_preconditions_hold {order : List Command}
    (hp : order.Perm withDependent.commands) :
    PreconditionsAtStart cat graph (setToFirstArg cat) snap.valuation order :=
  admitted_preconditions_hold withDependent_valid (setToFirstArg_respects cat)
    withDependent_undisturbed hp

/-- The heating command fails and does nothing; the others behave as `setToFirstArg`. -/
def heatFails : Behaviour := fun c v => if c = cHeat then v else setToFirstArg cat c v

theorem heatFails_act : Act cat (setToFirstArg cat) (· = cHeat) heatFails where
  frame c v k hk := by unfold heatFails; split <;> simp [setToFirstArg, hk]
  ok c hc := by funext v; simp [heatFails, hc]

/-- Even though heating fails, dimming passes observed-state validation, in every order. -/
theorem withDependent_dim_validated {order : List Command}
    (hp : order.Perm withDependent.commands) :
    Validated cat (setToFirstArg cat) snap.valuation (runSeq heatFails snap.valuation order)
      cDim :=
  admitted_successful_validated withDependent_valid (setToFirstArg_respects cat) heatFails_act
    hp _ (by decide) (by decide)

/-! ### Local repair over two rounds -/

/-- The declared task dependencies of the example graph (the heating task 1 depends on the
search task 0). -/
def exDeps (t : Nat) : List Nat := ((lookupTask graph t).map (·.deps)).getD []

/-- The search command of task 0. -/
def cSearch : Command := cmd 0 dSearch 3 [8]

/-- The volume change fails: only it has to be replanned; the heating in the next round still
runs. -/
example : (runPlan cat exDeps (· == cVol) (setToFirstArg cat) (RunState.start snap.valuation)
    [[cDim, cVol], [cHeat]]).bad = [cVol.task] := by decide

/-- The search fails: the heating depends on its answer and is skipped, the volume change in
the same round still runs. -/
example : (runPlan cat exDeps (· == cSearch) (setToFirstArg cat) (RunState.start snap.valuation)
    [[cSearch, cDim], [cHeat, cVol]]).bad = [cSearch.task, cHeat.task] := by decide

/-! ### Safety limits over time -/

/-- A device behaviour where every key a command writes takes its first argument (if any). -/
def setToHead (cat : Catalogue) : Behaviour := fun c v k =>
  if k ∈ (footprint cat c).writes then c.args.head? else v k

theorem setToHead_respects (cat : Catalogue) : Respects cat (setToHead cat) where
  frame _ _ _ hk := by simp [setToHead, hk]
  onlyReads _ _ _ _ _ hk := by simp [setToHead, hk]

/-- `setToHead` follows assignments from the first argument of operations that write the
assigned attribute. -/
theorem setToHead_follows (cat : Catalogue) (assigns : List Assign)
    (h : ∀ a ∈ assigns, a.arg = 0 ∧ (lookupOp cat a.op).any (a.attr ∈ ·.writes) = true) :
    Follows assigns (setToHead cat) := by
  intro a ha c v hc
  obtain ⟨h0, hw⟩ := h a ha
  cases hl : lookupOp cat a.op with
  | none => simp [hl] at hw
  | some o =>
    simp only [hl, Option.any_some, decide_eq_true_eq] at hw
    have hfp : footprint cat c = opFootprint cat o c.device := by
      simp [footprint, cmdFootprint?, hc, hl]
    have hk : (⟨c.device, a.attr⟩ : StateKey) ∈ (footprint cat c).writes := by
      rw [hfp]
      exact List.mem_append_left _ (List.mem_map_of_mem hw)
    simp only [setToHead, hk, if_true, h0]
    cases c.args <;> rfl

/-- Brightness and volume within 0–100, thermostat setpoint within 10–30 °C. -/
def exLimits : List Limit :=
  [⟨aBrightness, 0, 100⟩, ⟨aSetpoint, 10, 30⟩, ⟨aVolume, 0, 100⟩]

/-- Every operation sets the attribute it writes to its first argument. -/
def exAssigns : List Assign :=
  [⟨0, aBrightness, 0⟩, ⟨1, aSetpoint, 0⟩, ⟨2, aVolume, 0⟩, ⟨3, aResult, 0⟩]

theorem exLimits_respected : RespectsLimits cat exAssigns exLimits := by decide

theorem snap_within : Within exLimits snap.valuation := within_of_entries (by decide)

/-- Starting from the example snapshot, brightness, setpoint and volume stay within their
limits in every state reachable by running admitted batches, whatever the planner proposes. -/
theorem home_within {v : Valuation} (h : Reachable cat (setToHead cat) snap.valuation v) :
    Within exLimits v :=
  reachable_within (setToHead_respects cat) (setToHead_follows cat exAssigns (by decide))
    exLimits_respected snap_within h

/-- A stricter household rule, setpoint within 15–25 °C, is not guaranteed by the catalogue:
the thermostat operation accepts 10–30 °C. -/
example : ¬ RespectsLimits cat exAssigns [⟨aSetpoint, 15, 25⟩] := by decide

end AllocationChecker.ExecutionExamples
