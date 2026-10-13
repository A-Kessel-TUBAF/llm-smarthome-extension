/-
  The execution theorems applied to the examples of `Examples.lean`, with a concrete device
  behaviour: a command sets every key it writes to its first argument. This shows that the
  assumption `Respects` is satisfiable, and how the theorems are used.
-/
import AllocationChecker.Repair
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

end AllocationChecker.ExecutionExamples
