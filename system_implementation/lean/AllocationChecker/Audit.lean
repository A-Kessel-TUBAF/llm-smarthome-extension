/-
  Axiom audit. Building this module prints the axiom closure of the audited
  declarations and fails the build if any axiom outside the permitted
  built-ins appears (for example `sorryAx` or `Lean.ofReduceBool`).

  Run:  lake build AllocationChecker.Audit
   or:  lake env lean AllocationChecker/Audit.lean
-/
import AllocationChecker
import AllocationChecker.TypedValues
import Lean

open Lean Elab Command

namespace AllocationChecker.Audit

/-- Lean's standard built-in axioms. Nothing else is permitted. -/
def permittedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

/-- `#audit_axioms n₁ n₂ ...` logs `AXIOM-AUDIT <name>: [axioms]` for each
    declaration and throws an error on a non-permitted axiom. -/
elab "#audit_axioms " ids:ident* : command => do
  for id in ids do
    let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
    let axs ← collectAxioms name
    let sorted := axs.qsort (fun a b => a.toString < b.toString)
    logInfo m!"AXIOM-AUDIT {name}: {sorted.toList}"
    let bad := sorted.filter (fun a => !permittedAxioms.contains a)
    unless bad.isEmpty do
      throwError "AXIOM-AUDIT FAILED {name}: non-permitted axioms {bad.toList}"

end AllocationChecker.Audit

#print axioms AllocationChecker.checkAllocation_sound

#audit_axioms
  AllocationChecker.checkAllocation_sound
  AllocationChecker.checkAllocation_complete
  AllocationChecker.checkAllocation_iff
  AllocationChecker.not_valid_of_rejected
  AllocationChecker.GraphWF.acyclic
  AllocationChecker.checkAllocation_acyclic
  AllocationChecker.Examples.twoIndependent_accepted
  AllocationChecker.Examples.twoIndependent_valid
  AllocationChecker.Examples.withDependent_valid
  AllocationChecker.Examples.conflicting_invalid
  AllocationChecker.Examples.stale_invalid
  AllocationChecker.Examples.expired_invalid
  AllocationChecker.Examples.wrongInput_rejected
  AllocationChecker.Examples.overCapacity_rejected
  AllocationChecker.Examples.running_rejected
  AllocationChecker.Examples.overBudget_rejected
  AllocationChecker.Examples.activeConflict_rejected
  AllocationChecker.Examples.outOfBounds_rejected
  AllocationChecker.Examples.duplicate_rejected
  AllocationChecker.Examples.staleRevision_rejected
  AllocationChecker.Examples.wrongDevice_rejected
  AllocationChecker.Examples.preAlone_accepted
  AllocationChecker.Examples.preDisturbed_invalid
  AllocationChecker.Examples.preRunning_rejected
  AllocationChecker.Examples.opPreRunning_rejected
  AllocationChecker.TypedValues.decodeTyped_iff
  AllocationChecker.TypedValues.decodeTyped_wrong_type
  AllocationChecker.TypedValues.decodeTypedArgs_iff
  AllocationChecker.admitted_order_irrelevant
  AllocationChecker.admitted_reads_stable
  AllocationChecker.preconditionsUndisturbed_of_valid
  AllocationChecker.admitted_preconditions_hold
  AllocationChecker.admitted_failures_isolated
  AllocationChecker.validated_iff_predicted
  AllocationChecker.admitted_successful_validated
  AllocationChecker.repair_sound
  AllocationChecker.repair_no_failures
  AllocationChecker.reachable_within
  AllocationChecker.ExecutionExamples.home_within
  AllocationChecker.ExecutionExamples.withDependent_order_irrelevant
  AllocationChecker.ExecutionExamples.withDependent_preconditions_hold
  AllocationChecker.ExecutionExamples.withDependent_dim_validated
