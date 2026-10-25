/-
  Protected target statements (QED_LEAF). The proofs are deliberately `sorry`:
  this file fixes WHAT must be proved; the candidate supplies the proofs.
  `DependsOn` is restated verbatim from Soundness.lean because the acyclicity
  statement depends on it.
-/
import AllocationChecker.Spec
import AllocationChecker.Check

namespace AllocationChecker

theorem checkAllocation_sound
    (snapshot : Snapshot)
    (catalogue : Catalogue)
    (graph : TaskGraph)
    (batch : AllocationBatch) :
    checkAllocation snapshot catalogue graph batch = true →
      AllocationValid snapshot catalogue graph batch := sorry

theorem checkAllocation_complete
    (snapshot : Snapshot)
    (catalogue : Catalogue)
    (graph : TaskGraph)
    (batch : AllocationBatch) :
    AllocationValid snapshot catalogue graph batch →
      checkAllocation snapshot catalogue graph batch = true := sorry

/-- `DependsOn g a b`: the task with identifier `a` transitively depends on `b`. -/
inductive DependsOn (g : TaskGraph) : Nat → Nat → Prop
  | edge {t : Task} {b : Nat} : t ∈ g.tasks → b ∈ t.deps → DependsOn g t.id b
  | trans {a b c : Nat} : DependsOn g a b → DependsOn g b c → DependsOn g a c

theorem GraphWF.acyclic {g : TaskGraph} (hwf : GraphWF g) {a : Nat} : ¬ DependsOn g a a := sorry

end AllocationChecker
