/-
  Partial failures and observed-state validation.

  Devices can fail. A failed command may leave any values on the keys it writes, but changes
  nothing else (`Act`). For a batch admitted by the checker:

  * `admitted_failures_isolated`: in every execution order, every key that no failed command
    writes ends with its predicted value (the value without failures). So every command that
    succeeded achieves exactly its predicted effect, whatever failed around it.

  Observed-state validation checks each command on its own: the observed values of the keys it
  writes must be those it was predicted to write, computed from its own reads (`Validated`).

  * `runPar_writer` / `validated_iff_predicted`: a command's own prediction is the batch's
    prediction on its keys, so checking every command on its own is the same as checking the
    whole batch;
  * `admitted_successful_validated`: a command that succeeded always passes validation, whatever
    else failed. A failed command is reported exactly when it left a value other than predicted.
-/
import AllocationChecker.Execution

namespace AllocationChecker

/-- The actual behaviour `act`, where the commands in `failed` may fail: every command changes
only the keys it writes, and the commands that did not fail behave as `run`. -/
structure Act (cat : Catalogue) (run : Behaviour) (failed : Command → Prop) (act : Behaviour) :
    Prop where
  frame : ∀ c v k, k ∉ (footprint cat c).writes → act c v k = v k
  ok : ∀ c, ¬ failed c → act c = run c

/-- In a pairwise list, any two different elements are related (for a symmetric relation). -/
theorem pairwise_of_ne {α : Type} {R : α → α → Prop} (hs : ∀ {a b}, R a b → R b a) :
    ∀ {l : List α}, l.Pairwise R → ∀ {a b}, a ∈ l → b ∈ l → a ≠ b → R a b
  | [], _, _, _, ha, _, _ => absurd ha List.not_mem_nil
  | x :: l, h, a, b, ha, hb, hne => by
    rw [List.pairwise_cons] at h
    rcases List.mem_cons.mp ha with rfl | ha'
    · rcases List.mem_cons.mp hb with rfl | hb'
      · exact absurd rfl hne
      · exact h.1 b hb'
    · rcases List.mem_cons.mp hb with rfl | hb'
      · exact hs (h.1 a ha')
      · exact pairwise_of_ne hs h.2 ha' hb' hne

/-- The value the batch writes on a key is that of the command writing it. -/
theorem runPar_writer {cat : Catalogue} {run : Behaviour} {v : Valuation} :
    ∀ {cs : List Command}, cs.Pairwise (IndepCmd cat) → ∀ {c : Command}, c ∈ cs →
      ∀ {k : StateKey}, k ∈ (footprint cat c).writes → runPar cat run v cs k = run c v k
  | [], _, _, hc, _, _ => absurd hc List.not_mem_nil
  | d :: ds, hi, c, hc, k, hk => by
    rw [List.pairwise_cons] at hi
    by_cases hd : k ∈ (footprint cat d).writes
    · have hcd : c = d := by
        rcases List.mem_cons.mp hc with h | h
        · exact h
        · exact absurd (List.mem_append_right _ hk) ((hi.1 c h).1 k hd)
      subst hcd
      simp [runPar, hd]
    · have hc' : c ∈ ds := by
        rcases List.mem_cons.mp hc with h | h
        · exact absurd (h ▸ hk) hd
        · exact h
      have := runPar_writer (run := run) (v := v) hi.2 hc' hk
      simp only [runPar, List.find?_cons, hd, decide_false] at this ⊢
      exact this

section
variable {cat : Catalogue} {run : Behaviour} {failed : Command → Prop} {act : Behaviour}
  (hr : Respects cat run) (ha : Act cat run failed act)
include hr ha

theorem runSeq_with_failures : ∀ {cs : List Command} (v : Valuation) (k : StateKey),
    cs.Pairwise (IndepCmd cat) → (∀ c ∈ cs, failed c → k ∉ (footprint cat c).writes) →
      runSeq act v cs k = runPar cat run v cs k
  | [], v, k, _, _ => by simp [runSeq, runPar]
  | c :: cs, v, k, hi, hk => by
    rw [List.pairwise_cons] at hi
    simp only [runSeq]
    rw [runSeq_with_failures (act c v) k hi.2 (fun u hu => hk u (by simp [hu]))]
    by_cases hw : k ∈ (footprint cat c).writes
    · have hok : ¬ failed c := fun hf => hk c (by simp) hf hw
      have hnone : cs.find? (fun c' => decide (k ∈ (footprint cat c').writes)) = none :=
        List.find?_eq_none.mpr fun c' hc' hw' =>
          (hi.1 c' hc').1 k hw (List.mem_append_right _ (by simpa using hw'))
      simp [runPar, hnone, hw, ha.ok c hok]
    · cases hf : cs.find? (fun c' => decide (k ∈ (footprint cat c').writes)) with
      | none => simp [runPar, hf, hw, ha.frame c v k hw]
      | some c' =>
        have hc' := List.mem_of_find?_eq_some hf
        have hw' : k ∈ (footprint cat c').writes := by simpa using List.find?_some hf
        simp only [runPar, List.find?_cons, hf, hw, decide_false]
        exact hr.onlyReads c' (act c v) v
          (fun k' hk' => ha.frame c v k' (fun hw'' => (hi.1 c' hc').1 k' hw''
            (List.mem_append_left _ hk'))) k hw'

end

theorem runPar_perm {cat : Catalogue} {run : Behaviour} (hr : Respects cat run)
    {l l' : List Command} (hp : l.Perm l') (hi : l.Pairwise (IndepCmd cat)) (v : Valuation) :
    runPar cat run v l = runPar cat run v l' := by
  rw [← runSeq_eq_runPar hr v hi, ← runSeq_eq_runPar hr v (hi.perm hp IndepCmd.symm),
    runSeq_perm hr hp hi v]

/-- **Failures stay contained.** For an admitted batch, in every execution order, every key that
no failed command writes ends with its predicted value. -/
theorem admitted_failures_isolated {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} {run : Behaviour} {failed : Command → Prop} {act : Behaviour}
    (hv : AllocationValid s cat g b) (hr : Respects cat run) (ha : Act cat run failed act)
    {order : List Command} (hp : order.Perm b.commands) (v : Valuation) {k : StateKey}
    (hk : ∀ c ∈ b.commands, failed c → k ∉ (footprint cat c).writes) :
    runSeq act v order k = runPar cat run v b.commands k := by
  have hi := (pairwise_of_valid hv).perm hp.symm IndepCmd.symm
  rw [runSeq_with_failures hr ha v k hi (fun c hc => hk c (hp.mem_iff.mp hc)),
    runPar_perm hr hp hi v]

/-! ### Observed-state validation -/

/-- Command `c` passes observed-state validation in the observed valuation `o`: the keys it
writes have the values it was predicted to write, computed from `v`. -/
def Validated (cat : Catalogue) (run : Behaviour) (v o : Valuation) (c : Command) : Prop :=
  ∀ k ∈ (footprint cat c).writes, o k = run c v k

/-- **Checking each command on its own is enough.** For an admitted batch, every command passes
validation exactly when the observed valuation has the batch's predicted values on all keys the
batch writes. -/
theorem validated_iff_predicted {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} {run : Behaviour} (hv : AllocationValid s cat g b) (v o : Valuation) :
    (∀ c ∈ b.commands, Validated cat run v o c) ↔
      ∀ c ∈ b.commands, ∀ k ∈ (footprint cat c).writes,
        o k = runPar cat run v b.commands k := by
  have hi := pairwise_of_valid hv
  constructor
  · intro h c hc k hk
    rw [runPar_writer hi hc hk]
    exact h c hc k hk
  · intro h c hc k hk
    rw [← runPar_writer (run := run) (v := v) hi hc hk]
    exact h c hc k hk

/-- **No false alarms.** For an admitted batch, in every execution order, a command that did not
fail passes validation in the resulting valuation, whatever else failed. -/
theorem admitted_successful_validated {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} {run : Behaviour} {failed : Command → Prop} {act : Behaviour}
    (hv : AllocationValid s cat g b) (hr : Respects cat run) (ha : Act cat run failed act)
    {order : List Command} (hp : order.Perm b.commands) (v : Valuation) {c : Command}
    (hc : c ∈ b.commands) (hok : ¬ failed c) : Validated cat run v (runSeq act v order) c := by
  have hi := pairwise_of_valid hv
  intro k hk
  rw [admitted_failures_isolated hv hr ha hp v ?_, runPar_writer hi hc hk]
  intro c' hc' hf hk'
  have hne : c' ≠ c := fun h => hok (h ▸ hf)
  exact (pairwise_of_ne IndepCmd.symm hi hc' hc hne).1 k hk' (List.mem_append_right _ hk)

end AllocationChecker
