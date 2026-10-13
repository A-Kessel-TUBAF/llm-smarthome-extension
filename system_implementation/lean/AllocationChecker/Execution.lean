/-
  What happens while an admitted batch runs.

  The specification (`Spec.lean`) says when a batch may be admitted; it does not describe
  execution. This module adds a minimal execution model and proves that admission is enough for
  safe parallel execution.

  * A `Valuation` gives every state key its current value (`none`: unknown).
  * A `Behaviour` says how the devices react: the valuation after running a command. Nothing is
    assumed about the values written, only that the behaviour respects the catalogue's
    footprints (`Respects`): a command changes only the keys it writes, and what it writes
    depends only on the keys it reads.

  For a batch admitted by the checker (`AllocationValid`):

  * `admitted_order_irrelevant`: every execution order gives the same final valuation, namely
    that of running all commands at once (`runPar`);
  * `admitted_reads_stable`: in every order, each command starts with the initial values of the
    keys it reads;
  * `admitted_preconditions_hold`: if no command writes a key another command's preconditions
    are about (`PreconditionsUndisturbed`, decidable; it follows from preconditions on read keys),
    every command's preconditions still hold when it starts.

  These are Bernstein's conditions for parallel execution, proved for the checker's
  independence relation `Indep`. This module imports only `Init` (through the checker).
-/
import AllocationChecker.Completeness

namespace AllocationChecker

/-! ### Execution model -/

/-- The value of every state key (`none`: unknown). -/
abbrev Valuation := StateKey → Option Int

/-- The valuation observed in a snapshot. -/
def Snapshot.valuation (s : Snapshot) : Valuation := fun k => (lookupEntry s k).map (·.val)

/-- How the devices react: the valuation after running a command. -/
abbrev Behaviour := Command → Valuation → Valuation

/-- The footprint of command `c` (empty if its operation is unknown). -/
def footprint (cat : Catalogue) (c : Command) : Footprint :=
  (cmdFootprint? cat c).getD ⟨[], [], []⟩

/-- The behaviour respects the catalogue's footprints. -/
structure Respects (cat : Catalogue) (run : Behaviour) : Prop where
  /-- A command changes only the keys it writes. -/
  frame : ∀ c v k, k ∉ (footprint cat c).writes → run c v k = v k
  /-- What a command writes depends only on the keys it reads. -/
  onlyReads : ∀ c v v', (∀ k ∈ (footprint cat c).reads, v k = v' k) →
    ∀ k ∈ (footprint cat c).writes, run c v k = run c v' k

/-- Run commands one after another. -/
def runSeq (run : Behaviour) (v : Valuation) : List Command → Valuation
  | [] => v
  | c :: cs => runSeq run (run c v) cs

/-- Run commands at once: every key takes the value written by the command that writes it
(the first one, if several do), computed in the initial valuation. -/
def runPar (cat : Catalogue) (run : Behaviour) (v : Valuation) (cs : List Command) :
    Valuation := fun k =>
  match cs.find? (fun c => decide (k ∈ (footprint cat c).writes)) with
  | some c => run c v k
  | none => v k

/-- Two commands are independent: Bernstein's conditions on their footprints. -/
def IndepCmd (cat : Catalogue) (a b : Command) : Prop :=
  Indep (footprint cat a) (footprint cat b)

/-! ### Independence -/

theorem Indep.symm {f g : Footprint} (h : Indep f g) : Indep g f := by
  obtain ⟨h₁, h₂, h₃⟩ := h
  refine ⟨fun k hk hk' => ?_, fun k hk hk' => h₁ k hk (List.mem_append_left _ hk'),
    fun k hk hk' => h₃ k hk' hk⟩
  rcases List.mem_append.mp hk' with hr | hw
  · exact h₂ k hk hr
  · exact h₁ k hw (List.mem_append_right _ hk)

theorem IndepCmd.symm {cat : Catalogue} {a b : Command} (h : IndepCmd cat a b) :
    IndepCmd cat b a :=
  Indep.symm h

/-- Every command of an admitted batch has the catalogue's footprint. -/
theorem footprint_of_valid {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} (hv : AllocationValid s cat g b) {c : Command} (hc : c ∈ b.commands) :
    CmdFootprint cat c (footprint cat c) := by
  obtain ⟨_, o, _, _, ho, _, _⟩ := hv.commands c hc
  have hl := lookupOp_complete hv.catalogueWF ho
  refine ⟨o, ho, ?_⟩
  simp [footprint, cmdFootprint?, hl]

/-- The commands of an admitted batch are pairwise independent. -/
theorem pairwise_of_valid {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} (hv : AllocationValid s cat g b) :
    b.commands.Pairwise (IndepCmd cat) :=
  hv.independent.1.imp_of_mem fun ha hb h =>
    h _ _ (footprint_of_valid hv ha) (footprint_of_valid hv hb)

/-! ### Order independence -/

section
variable {cat : Catalogue} {run : Behaviour} (hr : Respects cat run)
include hr

/-- Two independent commands can be swapped. -/
theorem run_swap {a b : Command} (h : IndepCmd cat a b) (v : Valuation) :
    run b (run a v) = run a (run b v) := by
  have h' := IndepCmd.symm h
  funext k
  by_cases ka : k ∈ (footprint cat a).writes
  · have kb : k ∉ (footprint cat b).writes := fun kb => h.1 k ka (List.mem_append_right _ kb)
    rw [hr.frame b _ k kb]
    exact hr.onlyReads a v (run b v)
      (fun k' hk' => (hr.frame b v k' (fun hw => h.2.1 k' hw hk')).symm) k ka
  · by_cases kb : k ∈ (footprint cat b).writes
    · rw [hr.frame a _ k ka]
      exact hr.onlyReads b (run a v) v
        (fun k' hk' => hr.frame a v k' (fun hw => h'.2.1 k' hw hk')) k kb
    · rw [hr.frame b _ k kb, hr.frame a _ k ka, hr.frame a _ k ka, hr.frame b _ k kb]

/-- Pairwise independent commands give the same result in every order. -/
theorem runSeq_perm {l l' : List Command} (hp : l.Perm l') :
    l.Pairwise (IndepCmd cat) → ∀ v, runSeq run v l = runSeq run v l' := by
  induction hp with
  | nil => intro _ _; rfl
  | cons x _ ih => intro hi v; exact ih (List.pairwise_cons.mp hi).2 _
  | swap x y l =>
    intro hi v
    simp only [runSeq]
    rw [run_swap hr ((List.pairwise_cons.mp hi).1 x (by simp)) v]
  | trans h₁₂ _ ih₁₂ ih₂₃ =>
    intro hi v
    rw [ih₁₂ hi v, ih₂₃ (hi.perm h₁₂ IndepCmd.symm) v]

/-- Running pairwise independent commands one after another is the same as running them at
once. -/
theorem runSeq_eq_runPar : ∀ {cs : List Command} (v : Valuation),
    cs.Pairwise (IndepCmd cat) → runSeq run v cs = runPar cat run v cs
  | [], v, _ => by funext k; simp [runSeq, runPar]
  | c :: cs, v, hi => by
    rw [List.pairwise_cons] at hi
    simp only [runSeq]
    rw [runSeq_eq_runPar (run c v) hi.2]
    funext k
    by_cases hk : k ∈ (footprint cat c).writes
    · have hnone : cs.find? (fun c' => decide (k ∈ (footprint cat c').writes)) = none :=
        List.find?_eq_none.mpr fun c' hc' hw =>
          (hi.1 c' hc').1 k hk (List.mem_append_right _ (by simpa using hw))
      simp [runPar, hnone, hk]
    · cases hf : cs.find? (fun c' => decide (k ∈ (footprint cat c').writes)) with
      | none => simp [runPar, hf, hk, hr.frame c v k hk]
      | some c' =>
        have hc' := List.mem_of_find?_eq_some hf
        have hw : k ∈ (footprint cat c').writes := by simpa using List.find?_some hf
        simp only [runPar, List.find?_cons, hf, hk, decide_false]
        exact hr.onlyReads c' (run c v) v
          (fun k' hk' => hr.frame c v k' (fun hw' => (hi.1 c' hc').1 k' hw'
            (List.mem_append_left _ hk'))) k hw

end

/-! ### What each command sees when it starts -/

/-- Running the commands in list order, each one starts with the values of `v₀` on its keys
`keys c`. -/
def Stable (run : Behaviour) (keys : Command → List StateKey) (v₀ : Valuation) :
    Valuation → List Command → Prop
  | _, [] => True
  | v, c :: cs => (∀ k ∈ keys c, v k = v₀ k) ∧ Stable run keys v₀ (run c v) cs

/-- Neither command writes a key of the other. -/
def Undisturbed (cat : Catalogue) (keys : Command → List StateKey) (a b : Command) : Prop :=
  KeysDisjoint (footprint cat a).writes (keys b) ∧ KeysDisjoint (footprint cat b).writes (keys a)

theorem Undisturbed.symm {cat : Catalogue} {keys : Command → List StateKey} {a b : Command}
    (h : Undisturbed cat keys a b) : Undisturbed cat keys b a :=
  ⟨h.2, h.1⟩

/-- If no command writes a key of another, each command starts with the initial values of its
keys, whatever the order. -/
theorem stable_of {cat : Catalogue} {run : Behaviour} (hr : Respects cat run)
    {keys : Command → List StateKey} {v₀ : Valuation} : ∀ {cs : List Command} {v : Valuation},
    cs.Pairwise (Undisturbed cat keys) → (∀ c ∈ cs, ∀ k ∈ keys c, v k = v₀ k) →
      Stable run keys v₀ v cs
  | [], _, _, _ => trivial
  | c :: cs, v, hi, h => by
    rw [List.pairwise_cons] at hi
    refine ⟨h c (by simp), stable_of hr hi.2 fun u hu k hk => ?_⟩
    rw [hr.frame c v k (fun hw => (hi.1 u hu).1 k hw hk)]
    exact h u (by simp [hu]) k hk

/-- Each command starts with the values of `v₀` on the keys it reads. -/
abbrev ReadsStable (cat : Catalogue) (run : Behaviour) (v₀ : Valuation) :
    Valuation → List Command → Prop :=
  Stable run (fun c => (footprint cat c).reads) v₀

/-- Independent commands do not write each other's reads. -/
theorem undisturbed_reads {cat : Catalogue} {a b : Command} (h : IndepCmd cat a b) :
    Undisturbed cat (fun c => (footprint cat c).reads) a b :=
  ⟨fun k hk hk' => h.1 k hk (List.mem_append_left _ hk'), h.2.1⟩

/-- A precondition holds in a valuation. -/
def HoldsIn (v : Valuation) (p : Precondition) : Prop :=
  ∃ x, v p.key = some x ∧ p.cmp.Holds x p.val

/-- The preconditions of command `c`: those of its task contract and of its operation. -/
def preconditions (cat : Catalogue) (g : TaskGraph) (c : Command) : List Precondition :=
  match lookupTask g c.task, lookupOp cat c.op with
  | some t, some o => t.pre ++ o.pre.map (·.at c.device)
  | _, _ => []

/-- The keys the preconditions of command `c` are about. -/
def preKeys (cat : Catalogue) (g : TaskGraph) (c : Command) : List StateKey :=
  (preconditions cat g c).map (·.key)

/-- No command of the batch writes a key that another command's preconditions are about. -/
def PreconditionsUndisturbed (cat : Catalogue) (g : TaskGraph) (cs : List Command) : Prop :=
  cs.Pairwise (Undisturbed cat (preKeys cat g))

/-- Every precondition of the commands is on a key the command reads. This is a simple way to
satisfy `PreconditionsUndisturbed` (`preconditionsUndisturbed_of_read`). -/
def PreconditionsRead (cat : Catalogue) (g : TaskGraph) (cs : List Command) : Prop :=
  ∀ c ∈ cs, ∀ p ∈ preconditions cat g c, p.key ∈ (footprint cat c).reads

instance (cat : Catalogue) (g : TaskGraph) (cs : List Command) :
    Decidable (PreconditionsUndisturbed cat g cs) := by
  unfold PreconditionsUndisturbed Undisturbed KeysDisjoint; infer_instance

instance (cat : Catalogue) (g : TaskGraph) (cs : List Command) :
    Decidable (PreconditionsRead cat g cs) := by
  unfold PreconditionsRead; infer_instance

/-- Running the commands in list order, each one's preconditions hold when it starts. -/
def PreconditionsAtStart (cat : Catalogue) (g : TaskGraph) (run : Behaviour) :
    Valuation → List Command → Prop
  | _, [] => True
  | v, c :: cs =>
    (∀ p ∈ preconditions cat g c, HoldsIn v p) ∧ PreconditionsAtStart cat g run (run c v) cs

theorem preconditionsAtStart_of {cat : Catalogue} {g : TaskGraph} {run : Behaviour}
    {v₀ : Valuation} : ∀ {cs : List Command} {v : Valuation},
    Stable run (preKeys cat g) v₀ v cs → (∀ c ∈ cs, ∀ p ∈ preconditions cat g c, HoldsIn v₀ p) →
    PreconditionsAtStart cat g run v cs
  | [], _, _, _ => trivial
  | c :: cs, v, ⟨hs, hrest⟩, h =>
    ⟨fun p hp => by
        obtain ⟨x, hx, hc⟩ := h c (by simp) p hp
        exact ⟨x, (hs p.key (List.mem_map_of_mem hp)).trans hx, hc⟩,
      preconditionsAtStart_of hrest (fun u hu => h u (by simp [hu]))⟩

/-- In a well-formed snapshot, a precondition that holds in the snapshot holds in its
valuation. -/
theorem holdsIn_of_preHolds {s : Snapshot} {p : Precondition} (hwf : SnapshotWF s)
    (h : PreHolds s p) : HoldsIn s.valuation p := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e.val, by simp [Snapshot.valuation, lookupEntry_complete hwf he], hc⟩

/-- The preconditions of every command of an admitted batch hold in the snapshot. -/
theorem preconditions_of_valid {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} (hv : AllocationValid s cat g b) :
    ∀ c ∈ b.commands, ∀ p ∈ preconditions cat g c, HoldsIn s.valuation p := by
  intro c hc p hp
  obtain ⟨t, o, _, ht, ho, _, hfit⟩ := hv.commands c hc
  simp only [preconditions, lookupTask_complete hv.graphWF ht,
    lookupOp_complete hv.catalogueWF ho, List.mem_append, List.mem_map] at hp
  rcases hp with hp | ⟨q, hq, rfl⟩
  · exact holdsIn_of_preHolds hv.snapshotWF (hfit.taskPre p hp)
  · exact holdsIn_of_preHolds hv.snapshotWF (hfit.opPre q hq)

/-! ### Safety of admitted batches -/

/-- **Execution order does not matter.** For a batch admitted by the checker and a behaviour
that respects the footprints, running the commands in any order gives the same valuation as
running them all at once. -/
theorem admitted_order_irrelevant {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} {run : Behaviour} (hv : AllocationValid s cat g b)
    (hr : Respects cat run) {order : List Command} (hp : order.Perm b.commands)
    (v : Valuation) : runSeq run v order = runPar cat run v b.commands := by
  have hi := pairwise_of_valid hv
  rw [runSeq_perm hr hp (hi.perm hp.symm IndepCmd.symm) v, runSeq_eq_runPar hr v hi]

/-- **No command disturbs another.** In every execution order of an admitted batch, each
command starts with the initial values of the keys it reads. -/
theorem admitted_reads_stable {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} {run : Behaviour} (hv : AllocationValid s cat g b)
    (hr : Respects cat run) {order : List Command} (hp : order.Perm b.commands)
    (v : Valuation) : ReadsStable cat run v v order :=
  stable_of hr (((pairwise_of_valid hv).perm hp.symm IndepCmd.symm).imp undisturbed_reads)
    fun _ _ _ _ => rfl

/-- If the preconditions are on keys the commands read, no other command of an admitted batch
writes them. -/
theorem preconditionsUndisturbed_of_read {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} (hv : AllocationValid s cat g b)
    (hpr : PreconditionsRead cat g b.commands) : PreconditionsUndisturbed cat g b.commands := by
  refine (pairwise_of_valid hv).imp_of_mem fun {a c} ha hc h => ⟨?_, ?_⟩
  · intro k hk hk'
    obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hk'
    exact h.1 p.key hk (List.mem_append_left _ (hpr c hc p hp))
  · intro k hk hk'
    obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hk'
    exact h.2.1 p.key hk (hpr a ha p hp)

/-- **Preconditions still hold when each command starts.** If no command of an admitted batch
writes a key another command's preconditions are about (decidable; e.g. when preconditions are
on read keys, `preconditionsUndisturbed_of_read`), then in every execution order each command's
preconditions hold when it starts. -/
theorem admitted_preconditions_hold {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} {run : Behaviour} (hv : AllocationValid s cat g b)
    (hr : Respects cat run) (hpu : PreconditionsUndisturbed cat g b.commands)
    {order : List Command} (hp : order.Perm b.commands) :
    PreconditionsAtStart cat g run s.valuation order :=
  preconditionsAtStart_of (stable_of hr (hpu.perm hp.symm Undisturbed.symm) fun _ _ _ _ => rfl)
    (fun c hc => preconditions_of_valid hv c (hp.mem_iff.mp hc))

end AllocationChecker
