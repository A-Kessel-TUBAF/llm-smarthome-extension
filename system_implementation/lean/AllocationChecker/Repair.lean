/-
  Dependency-aware local repair across rounds.

  A plan runs in rounds; the commands of a round run in parallel (each round is a batch the
  checker admits, so its commands are pairwise independent). When commands fail, the rest of
  the plan does not have to be replanned as a whole. Local repair skips only the commands that
  may be affected:

  * commands that read a key written by a failed or skipped command (a data dependency, found
    from the catalogue's footprints), and
  * commands whose task depends on a failed or skipped task (a declared dependency, as the
    scheduler already does),

  and runs everything else as planned.

  * `repair_sound`: whatever fails (each failed command may leave any values on its own keys),
    every command that still runs starts with its predicted values on the keys it reads, so it
    behaves as planned; and at the end every key outside the tainted ones (written by failed or
    skipped commands) has its predicted value.
  * `repair_no_failures`: without failures nothing is skipped, and the result is the prediction.

  The failed and skipped commands (`RunState.bad`) are what has to be replanned.
-/
import AllocationChecker.Failures

namespace AllocationChecker

/-- Bookkeeping while a plan runs. -/
structure RunState where
  /-- The actual valuation. -/
  val : Valuation
  /-- Tasks of failed or skipped commands; these have to be replanned. -/
  bad : List Nat
  /-- Keys written by failed or skipped commands: their values may differ from the plan. -/
  taint : List StateKey

/-- A run that has not started yet. -/
def RunState.start (v : Valuation) : RunState := ⟨v, [], []⟩

section
variable (cat : Catalogue) (deps : Nat → List Nat) (failed : Command → Bool) (act : Behaviour)

/-- Local repair: skip a command that reads a tainted key, or whose task depends on a failed or
skipped task. -/
def skip (r : RunState) (c : Command) : Bool :=
  (footprint cat c).reads.any (fun k => decide (k ∈ r.taint)) ||
    (deps c.task).any (fun d => decide (d ∈ r.bad))

/-- A command is bad in a round if it is skipped or fails. -/
def isBad (r : RunState) (c : Command) : Bool := skip cat deps r c || failed c

/-- What a command actually does in a round: nothing if skipped, otherwise its actual
behaviour. -/
def stepAct (r : RunState) : Behaviour := fun c v => if skip cat deps r c then v else act c v

/-- Run one round. -/
def runRound (r : RunState) (cs : List Command) : RunState :=
  let bad := cs.filter (isBad cat deps failed r)
  { val := runSeq (stepAct cat deps act r) r.val cs
    bad := r.bad ++ bad.map (·.task)
    taint := r.taint ++ bad.flatMap fun c => (footprint cat c).writes }

/-- Run the rounds one after another. -/
def runPlan : RunState → List (List Command) → RunState
  | r, [] => r
  | r, cs :: rest => runPlan (runRound cat deps failed act r cs) rest

end

/-- The predicted valuation after the rounds, without failures. -/
def predictPlan (run : Behaviour) : Valuation → List (List Command) → Valuation
  | v, [] => v
  | v, cs :: rest => predictPlan run (runSeq run v cs) rest

/-- In every round, every command that is not skipped starts with the predicted values on the
keys it reads. -/
def StartsAsPredicted (cat : Catalogue) (deps : Nat → List Nat) (r : RunState)
    (behave run : Behaviour) : Valuation → Valuation → List Command → Prop
  | _, _, [] => True
  | v, p, c :: cs =>
    (skip cat deps r c = false → ∀ k ∈ (footprint cat c).reads, v k = p k) ∧
      StartsAsPredicted cat deps r behave run (behave c v) (run c p) cs

def PlanStartsAsPredicted (cat : Catalogue) (deps : Nat → List Nat) (failed : Command → Bool)
    (act run : Behaviour) : RunState → Valuation → List (List Command) → Prop
  | _, _, [] => True
  | r, p, cs :: rest =>
    StartsAsPredicted cat deps r (stepAct cat deps act r) run r.val p cs ∧
      PlanStartsAsPredicted cat deps failed act run (runRound cat deps failed act r cs)
        (runSeq run p cs) rest

section
variable {cat : Catalogue} {deps : Nat → List Nat} {failed : Command → Bool}
  {act run : Behaviour} (hr : Respects cat run) (ha : Act cat run (failed · = true) act)
include hr ha

omit hr in
theorem stepAct_frame (r : RunState) :
    ∀ c v k, k ∉ (footprint cat c).writes → stepAct cat deps act r c v k = v k := by
  intro c v k hk
  unfold stepAct
  split
  · rfl
  · exact ha.frame c v k hk

/-- One round: commands that are not skipped start with their predicted reads, and the actual
valuation keeps agreeing with the prediction outside the tainted keys. -/
theorem round_sound (r : RunState) : ∀ {cs : List Command} {v p : Valuation}
    {T : List StateKey}, cs.Pairwise (IndepCmd cat) → (∀ k, k ∉ T → v k = p k) →
    (∀ c ∈ cs, skip cat deps r c = false → ∀ k ∈ (footprint cat c).reads, k ∉ T) →
    StartsAsPredicted cat deps r (stepAct cat deps act r) run v p cs ∧
      ∀ k, k ∉ T → k ∉ (cs.filter (isBad cat deps failed r)).flatMap
          (fun c => (footprint cat c).writes) →
        runSeq (stepAct cat deps act r) v cs k = runSeq run p cs k
  | [], v, p, T, _, hvp, _ => ⟨trivial, fun k hk _ => hvp k hk⟩
  | c :: cs, v, p, T, hi, hvp, hread => by
    rw [List.pairwise_cons] at hi
    -- the agreement after `c`, outside `T` and (if `c` is bad) the keys `c` writes
    let T' := T ++ (if isBad cat deps failed r c then (footprint cat c).writes else [])
    have hvp' : ∀ k, k ∉ T' → stepAct cat deps act r c v k = run c p k := by
      intro k hk
      have hkT : k ∉ T := fun h => hk (List.mem_append_left _ h)
      by_cases hw : k ∈ (footprint cat c).writes
      · -- `c` writes `k`, so `c` is not bad: it runs, does not fail, and reads predicted values
        have hbad : isBad cat deps failed r c = false := by
          cases hb : isBad cat deps failed r c
          · rfl
          · exact absurd (List.mem_append_right _ (by simp [hb, hw])) hk
        simp only [isBad, Bool.or_eq_false_iff] at hbad
        have hok : act c = run c := ha.ok c (by simp [hbad.2])
        simp only [stepAct, hbad.1, if_false, hok, Bool.false_eq_true]
        exact hr.onlyReads c v p
          (fun k' hk' => hvp k' (hread c (by simp) hbad.1 k' hk')) k hw
      · rw [stepAct_frame ha r c v k hw, hr.frame c p k hw]
        exact hvp k hkT
    have hread' : ∀ u ∈ cs, skip cat deps r u = false → ∀ k ∈ (footprint cat u).reads,
        k ∉ T' := by
      intro u hu hs k hk hkT'
      rcases List.mem_append.mp hkT' with h | h
      · exact hread u (by simp [hu]) hs k hk h
      · split at h
        · exact (hi.1 u hu).1 k h (List.mem_append_left _ hk)
        · exact absurd h List.not_mem_nil
    obtain ⟨ih₁, ih₂⟩ := round_sound r hi.2 hvp' hread'
    refine ⟨⟨fun hs k hk => hvp k (hread c (by simp) hs k hk), ih₁⟩,
      fun k hkT hkbad => ?_⟩
    simp only [runSeq]
    apply ih₂ k
    · intro h
      rcases List.mem_append.mp h with h | h
      · exact hkT h
      · split at h
        · rename_i hb
          exact hkbad (List.mem_flatMap.mpr ⟨c, by simp [hb], h⟩)
        · exact absurd h List.not_mem_nil
    · intro h
      obtain ⟨u, hu, hk⟩ := List.mem_flatMap.mp h
      exact hkbad (List.mem_flatMap.mpr ⟨u, List.mem_filter.mpr
        ⟨List.mem_cons_of_mem _ (List.mem_filter.mp hu).1, (List.mem_filter.mp hu).2⟩, hk⟩)

/-- **Local repair is sound.** If the actual valuation agrees with the prediction outside the
tainted keys, and every round consists of pairwise independent commands (as the checker
guarantees for every admitted batch), then whatever fails: every command that is not skipped
starts with its predicted reads, and at the end every key outside the tainted ones has its
predicted value. -/
theorem repair_sound : ∀ {plan : List (List Command)} {r : RunState} {p : Valuation},
    (∀ cs ∈ plan, cs.Pairwise (IndepCmd cat)) → (∀ k, k ∉ r.taint → r.val k = p k) →
    PlanStartsAsPredicted cat deps failed act run r p plan ∧
      ∀ k, k ∉ (runPlan cat deps failed act r plan).taint →
        (runPlan cat deps failed act r plan).val k = predictPlan run p plan k
  | [], r, p, _, h => ⟨trivial, h⟩
  | cs :: rest, r, p, hi, h => by
    have hread : ∀ c ∈ cs, skip cat deps r c = false → ∀ k ∈ (footprint cat c).reads,
        k ∉ r.taint := by
      intro c _ hs k hk hkT
      simp only [skip, Bool.or_eq_false_iff, List.any_eq_false, decide_eq_true_eq] at hs
      exact hs.1 k hk hkT
    obtain ⟨h₁, h₂⟩ := round_sound hr ha r (hi cs (by simp)) h hread
    have hnext : ∀ k, k ∉ (runRound cat deps failed act r cs).taint →
        (runRound cat deps failed act r cs).val k = runSeq run p cs k := by
      intro k hk
      simp only [runRound, List.mem_append, not_or] at hk ⊢
      exact h₂ k hk.1 hk.2
    obtain ⟨ih₁, ih₂⟩ := repair_sound (plan := rest)
      (fun cs' h' => hi cs' (List.mem_cons_of_mem _ h')) hnext
    exact ⟨⟨h₁, ih₁⟩, ih₂⟩

end

/-- **Without failures, the plan runs as predicted.** Nothing is skipped, nothing is tainted,
and the final valuation is the prediction. -/
theorem repair_no_failures {cat : Catalogue} {deps : Nat → List Nat} {run : Behaviour}
    (hr : Respects cat run) : ∀ {plan : List (List Command)} (v : Valuation),
    (∀ cs ∈ plan, cs.Pairwise (IndepCmd cat)) →
      (runPlan cat deps (fun _ => false) run (RunState.start v) plan).bad = [] ∧
        (runPlan cat deps (fun _ => false) run (RunState.start v) plan).val =
          predictPlan run v plan := by
  -- without failures, a clean run state stays clean
  have clean : ∀ (plan : List (List Command)) (r : RunState), r.bad = [] → r.taint = [] →
      (runPlan cat deps (fun _ => false) run r plan).bad = [] ∧
        (runPlan cat deps (fun _ => false) run r plan).taint = [] := by
    intro plan
    induction plan with
    | nil => intro r hb ht; exact ⟨hb, ht⟩
    | cons cs rest ih =>
      intro r hb ht
      have hnone : cs.filter (isBad cat deps (fun _ => false) r) = [] :=
        List.filter_eq_nil_iff.mpr fun c _ => by simp [isBad, skip, hb, ht]
      exact ih _ (by simp [runRound, hb, hnone]) (by simp [runRound, ht, hnone])
  intro plan v hi
  obtain ⟨hb, ht⟩ := clean plan (RunState.start v) rfl rfl
  refine ⟨hb, funext fun k => ?_⟩
  have ha : Act cat run ((fun _ => false) · = true) run := ⟨hr.frame, fun _ _ => rfl⟩
  exact (repair_sound hr ha hi (fun _ _ => rfl)).2 k (by simp [ht])

end AllocationChecker
