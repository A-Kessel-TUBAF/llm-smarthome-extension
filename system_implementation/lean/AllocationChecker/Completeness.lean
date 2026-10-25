/-
  Completeness of the executable allocation checker for the restricted
  specification:

    checkAllocation_complete :
      AllocationValid snapshot catalogue graph batch →
        checkAllocation snapshot catalogue graph batch = true

  Together with `checkAllocation_sound` this yields `checkAllocation_iff`.
  Completeness uses the well-formedness conjuncts of `AllocationValid`
  (duplicate-free identifiers and state keys) to show that every lookup of the
  checker returns the unique member named by the specification. It means a
  rejection by the checker is a genuine violation of `AllocationValid`, not an
  artefact of a conservative implementation.
-/
import AllocationChecker.Soundness

namespace AllocationChecker

theorem find?_key_complete {l : List α} {key : α → β} [BEq β] [LawfulBEq β] {i : β} {x : α}
    (hnd : (l.map key).Nodup) (hx : x ∈ l) (hk : key x = i) :
    l.find? (fun y => key y == i) = some x := by
  cases hf : l.find? (fun y => key y == i) with
  | none =>
    rw [List.find?_eq_none] at hf
    exact absurd (by simp [hk]) (hf x hx)
  | some y =>
    have hm := List.mem_of_find?_eq_some hf
    have hp : key y = i := by simpa using List.find?_some hf
    rw [eq_of_mem_of_nodup_map hnd hm hx (hp.trans hk.symm)]

theorem lookupOp_complete {cat : Catalogue} {i : Nat} {o : OpSpec} (hwf : CatalogueWF cat)
    (h : OpOf cat i o) : lookupOp cat i = some o := by
  unfold lookupOp
  exact find?_key_complete (key := fun o : OpSpec => o.id) hwf.1 h.1 h.2

theorem lookupDevice_complete {cat : Catalogue} {i : Nat} {d : Device} (hwf : CatalogueWF cat)
    (h : DeviceOf cat i d) : lookupDevice cat i = some d := by
  unfold lookupDevice
  exact find?_key_complete (key := fun d : Device => d.id) hwf.2 h.1 h.2

theorem lookupTask_complete {g : TaskGraph} {i : Nat} {t : Task} (hwf : GraphWF g)
    (h : TaskOf g i t) : lookupTask g i = some t := by
  unfold lookupTask
  exact find?_key_complete (key := fun t : Task => t.id) hwf.1 h.1 h.2

theorem lookupEntry_complete {s : Snapshot} {k : StateKey} {e : Entry} (hwf : SnapshotWF s)
    (h : EntryOf s k e) : lookupEntry s k = some e := by
  unfold lookupEntry
  exact find?_key_complete (key := fun e : Entry => e.key) (by unfold SnapshotWF at hwf; exact hwf) h.1 h.2

theorem eligibleB_complete {st : Status} (h : Eligible st) : eligibleB st = true := by
  rcases h with h | h <;> simp [h, eligibleB]

theorem preHoldsB_complete {s : Snapshot} {p : Precondition} (hwf : SnapshotWF s)
    (h : PreHolds s p) : preHoldsB s p = true := by
  obtain ⟨e, he, hc⟩ := h
  unfold preHoldsB
  rw [lookupEntry_complete hwf he]
  exact cmpB_iff.mpr hc

theorem basisCurrentB_complete {s : Snapshot} {kv : StateKey × Nat} (hwf : SnapshotWF s)
    (h : ∃ e, EntryOf s kv.1 e ∧ e.ver = kv.2) : basisCurrentB s kv = true := by
  obtain ⟨e, he, hv⟩ := h
  unfold basisCurrentB
  rw [lookupEntry_complete hwf he]
  simpa using hv

theorem evidenceValidB_complete {s : Snapshot} {ev : Evidence} {d : Nat} (hwf : SnapshotWF s)
    (h : EvidenceValid s ev d) : evidenceValidB s ev d = true := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  simp only [evidenceValidB, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, List.all_eq_true]
  exact ⟨⟨⟨h1, h2⟩, h3⟩, fun kv hkv => basisCurrentB_complete hwf (h4 kv hkv)⟩

theorem depSatisfiedB_complete {s : Snapshot} {g : TaskGraph} {d : Nat} (hs : SnapshotWF s)
    (hg : GraphWF g) (h : DepSatisfied s g d) : depSatisfiedB s g d = true := by
  obtain ⟨⟨u, hu, hst⟩, ev, hmem, hev⟩ := h
  unfold depSatisfiedB
  rw [lookupTask_complete hg hu, Bool.and_eq_true, List.any_eq_true]
  exact ⟨by simp [hst], ev, hmem, evidenceValidB_complete hs hev⟩

theorem argResolvedB_complete {s : Snapshot} {t : Task} (hwf : SnapshotWF s) :
    ∀ {a : ArgSrc} {v : Int}, ArgResolved s t a v → argResolvedB s t a v = true
  | .lit w, v, h => by simpa [argResolvedB, ArgResolved] using h
  | .ref d off, v, h => by
      obtain ⟨hd, ev, hmem, hev, hout⟩ := h
      simp only [argResolvedB, Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true, beq_iff_eq]
      exact ⟨hd, ev, hmem, evidenceValidB_complete hwf hev, hout⟩

theorem inBoundsB_complete {b : ArgBound} {v : Int} (h : InBounds b v) : inBoundsB b v = true := by
  simp only [inBoundsB, Bool.and_eq_true, decide_eq_true_eq]
  exact h

theorem policyB_complete {cat : Catalogue} {s : Snapshot} {o : OpSpec} {c : Command}
    (hwf : SnapshotWF s) (h : PolicyHolds cat s o c) : policyB cat s o c = true := by
  unfold policyB
  cases hc : o.coupled with
  | false => rfl
  | true =>
    obtain ⟨m, cur, hm, hcur, hle⟩ := h hc
    simp only [if_true, lookupEntry_complete hwf hm, lookupEntry_complete hwf hcur,
      decide_eq_true_eq]
    exact hle

theorem commandFitsB_complete {cat : Catalogue} {s : Snapshot} {g : TaskGraph} {c : Command}
    {t : Task} {o : OpSpec} {d : Device} (hs : SnapshotWF s) (hg : GraphWF g)
    (h : CommandFits cat s g c t o d) : commandFitsB cat s g c t o d = true := by
  simp only [commandFitsB, Bool.and_eq_true, and_assoc, beq_iff_eq, decide_eq_true_eq,
    List.all_eq_true]
  exact ⟨eligibleB_complete h.eligible, h.opMatches, h.devMatches, h.capability, h.supported,
    h.arity, h.contractArity, fun p hp => inBoundsB_complete (h.bounds p hp),
    fun p hp => argResolvedB_complete hs (h.resolved p hp),
    fun dep hd => depSatisfiedB_complete hs hg (h.deps dep hd),
    fun p hp => preHoldsB_complete hs (h.taskPre p hp),
    fun p hp => preHoldsB_complete hs (h.opPre p hp), h.deadline,
    policyB_complete hs h.policy⟩

theorem checkCommand_complete {cat : Catalogue} {s : Snapshot} {g : TaskGraph} {c : Command}
    (hc : CatalogueWF cat) (hs : SnapshotWF s) (hg : GraphWF g)
    (h : CommandValid cat s g c) : checkCommand cat s g c = true := by
  obtain ⟨t, o, d, ht, ho, hd, hf⟩ := h
  unfold checkCommand
  rw [lookupTask_complete hg ht, lookupOp_complete hc ho, lookupDevice_complete hc hd]
  exact commandFitsB_complete hs hg hf

theorem disjointB_complete {a b : List StateKey} (h : KeysDisjoint a b) : disjointB a b = true :=
  disjointB_iff.mpr h

theorem indepB_complete {f g : Footprint} (h : Indep f g) : indepB f g = true := by
  simp only [indepB, Bool.and_eq_true, disjointB_iff]
  exact ⟨⟨h.1, h.2.1⟩, h.2.2⟩

theorem cmdFootprint_of_valid {cat : Catalogue} {s : Snapshot} {g : TaskGraph} {c : Command}
    (hc : CatalogueWF cat) (h : CommandValid cat s g c) :
    ∃ f, cmdFootprint? cat c = some f ∧ CmdFootprint cat c f := by
  obtain ⟨_, o, _, _, ho, _, _⟩ := h
  refine ⟨opFootprint cat o c.device, ?_, o, ho, rfl⟩
  simp [cmdFootprint?, lookupOp_complete hc ho]

theorem checkIndependence_complete {cat : Catalogue} {s : Snapshot} {g : TaskGraph}
    {cs : List Command} (hc : CatalogueWF cat)
    (hv : ∀ c ∈ cs, CommandValid cat s g c) (h : ResourceIndependent cat s cs) :
    checkIndependence cat s cs = true := by
  obtain ⟨hp, ha⟩ := h
  simp only [checkIndependence, Bool.and_eq_true, pairwiseB_iff, List.all_eq_true]
  refine ⟨hp.imp_of_mem ?_, ?_⟩
  · intro a b ha hb hab
    obtain ⟨fa, hfa, hfa'⟩ := cmdFootprint_of_valid hc (hv a ha)
    obtain ⟨fb, hfb, hfb'⟩ := cmdFootprint_of_valid hc (hv b hb)
    unfold pairIndepB
    rw [hfa, hfb]
    exact indepB_complete (hab fa fb hfa' hfb')
  · intro c hcs
    obtain ⟨fc, hfc, hfc'⟩ := cmdFootprint_of_valid hc (hv c hcs)
    unfold activeIndepB
    rw [hfc, List.all_eq_true]
    exact fun f hf => indepB_complete (ha c hcs fc hfc' f hf)

theorem checkTaskPreStable_complete {cat : Catalogue} {s : Snapshot} {g : TaskGraph}
    {cs : List Command} (hc : CatalogueWF cat) (hg : GraphWF g)
    (hv : ∀ c ∈ cs, CommandValid cat s g c) (h : TaskPreStable cat s g cs) :
    checkTaskPreStable cat s g cs = true := by
  obtain ⟨hp, ha⟩ := h
  have hkeys : ∀ c ∈ cs, ∃ t, TaskOf g c.task t ∧ taskPreKeys g c = t.pre.map (·.key) := by
    intro c hcs
    obtain ⟨t, _, _, ht, _⟩ := hv c hcs
    exact ⟨t, ht, taskPreKeys_eq hg ht⟩
  have hkeeps : ∀ {a b}, a ∈ cs → b ∈ cs → KeepsTaskPre cat g a b →
      keepsTaskPreB cat g a b = true := by
    intro a b ha' hb hab
    obtain ⟨fa, hfa, hfa'⟩ := cmdFootprint_of_valid hc (hv a ha')
    obtain ⟨t, ht, hk⟩ := hkeys b hb
    unfold keepsTaskPreB
    rw [hfa, hk]
    exact disjointB_complete (hab fa t hfa' ht)
  simp only [checkTaskPreStable, Bool.and_eq_true, pairwiseB_iff, List.all_eq_true]
  refine ⟨hp.imp_of_mem fun ha' hb hab => ⟨hkeeps ha' hb hab.1, hkeeps hb ha' hab.2⟩,
    fun c hcs f hf => ?_⟩
  obtain ⟨t, ht, hk⟩ := hkeys c hcs
  rw [hk]
  exact disjointB_complete (ha c hcs t ht f hf)

theorem checkCatalogueWF_complete {cat : Catalogue} (h : CatalogueWF cat) :
    checkCatalogueWF cat = true := by
  simp only [checkCatalogueWF, Bool.and_eq_true, nodupB_iff]
  exact h

theorem checkGraphWF_complete {g : TaskGraph} (h : GraphWF g) : checkGraphWF g = true := by
  simp only [checkGraphWF, Bool.and_eq_true, nodupB_iff, List.all_eq_true]
  refine ⟨h.1, fun t ht d hd => ?_⟩
  obtain ⟨u, hu, hlt⟩ := h.2 t ht d hd
  unfold rankOk
  rw [lookupTask_complete h hu]
  simpa using hlt

/-- **Completeness for the restricted specification.** Every batch satisfying
    `AllocationValid` is accepted by the executable checker. -/
theorem checkAllocation_complete
    (snapshot : Snapshot)
    (catalogue : Catalogue)
    (graph : TaskGraph)
    (batch : AllocationBatch) :
    AllocationValid snapshot catalogue graph batch →
      checkAllocation snapshot catalogue graph batch = true := by
  intro hv
  simp only [checkAllocation, Bool.and_eq_true, and_assoc, beq_iff_eq, decide_eq_true_eq,
    List.all_eq_true]
  exact ⟨checkCatalogueWF_complete hv.catalogueWF, nodupB_iff.mpr hv.snapshotWF,
    checkGraphWF_complete hv.graphWF, hv.graphRev, hv.catalogueRev, hv.workerCap,
    nodupB_iff.mpr hv.uniqueTasks,
    fun c hc => checkCommand_complete hv.catalogueWF hv.snapshotWF hv.graphWF (hv.commands c hc),
    checkIndependence_complete hv.catalogueWF hv.commands hv.independent,
    checkTaskPreStable_complete hv.catalogueWF hv.graphWF hv.commands hv.taskPreStable⟩

/-- The checker decides the restricted specification exactly. -/
theorem checkAllocation_iff (snapshot : Snapshot) (catalogue : Catalogue) (graph : TaskGraph)
    (batch : AllocationBatch) :
    checkAllocation snapshot catalogue graph batch = true ↔
      AllocationValid snapshot catalogue graph batch :=
  ⟨checkAllocation_sound snapshot catalogue graph batch,
    checkAllocation_complete snapshot catalogue graph batch⟩

/-- A rejected batch genuinely violates the specification. -/
theorem not_valid_of_rejected {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} (h : checkAllocation s cat g b = false) : ¬ AllocationValid s cat g b :=
  fun hv => by simp [checkAllocation_complete s cat g b hv] at h

end AllocationChecker
