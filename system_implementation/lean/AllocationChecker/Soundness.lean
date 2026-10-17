/-
  Soundness of the executable allocation checker with respect to the
  declarative specification:

    checkAllocation_sound :
      checkAllocation snapshot catalogue graph batch = true →
        AllocationValid snapshot catalogue graph batch

  Proof structure
  ---------------
  1. Generic reflection lemmas: `nodupB_iff`, `pairwiseB_iff`, `disjointB_iff`.
  2. Lookup lemmas: a successful `find?` lookup yields a member with the
     requested identifier (`lookupOp_sound`, ...).
  3. Uniqueness: with duplicate-free identifiers (checked by the catalogue
     well-formedness check), every member with a given identifier equals the
     looked-up one (`eq_of_mem_of_nodup_map`, `cmdFootprint_unique`). This is
     what lets the universally quantified footprint condition of the
     specification follow from the single footprint the checker computes.
  4. Component lemmas for preconditions, evidence, dependencies, argument
     resolution, the load policy, per-command fit, resource independence and
     stable task preconditions.
  5. `checkAllocation_sound` splits the Boolean conjunction and composes 1–4.
-/
import AllocationChecker.Spec
import AllocationChecker.Check

namespace AllocationChecker

/-! ### 1. Generic reflection lemmas -/

theorem nodupB_iff [DecidableEq α] : ∀ {l : List α}, nodupB l = true ↔ l.Nodup
  | [] => by simp [nodupB]
  | x :: xs => by simp [nodupB, List.nodup_cons, nodupB_iff (l := xs)]

theorem pairwiseB_iff {r : α → α → Bool} :
    ∀ {l : List α}, pairwiseB r l = true ↔ l.Pairwise (fun a b => r a b = true)
  | [] => by simp [pairwiseB]
  | x :: xs => by simp [pairwiseB, pairwiseB_iff (l := xs)]

theorem disjointB_iff {a b : List StateKey} : disjointB a b = true ↔ KeysDisjoint a b := by
  simp [disjointB, KeysDisjoint]

theorem cmpB_iff {c : Cmp} {a b : Int} : cmpB c a b = true ↔ c.Holds a b := by
  cases c <;> simp [cmpB, Cmp.Holds]

theorem eligibleB_sound {st : Status} (h : eligibleB st = true) : Eligible st := by
  cases st <;> simp_all [eligibleB, Eligible]

/-! ### 2. Lookup lemmas -/

theorem lookupOp_sound {cat : Catalogue} {i : Nat} {o : OpSpec}
    (h : lookupOp cat i = some o) : OpOf cat i o :=
  ⟨List.mem_of_find?_eq_some h, by simpa using List.find?_some h⟩

theorem lookupDevice_sound {cat : Catalogue} {i : Nat} {d : Device}
    (h : lookupDevice cat i = some d) : DeviceOf cat i d :=
  ⟨List.mem_of_find?_eq_some h, by simpa using List.find?_some h⟩

theorem lookupTask_sound {g : TaskGraph} {i : Nat} {t : Task}
    (h : lookupTask g i = some t) : TaskOf g i t :=
  ⟨List.mem_of_find?_eq_some h, by simpa using List.find?_some h⟩

theorem lookupEntry_sound {s : Snapshot} {k : StateKey} {e : Entry}
    (h : lookupEntry s k = some e) : EntryOf s k e :=
  ⟨List.mem_of_find?_eq_some h, by simpa using List.find?_some h⟩

/-! ### 3. Uniqueness under duplicate-free identifiers -/

theorem eq_of_mem_of_nodup_map {f : α → β} :
    ∀ {l : List α}, (l.map f).Nodup → ∀ {a b : α}, a ∈ l → b ∈ l → f a = f b → a = b
  | [], _, _, _, ha, _, _ => absurd ha List.not_mem_nil
  | x :: xs, hnd, a, b, ha, hb, hab => by
      rw [List.map_cons, List.nodup_cons] at hnd
      obtain ⟨hx, hxs⟩ := hnd
      rcases List.mem_cons.mp ha with rfl | ha'
      · rcases List.mem_cons.mp hb with rfl | hb'
        · rfl
        · exact absurd (hab ▸ List.mem_map_of_mem hb') hx
      · rcases List.mem_cons.mp hb with rfl | hb'
        · exact absurd (hab.symm ▸ List.mem_map_of_mem ha') hx
        · exact eq_of_mem_of_nodup_map hxs ha' hb' hab

/-- With unambiguous operation identifiers, the footprint computed by the
    checker is the only trusted footprint of the command. -/
theorem cmdFootprint_unique {cat : Catalogue} {c : Command} {f f' : Footprint}
    (hwf : CatalogueWF cat) (h : cmdFootprint? cat c = some f)
    (h' : CmdFootprint cat c f') : f' = f := by
  obtain ⟨o', ho', hf'⟩ := h'
  unfold cmdFootprint? at h
  cases hl : lookupOp cat c.op with
  | none => simp [hl] at h
  | some o =>
    simp only [hl, Option.map_some, Option.some.injEq] at h
    have ho := lookupOp_sound hl
    have heq : o' = o := eq_of_mem_of_nodup_map hwf.1 ho'.1 ho.1 (ho'.2.trans ho.2.symm)
    subst heq
    rw [hf', h]

theorem cmdFootprint_sound {cat : Catalogue} {c : Command} {f : Footprint}
    (h : cmdFootprint? cat c = some f) : CmdFootprint cat c f := by
  unfold cmdFootprint? at h
  cases hl : lookupOp cat c.op with
  | none => simp [hl] at h
  | some o =>
    simp only [hl, Option.map_some, Option.some.injEq] at h
    exact ⟨o, lookupOp_sound hl, h.symm⟩

/-! ### 4. Component lemmas -/

theorem preHoldsB_sound {s : Snapshot} {p : Precondition}
    (h : preHoldsB s p = true) : PreHolds s p := by
  unfold preHoldsB at h
  cases hl : lookupEntry s p.key with
  | none => simp [hl] at h
  | some e =>
    simp only [hl] at h
    exact ⟨e, lookupEntry_sound hl, cmpB_iff.mp h⟩

theorem basisCurrentB_sound {s : Snapshot} {kv : StateKey × Nat}
    (h : basisCurrentB s kv = true) : ∃ e, EntryOf s kv.1 e ∧ e.ver = kv.2 := by
  unfold basisCurrentB at h
  cases hl : lookupEntry s kv.1 with
  | none => simp [hl] at h
  | some e =>
    simp only [hl, beq_iff_eq] at h
    exact ⟨e, lookupEntry_sound hl, h⟩

theorem evidenceValidB_sound {s : Snapshot} {ev : Evidence} {d : Nat}
    (h : evidenceValidB s ev d = true) : EvidenceValid s ev d := by
  simp only [evidenceValidB, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
    List.all_eq_true] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  exact ⟨h1, h2, h3, fun kv hkv => basisCurrentB_sound (h4 kv hkv)⟩

theorem depSatisfiedB_sound {s : Snapshot} {g : TaskGraph} {d : Nat}
    (h : depSatisfiedB s g d = true) : DepSatisfied s g d := by
  unfold depSatisfiedB at h
  rw [Bool.and_eq_true, List.any_eq_true] at h
  obtain ⟨hst, ev, hmem, hev⟩ := h
  refine ⟨?_, ev, hmem, evidenceValidB_sound hev⟩
  cases hl : lookupTask g d with
  | none => simp [hl] at hst
  | some u =>
    simp only [hl, beq_iff_eq] at hst
    exact ⟨u, lookupTask_sound hl, hst⟩

theorem argResolvedB_sound {s : Snapshot} {t : Task} :
    ∀ {a : ArgSrc} {v : Int}, argResolvedB s t a v = true → ArgResolved s t a v
  | .lit w, v, h => by simpa [argResolvedB, ArgResolved] using h
  | .ref d off, v, h => by
      simp only [argResolvedB, Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true,
        beq_iff_eq] at h
      obtain ⟨hd, ev, hmem, hev, hout⟩ := h
      exact ⟨hd, ev, hmem, evidenceValidB_sound hev, hout⟩

theorem inBoundsB_sound {b : ArgBound} {v : Int} (h : inBoundsB b v = true) : InBounds b v := by
  simp only [inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h

theorem policyB_sound {cat : Catalogue} {s : Snapshot} {o : OpSpec} {c : Command}
    (h : policyB cat s o c = true) : PolicyHolds cat s o c := by
  intro hc
  unfold policyB at h
  rw [if_pos hc] at h
  cases hm : lookupEntry s cat.meterKey with
  | none => simp [hm] at h
  | some m =>
    cases hcur : lookupEntry s { dev := c.device, attr := cat.loadAttr } with
    | none => simp [hm, hcur] at h
    | some cur =>
      simp only [hm, hcur, decide_eq_true_eq] at h
      exact ⟨m, cur, lookupEntry_sound hm, lookupEntry_sound hcur, h⟩

theorem commandFitsB_sound {cat : Catalogue} {s : Snapshot} {g : TaskGraph} {c : Command}
    {t : Task} {o : OpSpec} {d : Device}
    (h : commandFitsB cat s g c t o d = true) : CommandFits cat s g c t o d := by
  simp only [commandFitsB, Bool.and_eq_true, and_assoc, beq_iff_eq, decide_eq_true_eq,
    List.all_eq_true] at h
  obtain ⟨hel, hop, hdev, hcap, hsup, har, hcar, hbnd, hres, hdeps, htp, hop2, hdl, hpol⟩ := h
  exact
    { eligible := eligibleB_sound hel
      opMatches := hop
      devMatches := hdev
      capability := hcap
      supported := hsup
      arity := har
      contractArity := hcar
      bounds := fun p hp => inBoundsB_sound (hbnd p hp)
      resolved := fun p hp => argResolvedB_sound (hres p hp)
      deps := fun dep hd => depSatisfiedB_sound (hdeps dep hd)
      taskPre := fun p hp => preHoldsB_sound (htp p hp)
      opPre := fun p hp => preHoldsB_sound (hop2 p hp)
      deadline := hdl
      policy := policyB_sound hpol }

theorem checkCommand_sound {cat : Catalogue} {s : Snapshot} {g : TaskGraph} {c : Command}
    (h : checkCommand cat s g c = true) : CommandValid cat s g c := by
  unfold checkCommand at h
  cases ht : lookupTask g c.task with
  | none => simp [ht] at h
  | some t =>
    cases ho : lookupOp cat c.op with
    | none => simp [ht, ho] at h
    | some o =>
      cases hd : lookupDevice cat c.device with
      | none => simp [ht, ho, hd] at h
      | some d =>
        simp only [ht, ho, hd] at h
        exact ⟨t, o, d, lookupTask_sound ht, lookupOp_sound ho, lookupDevice_sound hd,
          commandFitsB_sound h⟩

theorem indepB_sound {f g : Footprint} (h : indepB f g = true) : Indep f g := by
  simp only [indepB, Bool.and_eq_true, disjointB_iff] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

theorem pairIndepB_sound {cat : Catalogue} {a b : Command} (hwf : CatalogueWF cat)
    (h : pairIndepB cat a b = true) :
    ∀ fa fb, CmdFootprint cat a fa → CmdFootprint cat b fb → Indep fa fb := by
  intro fa fb hfa hfb
  unfold pairIndepB at h
  cases ha : cmdFootprint? cat a with
  | none => simp [ha] at h
  | some fa' =>
    cases hb : cmdFootprint? cat b with
    | none => simp [ha, hb] at h
    | some fb' =>
      simp only [ha, hb] at h
      rw [cmdFootprint_unique hwf ha hfa, cmdFootprint_unique hwf hb hfb]
      exact indepB_sound h

theorem activeIndepB_sound {cat : Catalogue} {s : Snapshot} {c : Command}
    (hwf : CatalogueWF cat) (h : activeIndepB cat s c = true) :
    ∀ fc, CmdFootprint cat c fc → ∀ f ∈ s.active, Indep fc f := by
  intro fc hfc f hf
  unfold activeIndepB at h
  cases hc : cmdFootprint? cat c with
  | none => simp [hc] at h
  | some fc' =>
    simp only [hc, List.all_eq_true] at h
    rw [cmdFootprint_unique hwf hc hfc]
    exact indepB_sound (h f hf)

theorem checkIndependence_sound {cat : Catalogue} {s : Snapshot} {cs : List Command}
    (hwf : CatalogueWF cat) (h : checkIndependence cat s cs = true) :
    ResourceIndependent cat s cs := by
  simp only [checkIndependence, Bool.and_eq_true, pairwiseB_iff, List.all_eq_true] at h
  obtain ⟨hp, ha⟩ := h
  exact ⟨hp.imp (fun hab => pairIndepB_sound hwf hab),
    fun c hc => activeIndepB_sound hwf (ha c hc)⟩

/-- With unambiguous task identifiers, `taskPreKeys` gives the keys of the
    task's preconditions. -/
theorem taskPreKeys_eq {g : TaskGraph} {c : Command} {t : Task} (hwf : GraphWF g)
    (ht : TaskOf g c.task t) : taskPreKeys g c = t.pre.map (·.key) := by
  unfold taskPreKeys
  cases hl : lookupTask g c.task with
  | none =>
    simp only [lookupTask, List.find?_eq_none] at hl
    exact absurd (by simp [ht.2]) (hl t ht.1)
  | some u =>
    have hu := lookupTask_sound hl
    obtain rfl : u = t := eq_of_mem_of_nodup_map hwf.1 hu.1 ht.1 (hu.2.trans ht.2.symm)
    rfl

theorem keepsTaskPreB_sound {cat : Catalogue} {g : TaskGraph} {a b : Command}
    (hc : CatalogueWF cat) (hg : GraphWF g) (h : keepsTaskPreB cat g a b = true) :
    KeepsTaskPre cat g a b := by
  intro fa t hfa ht
  unfold keepsTaskPreB at h
  cases ha : cmdFootprint? cat a with
  | none => simp [ha] at h
  | some fa' =>
    simp only [ha] at h
    rw [cmdFootprint_unique hc ha hfa, ← taskPreKeys_eq hg ht]
    exact disjointB_iff.mp h

theorem checkTaskPreStable_sound {cat : Catalogue} {s : Snapshot} {g : TaskGraph}
    {cs : List Command} (hc : CatalogueWF cat) (hg : GraphWF g)
    (h : checkTaskPreStable cat s g cs = true) : TaskPreStable cat s g cs := by
  simp only [checkTaskPreStable, Bool.and_eq_true, pairwiseB_iff, List.all_eq_true] at h
  obtain ⟨hp, ha⟩ := h
  refine ⟨hp.imp fun hab => ⟨keepsTaskPreB_sound hc hg hab.1, keepsTaskPreB_sound hc hg hab.2⟩,
    fun c hcs t ht f hf => ?_⟩
  rw [← taskPreKeys_eq hg ht]
  exact disjointB_iff.mp (ha c hcs f hf)

theorem checkCatalogueWF_sound {cat : Catalogue} (h : checkCatalogueWF cat = true) :
    CatalogueWF cat := by
  simp only [checkCatalogueWF, Bool.and_eq_true, nodupB_iff] at h
  exact h

theorem checkSnapshotWF_sound {s : Snapshot} (h : checkSnapshotWF s = true) : SnapshotWF s :=
  nodupB_iff.mp h

theorem rankOk_sound {g : TaskGraph} {t : Task} {d : Nat} (h : rankOk g t d = true) :
    ∃ u, TaskOf g d u ∧ u.rank < t.rank := by
  unfold rankOk at h
  cases hl : lookupTask g d with
  | none => simp [hl] at h
  | some u =>
    simp only [hl, decide_eq_true_eq] at h
    exact ⟨u, lookupTask_sound hl, h⟩

theorem checkGraphWF_sound {g : TaskGraph} (h : checkGraphWF g = true) : GraphWF g := by
  simp only [checkGraphWF, Bool.and_eq_true, nodupB_iff, List.all_eq_true] at h
  exact ⟨h.1, fun t ht d hd => rankOk_sound (h.2 t ht d hd)⟩

/-! ### 5. Main theorem -/

/-- **Soundness of the allocation checker.** If the executable checker accepts a
    batch, the batch satisfies the declarative allocation specification. -/
theorem checkAllocation_sound
    (snapshot : Snapshot)
    (catalogue : Catalogue)
    (graph : TaskGraph)
    (batch : AllocationBatch) :
    checkAllocation snapshot catalogue graph batch = true →
      AllocationValid snapshot catalogue graph batch := by
  intro h
  simp only [checkAllocation, Bool.and_eq_true, and_assoc, beq_iff_eq, decide_eq_true_eq,
    List.all_eq_true] at h
  obtain ⟨hcat, hsnap, hgraph, hgr, hcr, hcap, hnd, hcmds, hind, hpre⟩ := h
  have hcatWF := checkCatalogueWF_sound hcat
  have hgraphWF := checkGraphWF_sound hgraph
  exact
    { catalogueWF := hcatWF
      snapshotWF := checkSnapshotWF_sound hsnap
      graphWF := hgraphWF
      graphRev := hgr
      catalogueRev := hcr
      workerCap := hcap
      uniqueTasks := nodupB_iff.mp hnd
      commands := fun c hc => checkCommand_sound (hcmds c hc)
      independent := checkIndependence_sound hcatWF hind
      taskPreStable := checkTaskPreStable_sound hcatWF hgraphWF hpre }

/-! ### Corollary: the rank witness certifies acyclicity -/

/-- `DependsOn g a b`: the task with identifier `a` transitively depends on `b`. -/
inductive DependsOn (g : TaskGraph) : Nat → Nat → Prop
  | edge {t : Task} {b : Nat} : t ∈ g.tasks → b ∈ t.deps → DependsOn g t.id b
  | trans {a b c : Nat} : DependsOn g a b → DependsOn g b c → DependsOn g a c

theorem DependsOn.source_task {g : TaskGraph} {a b : Nat} :
    DependsOn g a b → ∃ ta, TaskOf g a ta
  | .edge ht _ => ⟨_, ht, rfl⟩
  | .trans hab _ => hab.source_task

theorem GraphWF.rank_decreases {g : TaskGraph} (hwf : GraphWF g) {a b : Nat} :
    DependsOn g a b → ∀ ta, TaskOf g a ta → ∃ tb, TaskOf g b tb ∧ tb.rank < ta.rank
  | .edge (t := t) ht hb, ta, hta => by
      have heq : ta = t := eq_of_mem_of_nodup_map hwf.1 hta.1 ht hta.2
      subst heq
      exact hwf.2 ta ht _ hb
  | .trans hab hbc, ta, hta => by
      obtain ⟨tb, htb, hlt₁⟩ := hwf.rank_decreases hab ta hta
      obtain ⟨tc, htc, hlt₂⟩ := hwf.rank_decreases hbc tb htb
      exact ⟨tc, htc, Nat.lt_trans hlt₂ hlt₁⟩

/-- A well-formed graph has no dependency cycle. -/
theorem GraphWF.acyclic {g : TaskGraph} (hwf : GraphWF g) {a : Nat} : ¬ DependsOn g a a := by
  intro hcyc
  obtain ⟨ta, hta⟩ := hcyc.source_task
  obtain ⟨tb, htb, hlt⟩ := hwf.rank_decreases hcyc ta hta
  have heq : tb = ta := eq_of_mem_of_nodup_map hwf.1 htb.1 hta.1 (htb.2.trans hta.2.symm)
  subst heq
  exact Nat.lt_irrefl _ hlt

/-- Every graph accepted by the checker is acyclic. -/
theorem checkAllocation_acyclic {s : Snapshot} {cat : Catalogue} {g : TaskGraph}
    {b : AllocationBatch} (h : checkAllocation s cat g b = true) (a : Nat) :
    ¬ DependsOn g a a :=
  (checkAllocation_sound s cat g b h).graphWF.acyclic

end AllocationChecker
