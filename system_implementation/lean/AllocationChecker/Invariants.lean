/-
  Safety limits over time.

  The checker admits a command only if its arguments are within its operation's bounds. Many
  operations set a state attribute to one of their arguments (the registry's postconditions
  `equals_arg`, e.g. `set_brightness` sets `brightness_pct` to its argument). Then the attribute
  stays within that argument's bounds, and the home stays within its safety limits over time:

  * `Limit`: a safe range for an attribute, on every device (e.g. brightness within 0–100);
  * `RespectsLimits` (decidable): every operation that writes a limited attribute sets it from
    an argument whose bounds lie within the limit;
  * `reachable_within`: every valuation reachable by running admitted batches, in any order,
    stays within the limits, if the initial valuation does.

  This holds whatever batches the planner proposes. It concerns the commands of the system:
  a device that fails may write anything, which observed-state validation reports
  (`Failures.lean`).
-/
import AllocationChecker.Execution

namespace AllocationChecker

/-- Operation `op` sets attribute `attr` of its device to its argument number `arg`. -/
structure Assign where
  op : Nat
  attr : Nat
  arg : Nat
  deriving DecidableEq, Repr

/-- A safe range `[lo, hi]` for attribute `attr`, on every device. -/
structure Limit where
  attr : Nat
  lo : Int
  hi : Int
  deriving DecidableEq, Repr

/-- The behaviour follows the assignments: a command of an operation that assigns an attribute
sets it, on its device, to that argument. -/
def Follows (assigns : List Assign) (run : Behaviour) : Prop :=
  ∀ a ∈ assigns, ∀ c v, c.op = a.op → run c v ⟨c.device, a.attr⟩ = c.args[a.arg]?

/-- Every value in the valuation lies within the limits of its attribute. -/
def Within (limits : List Limit) (v : Valuation) : Prop :=
  ∀ l ∈ limits, ∀ k x, k.attr = l.attr → v k = some x → l.lo ≤ x ∧ x ≤ l.hi

/-- Argument number `i` of operation `o` has bounds within limit `l`. -/
def ArgWithin (o : OpSpec) (i : Nat) (l : Limit) : Prop :=
  match o.bounds[i]? with
  | some b => l.lo ≤ b.lo ∧ b.hi ≤ l.hi
  | none => False

instance (o : OpSpec) (i : Nat) (l : Limit) : Decidable (ArgWithin o i l) := by
  unfold ArgWithin; split <;> infer_instance

/-- The limits are respected by the catalogue: they are not on the attributes of the load
policy, and every operation that writes a limited attribute sets it from an argument whose
bounds lie within the limit. -/
def RespectsLimits (cat : Catalogue) (assigns : List Assign) (limits : List Limit) : Prop :=
  (∀ l ∈ limits, l.attr ≠ cat.loadAttr ∧ l.attr ≠ cat.meterKey.attr) ∧
    ∀ o ∈ cat.ops, ∀ a ∈ o.writes, ∀ l ∈ limits, l.attr = a →
      ∃ asg ∈ assigns, asg.op = o.id ∧ asg.attr = a ∧ ArgWithin o asg.arg l

instance (cat : Catalogue) (assigns : List Assign) (limits : List Limit) :
    Decidable (RespectsLimits cat assigns limits) := by
  unfold RespectsLimits; infer_instance

/-- The valuations reachable from `v₀` by running admitted batches, each in any order. -/
inductive Reachable (cat : Catalogue) (run : Behaviour) (v₀ : Valuation) : Valuation → Prop
  | init : Reachable cat run v₀ v₀
  | step {v : Valuation} {s : Snapshot} {g : TaskGraph} {b : AllocationBatch}
      {order : List Command} : Reachable cat run v₀ v → AllocationValid s cat g b →
      order.Perm b.commands → Reachable cat run v₀ (runSeq run v order)

/-- A snapshot is within the limits if its entries are. -/
theorem within_of_entries {s : Snapshot} {limits : List Limit}
    (h : ∀ e ∈ s.state, ∀ l ∈ limits, e.key.attr = l.attr →
      l.lo ≤ e.val ∧ e.val ≤ l.hi) :
    Within limits s.valuation := by
  intro l hl k x hk hx
  simp only [Snapshot.valuation, Option.map_eq_some_iff] at hx
  obtain ⟨e, he, rfl⟩ := hx
  have hkey : e.key = k := by simpa [lookupEntry] using List.find?_some he
  exact h e (List.mem_of_find?_eq_some he) l hl (hkey ▸ hk)

section
variable {cat : Catalogue} {assigns : List Assign} {limits : List Limit} {run : Behaviour}
  (hr : Respects cat run) (hf : Follows assigns run) (hl : RespectsLimits cat assigns limits)
include hr hf hl

/-- One command with arguments within its operation's bounds keeps the valuation within the
limits. -/
theorem within_step {c : Command} {o : OpSpec} (ho : OpOf cat c.op o)
    (hfp : footprint cat c = opFootprint cat o c.device)
    (hb : ∀ p ∈ o.bounds.zip c.args, InBounds p.1 p.2) (hlen : c.args.length = o.bounds.length)
    {v : Valuation} (hv : Within limits v) : Within limits (run c v) := by
  intro l hlim k x hk hx
  by_cases hw : k ∈ (footprint cat c).writes
  · rw [hfp] at hw
    simp only [opFootprint, List.mem_append, List.mem_map] at hw
    rcases hw with ⟨a, ha, rfl⟩ | hpol
    · -- an attribute of the device: set from a bounded argument
      obtain ⟨asg, hasg, hop, hattr, hwithin⟩ := hl.2 o ho.1 a ha l hlim hk.symm
      have hrun := hf asg hasg c v (ho.2.symm.trans hop.symm)
      rw [hattr] at hrun
      rw [hrun] at hx
      unfold ArgWithin at hwithin
      split at hwithin
      · rename_i bd hbd
        obtain ⟨hi, rfl⟩ := List.getElem?_eq_some_iff.mp hbd
        have hi' : asg.arg < c.args.length := hlen ▸ hi
        rw [List.getElem?_eq_getElem hi', Option.some.injEq] at hx
        have hz : (o.bounds[asg.arg], c.args[asg.arg]) ∈ o.bounds.zip c.args := by
          rw [← List.getElem_zip (h := by rw [List.length_zip]; omega)]
          exact List.getElem_mem _
        obtain ⟨h₁, h₂⟩ := hb _ hz
        rw [hx] at h₁ h₂
        exact ⟨Int.le_trans hwithin.1 h₁, Int.le_trans h₂ hwithin.2⟩
      · exact hwithin.elim
    · -- a key of the load policy: not limited
      split at hpol
      · simp only [List.mem_cons, List.mem_nil_iff, or_false] at hpol
        rcases hpol with rfl | rfl
        · exact absurd hk.symm (hl.1 l hlim).2
        · exact absurd hk.symm (hl.1 l hlim).1
      · exact absurd hpol List.not_mem_nil
  · rw [hr.frame c v k hw] at hx
    exact hv l hlim k x hk hx

/-- The commands of an admitted batch, run in any order, keep the valuation within the
limits. -/
theorem within_batch {s : Snapshot} {g : TaskGraph} {b : AllocationBatch}
    (hv : AllocationValid s cat g b) : ∀ {order : List Command} {v : Valuation},
    (∀ c ∈ order, c ∈ b.commands) → Within limits v → Within limits (runSeq run v order)
  | [], _, _, h => h
  | c :: cs, v, hmem, h => by
    obtain ⟨_, o, _, _, ho, _, hfit⟩ := hv.commands c (hmem c (by simp))
    have hfp : footprint cat c = opFootprint cat o c.device := by
      simp [footprint, cmdFootprint?, lookupOp_complete hv.catalogueWF ho]
    exact within_batch hv (v := run c v) (fun u hu => hmem u (by simp [hu]))
      (within_step hr hf hl ho hfp hfit.bounds hfit.arity h)

/-- **Safety limits hold over time.** If the catalogue respects the limits and the initial
valuation is within them, every valuation reachable by running admitted batches, in any order,
is within them. -/
theorem reachable_within {v₀ v : Valuation} (h₀ : Within limits v₀)
    (h : Reachable cat run v₀ v) : Within limits v := by
  induction h with
  | init => exact h₀
  | step _ hv hp ih => exact within_batch hr hf hl hv (fun c hc => hp.mem_iff.mp hc) ih

end

end AllocationChecker
