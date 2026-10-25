| Proof version | Plain Lean compiles | Axioms of checkAllocation_sound (plain Lean) | QED_LEAF | QED_LEAF failed checks |
|---|---|---|---|---|
| Current proofs (after the fix) | yes | propext, Quot.sound | **passed** | none |
| Upstream proofs before the fix | yes | propext, Classical.choice, Quot.sound | **failed** | checkAllocation_complete: axiom-policy; checkAllocation_sound: axiom-policy |
| Negative control: weakened theorem statement | yes | propext, Quot.sound | **failed** | checkAllocation_sound: statement-match |
| Negative control: weakened independence definition | yes | propext, Quot.sound | **failed** | acyclic: definition-hole; checkAllocation_sound: definition-hole |
