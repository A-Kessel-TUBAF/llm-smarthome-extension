# QED_LEAF offline audit

This is Vaibhav Bajpai's audit integration from the supplied
`dexa-verified-smarthome-main(5).zip`, applied to this project's Lean sources.
QED_LEAF audits the verification artifact; it does not execute smart-home
requests or provide another system configuration in `run_experiments.py`.

The following files are copied **byte for byte** from his project:

- `run_audit.py`: stage the Lean candidate and call the pinned QED_LEAF analyzer.
- `negative_controls.py`: weaken a theorem statement and an independence
  definition, with corresponding proofs, to test whether the analyzer detects it.
- `sanitize.py`: redact machine-specific paths in publication copies of results.
- `qed-leaf-policy.json`: the original targets, definitions, checkers, limits,
  permitted axioms (`propext`, `Quot.sound`) and Lean 4.33.1 toolchain.

No controller, simulator, model backend or additional experiment runner from
Vaibhav's project is copied. The existing system and benchmark code stay as they are.
The supplied archive does **not** contain the QED_LEAF engine itself: its
`vendor/QED_LEAF` directory is git-ignored in the original project.

## Scope and current Lean sources

The policy retains the original three targets:

1. `AllocationChecker.checkAllocation_sound`
2. `AllocationChecker.checkAllocation_complete`
3. `AllocationChecker.GraphWF.acyclic`

`protected/AllocationChecker/{Types,Spec,Check}.lean` is a frozen, verbatim copy
of **our current** definitions, including operation-precondition reads and stable
task preconditions. `Statements.lean` is the unchanged original statement template.
Its deliberate `sorry` proofs fix the statements to compare against; candidate
proofs come from the current `lean/AllocationChecker/` sources.
The manifest records the protected files and their combined digest.

The frozen specification is intentionally an extra copy: comparing against a
separate reference is the purpose of statement-match/definition-hole checks.
`analyze` never regenerates that reference. New execution, repair, invariant and
typed-value theorems remain part of our ordinary Lean build/axiom audit; this
unchanged QED_LEAF policy does not add them as audit targets.

## Run

From the project root, using the existing Python environment:

```sh
# Compile/check the current Lean project:
(cd lean && lake build)

# Requires Git read access to Vaibhav's private QED_LEAF repository:
python assurance/qed_leaf/run_audit.py analyze
python assurance/qed_leaf/negative_controls.py
```

The original script clones QED_LEAF into `vendor/QED_LEAF` and checks out its
pinned revision `9262eb0b3a175f4a3bd1b44c2dccdcd435817718`. It checks that this
dedicated checkout is clean and does not use or modify another local checkout.
Use QED_LEAF's own installation instructions for any dependencies required by
that private project. The copied wrapper scripts themselves use Python's
standard library. No model, Ollama request or benchmark run is involved.

Candidate sources and raw reports go to `work/`; result records go to `results/`.
Generated work, engine checkout and results are git-ignored locally. The original
sanitizer keeps redacted publication copies separate from raw content-addressed
reports in `work/`. No audit results from Vaibhav's older Lean sources are copied
into this project's results.

After a deliberate specification change, update and review the reference with:

```sh
python assurance/qed_leaf/run_audit.py protect
```

This replaces the protected reference and manifest; do not run it merely to
make a failing candidate match. A fresh audit is then needed for the new reference.

## Assurance and validation status

This preserves Vaibhav's **advisory analyzer** workflow (`analyze` with
`--local-trust`, no Lake and no sandbox). An analyzer pass is not a release-Gate
pass, a proof of mathematical fidelity, a verification of the Python/JSON
boundary, or a success score for a home request.

The release `verify` workflow is not implemented by these wrappers. It requires
QED_LEAF's pinned export/policy tools, protected profile/index, supported sandbox
and the corresponding review workflow, as described in Vaibhav's original README.
The policy is not weakened to obtain a pass.

Here, access to the private QED_LEAF repository is unavailable. The original
`analyze` entry point was exercised and stopped at Git authentication (exit 128),
before any analyzer result. Integration tests use explicitly mocked analyzer
responses; they do not establish that QED_LEAF accepts the current proofs or
detects the controls. Actual analyzer and release-Gate outcomes remain untested.
