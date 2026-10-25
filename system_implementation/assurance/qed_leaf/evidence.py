"""Evidence: the same proof versions under plain Lean and under QED_LEAF.

  python assurance/qed_leaf/evidence.py --lean <path to Lean 4.33.1>

Versions: the current proofs; the proofs before the taskPreKeys_eq fix (one simp reached
Classical.choice); and the two negative controls of negative_controls.py. QED_LEAF runs the
pinned analyzer with the policy and the protected reference; plain Lean compiles the same
modules and prints the axioms of the target theorems. Writes results/qed_leaf_evidence/
(evidence.json, evidence.md) under system_implementation/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import negative_controls as nc  # noqa: E402
import run_audit as ra  # noqa: E402

TARGETS = ("checkAllocation_sound", "checkAllocation_complete", "GraphWF.acyclic")
FIX = ("exact absurd (beq_iff_eq.mpr ht.2) (hl t ht.1)", "exact absurd (by simp [ht.2]) (hl t ht.1)")
OUT = ra.PROJECT / "results" / "qed_leaf_evidence"


def staged(name: str) -> Path:
    tree = ra.WORK / "evidence" / name
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(ra.stage_candidate(), tree)
    return tree


def current() -> tuple[Path, Path]:
    return staged("current"), ra.POLICY


def pre_fix() -> tuple[Path, Path]:
    tree = staged("pre_fix")
    nc.sub(tree / "AllocationChecker" / "Soundness.lean", *FIX)
    return tree, ra.POLICY


def plain_lean(lean: str, source: Path) -> dict:
    """Compile a copy of the tree with plain Lean and print the targets' axioms."""
    tree = source.parent / f"{source.name}_plain"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(source, tree)
    modules = [m for m in ra.CANDIDATE_MODULES if (tree / "AllocationChecker" / f"{m}.lean").exists()]
    env = {**os.environ, "LEAN_PATH": str(tree)}
    for module in modules:
        src = Path("AllocationChecker") / f"{module}.lean"
        p = subprocess.run([lean, "-o", str(src.with_suffix(".olean")), str(src)], cwd=tree, env=env,
                           capture_output=True, text=True)
        if p.returncode:
            return {"compiles": False, "error": (p.stdout + p.stderr)[-300:], "axioms": {}}
    probe = tree / "Axioms.lean"
    probe.write_text(f"import AllocationChecker.{modules[-1]}\n"
                     + "".join(f"#print axioms AllocationChecker.{t}\n" for t in TARGETS), encoding="utf-8")
    out = subprocess.run([lean, str(probe)], cwd=tree, env=env, capture_output=True, text=True).stdout
    axioms = {m.group(1).removeprefix("AllocationChecker."): [a for a in m.group(2).split(", ") if a]
              for m in re.finditer(r"'([^']+)' depends on axioms: \[([^\]]*)\]", out)}
    return {"compiles": True, "axioms": axioms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lean", default=os.environ.get("LEAN_BIN") or shutil.which("lean"))
    args = parser.parse_args()
    rev = ra.ensure_vendor()
    versions = [("current", "Current proofs (after the fix)", current),
                ("pre_fix", "Upstream proofs before the fix", pre_fix),
                ("weakened_statement", "Negative control: weakened theorem statement", nc.weakened_statement),
                ("weakened_definition", "Negative control: weakened independence definition", nc.weakened_definition)]
    rows = []
    for key, label, build in versions:
        tree, policy = build()
        qed = nc.run(f"evidence_{key}", tree, policy)  # before compiling: the snapshot must hold sources only
        lean = plain_lean(args.lean, tree)
        rows.append({"version": key, "label": label, "plain_lean": lean,
                     "qed_leaf": {"status": qed["status"], "failed_checks": [list(c) for c in qed["failed_checks"]]}})
        print(f"{key}: plain Lean compiles={lean['compiles']} | QED_LEAF {qed['status']} {qed['failed_checks']}", flush=True)
    version = subprocess.run([args.lean, "--version"], capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps({"qed_leaf_revision": rev, "lean": version,
                                                    "policy_permitted_axioms": json.loads(ra.POLICY.read_text())["permitted_axioms"],
                                                    "versions": rows}, indent=2) + "\n", encoding="utf-8")
    lines = ["| Proof version | Plain Lean compiles | Axioms of checkAllocation_sound (plain Lean) | QED_LEAF | QED_LEAF failed checks |",
             "|---|---|---|---|---|"]
    for r in rows:
        axioms = ", ".join(r["plain_lean"]["axioms"].get("checkAllocation_sound", [])) or "-"
        failed = "; ".join(f"{target}: {check}" for target, check in r["qed_leaf"]["failed_checks"]) or "none"
        lines.append(f"| {r['label']} | {'yes' if r['plain_lean']['compiles'] else 'no'} | {axioms} | "
                     f"**{r['qed_leaf']['status']}** | {failed} |")
    (OUT / "evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
