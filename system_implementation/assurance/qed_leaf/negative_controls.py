"""Hand-built negative controls for the QED_LEAF audit (run after `run_audit.py analyze`).

  uv run python assurance/qed_leaf/negative_controls.py

* weakened_statement: `checkAllocation_sound` keeps its name but states only
  `CatalogueWF catalogue`; the strong theorem is renamed. Must fail
  statement-match.
* weakened_definition: `Indep` drops the exclusive-resource condition and the
  soundness proof is adapted so everything still compiles (completeness, which
  would no longer hold, is removed together with its target). The theorem
  statement text is unchanged; only its meaning changed. Must fail
  definition-hole.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_audit as ra  # noqa: E402


def stage(name: str, modules: tuple[str, ...]) -> Path:
    d = ra.WORK / "controls" / name / "AllocationChecker"
    if d.parent.exists():
        shutil.rmtree(d.parent)
    d.mkdir(parents=True)
    for m in modules:
        shutil.copy2(ra.LEAN_SRC / f"{m}.lean", d / f"{m}.lean")
    return d


def sub(path: Path, old: str, new: str, count: int = -1) -> None:
    s = path.read_text(encoding="utf-8")
    assert old in s, (path.name, old[:60])
    path.write_text(s.replace(old, new, count) if count > 0 else s.replace(old, new), encoding="utf-8")


def weakened_statement() -> tuple[Path, Path]:
    d = stage("weakened_statement", ra.CANDIDATE_MODULES)
    for m in ("Soundness", "Completeness"):
        sub(d / f"{m}.lean", "checkAllocation_sound", "checkAllocation_sound_strong")
    sub(d / "Soundness.lean", "\nend AllocationChecker", '''
/-- Weakened statement under the protected name (negative control). -/
theorem checkAllocation_sound (snapshot : Snapshot) (catalogue : Catalogue) (graph : TaskGraph)
    (batch : AllocationBatch) :
    checkAllocation snapshot catalogue graph batch = true → CatalogueWF catalogue :=
  fun h => (checkAllocation_sound_strong snapshot catalogue graph batch h).catalogueWF

end AllocationChecker''')
    return d.parent, ra.POLICY


def weakened_definition() -> tuple[Path, Path]:
    d = stage("weakened_definition", ("Types", "Spec", "Check", "Soundness"))
    sub(d / "Spec.lean", '''  KeysDisjoint g.writes f.reads ∧
  KeysDisjoint f.excl g.excl''', '''  KeysDisjoint g.writes f.reads''')
    sub(d / "Soundness.lean", "  exact ⟨h.1.1, h.1.2, h.2⟩", "  exact ⟨h.1.1, h.1.2⟩")
    policy = json.loads(ra.POLICY.read_text(encoding="utf-8"))
    policy["targets"] = [t for t in policy["targets"] if not t.endswith("checkAllocation_complete")]
    pol = ra.WORK / "controls" / "weakened_definition-policy.json"
    pol.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return d.parent, pol


def run(name: str, cand: Path, policy: Path) -> dict:
    out = ra.WORK / "controls" / f"{name}-out"
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ, PYTHONPATH=str(ra.VENDOR))
    p = subprocess.run([sys.executable, "-m", "qed_leaf", "analyze", "--source", str(cand), "--policy", str(policy),
                        "--protected", str(ra.PROTECTED), "--output", str(out), "--local-trust", "--no-cache"],
                       capture_output=True, text=True, env=env, cwd=ra.VENDOR)
    summary = json.loads(p.stdout.strip().splitlines()[-1])
    report = json.loads(next(out.rglob(f"{summary['report']}.json")).read_text(encoding="utf-8"))
    failed = sorted({(c["target"].split(".")[-1], c["check"]) for c in report["checks"] if c["status"] == "failed"})
    return {"status": summary["status"], "failed_checks": failed}


def main() -> int:
    ra.ensure_vendor()
    results = {}
    for name, fn, expect in (("weakened_statement", weakened_statement, "statement-match"),
                             ("weakened_definition", weakened_definition, "definition-hole")):
        cand, pol = fn()
        r = run(name, cand, pol)
        r["expected_caught_by"] = expect
        r["discriminated"] = r["status"] != "passed" and any(c == expect for _, c in r["failed_checks"])
        results[name] = r
    (ra.HERE / "results").mkdir(exist_ok=True)
    (ra.HERE / "results" / "negative-controls-latest.json").write_text(json.dumps(results, indent=2) + "\n",
                                                                        encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if all(r["discriminated"] for r in results.values()) else 2


def _publish() -> None:
    from sanitize import sanitize_results

    sanitize_results()


if __name__ == "__main__":
    rc = main()
    _publish()
    sys.exit(rc)
