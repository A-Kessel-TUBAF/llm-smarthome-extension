"""Optional offline QED_LEAF audit of the released verification artifact.

  uv run python assurance/qed_leaf/run_audit.py protect   # (re)create the protected specification
  uv run python assurance/qed_leaf/run_audit.py analyze   # stage the candidate and run the QED_LEAF analyzer

The protected specification (`protected/`) is a frozen, verbatim copy of the
model, the declarative predicate and the checker (Types/Spec/Check) plus the
target theorem statements with `sorry`. QED_LEAF's statement-match and
definition-hole checks compare the candidate's elaborated targets and every
definition they depend on against it, so the meaning of the theorem cannot
change silently. The candidate is staged from `lean/AllocationChecker/`.

QED_LEAF is used at a pinned revision cloned into `vendor/QED_LEAF`
(git-ignored); the user's own checkout is never modified. This audit is not
part of the online control path and not part of ordinary CI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LEAN_SRC = PROJECT / "lean" / "AllocationChecker"
VENDOR = HERE / "vendor" / "QED_LEAF"
QED_LEAF_REPO = "https://github.com/vaibhav-baj/QED_LEAF.git"
QED_LEAF_REV = "9262eb0b3a175f4a3bd1b44c2dccdcd435817718"
PROTECTED = HERE / "protected"
WORK = HERE / "work"
POLICY = HERE / "qed-leaf-policy.json"
SPEC_MODULES = ("Types", "Spec", "Check")
CANDIDATE_MODULES = ("Types", "Spec", "Check", "Soundness", "Completeness")

STATEMENTS = '''/-
  Protected target statements (QED_LEAF). The proofs are deliberately `sorry`:
  this file fixes WHAT must be proved; the candidate supplies the proofs.
  `DependsOn` is restated verbatim from Soundness.lean because the acyclicity
  statement depends on it.
-/
import AllocationChecker.Spec
import AllocationChecker.Check

namespace AllocationChecker

theorem checkAllocation_sound
    (snapshot : Snapshot)
    (catalogue : Catalogue)
    (graph : TaskGraph)
    (batch : AllocationBatch) :
    checkAllocation snapshot catalogue graph batch = true →
      AllocationValid snapshot catalogue graph batch := sorry

theorem checkAllocation_complete
    (snapshot : Snapshot)
    (catalogue : Catalogue)
    (graph : TaskGraph)
    (batch : AllocationBatch) :
    AllocationValid snapshot catalogue graph batch →
      checkAllocation snapshot catalogue graph batch = true := sorry

/-- `DependsOn g a b`: the task with identifier `a` transitively depends on `b`. -/
inductive DependsOn (g : TaskGraph) : Nat → Nat → Prop
  | edge {t : Task} {b : Nat} : t ∈ g.tasks → b ∈ t.deps → DependsOn g t.id b
  | trans {a b c : Nat} : DependsOn g a b → DependsOn g b c → DependsOn g a c

theorem GraphWF.acyclic {g : TaskGraph} (hwf : GraphWF g) {a : Nat} : ¬ DependsOn g a a := sorry

end AllocationChecker
'''


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.lean")):
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return "sha256:" + h.hexdigest()


def candidate_digest() -> str:
    """`tree_digest` of the candidate that `stage_candidate` would stage, without staging it."""
    h = hashlib.sha256()
    for m in sorted(CANDIDATE_MODULES):
        h.update(f"AllocationChecker/{m}.lean".encode() + b"\0" + (LEAN_SRC / f"{m}.lean").read_bytes() + b"\0")
    return "sha256:" + h.hexdigest()


def protect() -> None:
    if PROTECTED.exists():
        shutil.rmtree(PROTECTED)
    (PROTECTED / "AllocationChecker").mkdir(parents=True)
    for m in SPEC_MODULES:
        shutil.copy2(LEAN_SRC / f"{m}.lean", PROTECTED / "AllocationChecker" / f"{m}.lean")
    (PROTECTED / "AllocationChecker" / "Statements.lean").write_text(STATEMENTS, encoding="utf-8", newline="\n")
    manifest = {"created_at": datetime.now(UTC).isoformat(), "protected_digest": tree_digest(PROTECTED),
                "files": {p.relative_to(PROTECTED).as_posix(): sha256(p) for p in sorted(PROTECTED.rglob("*.lean"))}}
    (HERE / "protected-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def ensure_vendor() -> str:
    if not (VENDOR / ".git").exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", QED_LEAF_REPO, str(VENDOR)], check=True)
    subprocess.run(["git", "-C", str(VENDOR), "checkout", "-q", QED_LEAF_REV], check=True)
    rev = subprocess.run(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(VENDOR), "status", "--porcelain"], capture_output=True, text=True,
                           check=True).stdout.strip()
    if rev != QED_LEAF_REV or dirty:
        raise SystemExit(f"vendored QED_LEAF is not the clean pinned revision ({rev}, dirty={bool(dirty)})")
    return rev


def stage_candidate() -> Path:
    cand = WORK / "candidate"
    if cand.exists():
        shutil.rmtree(cand)
    (cand / "AllocationChecker").mkdir(parents=True)
    for m in CANDIDATE_MODULES:
        shutil.copy2(LEAN_SRC / f"{m}.lean", cand / "AllocationChecker" / f"{m}.lean")
    return cand


def analyze() -> int:
    rev = ensure_vendor()
    cand = stage_candidate()
    out = WORK / "qed-leaf-out"
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ, PYTHONPATH=str(VENDOR))
    cmd = [sys.executable, "-m", "qed_leaf", "analyze", "--source", str(cand), "--policy", str(POLICY),
           "--protected", str(PROTECTED), "--output", str(out), "--local-trust", "--no-cache"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=VENDOR)
    try:
        summary = json.loads(p.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        summary = {"status": "inconclusive", "reason": "analyzer produced no JSON summary",
                   "stderr": p.stderr[-2000:]}
    report = None
    if summary.get("report"):
        candidates = list(out.rglob(f"{summary['report']}.json"))
        if candidates:
            report = json.loads(candidates[0].read_text(encoding="utf-8"))
    record = {
        "mode": "qed-leaf analyze (analyzer: no Lake, no sandbox; not the release Gate)",
        "assurance_level": "advisory",
        "qed_leaf_revision": rev,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "policy_digest": sha256(POLICY),
        "protected_digest": tree_digest(PROTECTED),
        "candidate_digest": tree_digest(cand),
        "exit_code": p.returncode,
        "summary": summary,
        "checks": [{k: c[k] for k in ("target", "check", "status")} for c in (report or {}).get("checks", [])],
        "definitions_checked": len((report or {}).get("definitions", [])),
        "run_at": datetime.now(UTC).isoformat(),
    }
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "analyze-latest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if report is not None:
        import gzip

        with gzip.open(res / "analyze-latest-report.json.gz", "wt", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("mode", "qed_leaf_revision", "exit_code", "summary")}, indent=2))
    return 0 if summary.get("status") == "passed" else 2


def current_record() -> dict:
    """The analyzer record for the current candidate, policy, reference and pinned revision.

    `results/analyze-latest.json` is reused when all four digests match; otherwise the analyzer
    runs again. A setup failure (e.g. no access to the QED_LEAF repository) is inconclusive.
    """
    expected = {"qed_leaf_revision": QED_LEAF_REV, "policy_digest": sha256(POLICY),
                "protected_digest": tree_digest(PROTECTED), "candidate_digest": candidate_digest()}
    latest = HERE / "results" / "analyze-latest.json"
    if latest.exists():
        record = json.loads(latest.read_text(encoding="utf-8"))
        if all(record.get(k) == v for k, v in expected.items()):
            return record
    try:
        analyze()
        _publish()
    except (OSError, subprocess.CalledProcessError, SystemExit) as exc:
        return {**expected, "summary": {"status": "inconclusive", "reason": f"{type(exc).__name__}: {exc}"}}
    return json.loads(latest.read_text(encoding="utf-8"))


def _publish() -> None:
    sys.path.insert(0, str(HERE))
    from sanitize import sanitize_results

    sanitize_results()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "protect":
        protect()
        sys.exit(0)
    rc = analyze()
    _publish()
    sys.exit(rc)
