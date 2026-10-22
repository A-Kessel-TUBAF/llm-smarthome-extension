"""QED_LEAF integration tests; analyzer calls are mocked, Lean compilation is optional."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "assurance/qed_leaf"
LEAN = os.environ.get("LEAN_BIN") or shutil.which("lean")


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, AUDIT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QEDLeafTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load("run_audit", "run_audit.py")
        path = list(sys.path)
        try:
            with patch.dict(sys.modules, {"run_audit": cls.audit}):
                cls.controls = load("negative_controls", "negative_controls.py")
        finally:
            sys.path[:] = path

    def test_protected_current_specification_and_original_targets(self):
        manifest = json.loads((AUDIT / "protected-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["protected_digest"], self.audit.tree_digest(self.audit.PROTECTED))
        for name, digest in manifest["files"].items():
            self.assertEqual(digest, self.audit.sha256(self.audit.PROTECTED / name))
        for name in self.audit.SPEC_MODULES:
            self.assertEqual((self.audit.LEAN_SRC / f"{name}.lean").read_bytes(),
                (self.audit.PROTECTED / "AllocationChecker" / f"{name}.lean").read_bytes())
        protected_spec = (self.audit.PROTECTED / "AllocationChecker/Spec.lean").read_text(encoding="utf-8")
        self.assertIn("taskPreStable : TaskPreStable", protected_spec)
        policy = json.loads(self.audit.POLICY.read_text(encoding="utf-8"))
        self.assertEqual(policy["toolchain"], (ROOT / "lean/lean-toolchain").read_text(encoding="utf-8").strip())
        self.assertEqual(policy["permitted_axioms"], ["propext", "Quot.sound"])
        self.assertEqual(policy["targets"], ["AllocationChecker.checkAllocation_sound",
            "AllocationChecker.checkAllocation_complete", "AllocationChecker.GraphWF.acyclic"])

    def test_staging_and_negative_controls_preserve_sources_and_reference(self):
        sources = {p: p.read_bytes() for p in self.audit.LEAN_SRC.glob("*.lean")}
        frozen = self.audit.tree_digest(self.audit.PROTECTED)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.audit, "WORK", Path(tmp)):
            candidate = self.audit.stage_candidate()
            for name in self.audit.CANDIDATE_MODULES:
                self.assertEqual((candidate / "AllocationChecker" / f"{name}.lean").read_bytes(),
                    (self.audit.LEAN_SRC / f"{name}.lean").read_bytes())
            statement, _ = self.controls.weakened_statement()
            self.assertIn("true → CatalogueWF catalogue", (statement / "AllocationChecker/Soundness.lean").read_text(encoding="utf-8"))
            definition, policy_path = self.controls.weakened_definition()
            self.assertNotIn("KeysDisjoint f.excl g.excl", (definition / "AllocationChecker/Spec.lean").read_text(encoding="utf-8"))
            self.assertNotIn("AllocationChecker.checkAllocation_complete", json.loads(policy_path.read_text(encoding="utf-8"))["targets"])
        self.assertEqual(sources, {p: p.read_bytes() for p in sources})
        self.assertEqual(frozen, self.audit.tree_digest(self.audit.PROTECTED))

    @unittest.skipUnless(LEAN, "Set LEAN_BIN or install Lean 4.33.1 to compile audit inputs")
    def test_current_protected_candidate_and_both_negative_controls_compile(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.audit, "WORK", Path(tmp) / "work"):
            candidate = self.audit.stage_candidate()
            protected = Path(tmp) / "protected"
            shutil.copytree(self.audit.PROTECTED, protected)
            statement, _ = self.controls.weakened_statement()
            definition, _ = self.controls.weakened_definition()
            trees = [(candidate, self.audit.CANDIDATE_MODULES),
                (protected, (*self.audit.SPEC_MODULES, "Statements")),
                (statement, self.audit.CANDIDATE_MODULES),
                (definition, ("Types", "Spec", "Check", "Soundness"))]
            for tree, modules in trees:
                with self.subTest(tree=tree.name):
                    for module in modules:
                        source = Path("AllocationChecker") / f"{module}.lean"
                        result = subprocess.run([LEAN, "-o", str(source.with_suffix(".olean")), str(source)],
                            cwd=tree, env={**os.environ, "LEAN_PATH": str(tree)},
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_advisory_result_records_and_missing_analyzer_summary(self):
        responses = [(json.dumps({"status": "passed"}), 0, 0, "passed"),
                     (json.dumps({"status": "failed"}), 2, 2, "failed"),
                     ("", 1, 2, "inconclusive")]
        for stdout, returncode, expected_rc, status in responses:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(self.audit, "HERE", root), patch.object(self.audit, "WORK", root / "work"), \
                     patch.object(self.audit, "ensure_vendor", return_value=self.audit.QED_LEAF_REV), \
                     patch.object(self.audit.subprocess, "run", return_value=SimpleNamespace(
                         stdout=stdout, stderr="fixture response", returncode=returncode)) as run:
                    rc = self.audit.analyze()
                record = json.loads((root / "results/analyze-latest.json").read_text(encoding="utf-8"))
                self.assertEqual(rc, expected_rc)
                self.assertEqual(record["summary"]["status"], status)
                self.assertEqual(record["assurance_level"], "advisory")
                self.assertEqual(record["qed_leaf_revision"], self.audit.QED_LEAF_REV)
                self.assertIn("--protected", run.call_args.args[0])
                self.assertIn("--no-cache", run.call_args.args[0])
                self.assertEqual(record["protected_digest"], self.audit.tree_digest(self.audit.PROTECTED))
                self.assertEqual(record["candidate_digest"], self.audit.tree_digest(root / "work/candidate"))

    def test_candidate_digest_matches_staged_candidate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.audit, "WORK", Path(tmp)):
            self.assertEqual(self.audit.candidate_digest(), self.audit.tree_digest(self.audit.stage_candidate()))

    def test_current_record_reuses_matching_result_and_reruns_stale_or_unavailable(self):
        current = {"qed_leaf_revision": self.audit.QED_LEAF_REV, "policy_digest": self.audit.sha256(self.audit.POLICY),
                   "protected_digest": self.audit.tree_digest(self.audit.PROTECTED),
                   "candidate_digest": self.audit.candidate_digest(), "summary": {"status": "passed"}}
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "results/analyze-latest.json"
            latest.parent.mkdir()
            rerun = lambda: latest.write_text(json.dumps({**current, "summary": {"status": "failed"}}), encoding="utf-8")
            with patch.object(self.audit, "HERE", Path(tmp)), patch.object(self.audit, "_publish"), \
                 patch.object(self.audit, "analyze", side_effect=rerun) as analyze:
                latest.write_text(json.dumps(current), encoding="utf-8")
                self.assertEqual(self.audit.current_record()["summary"]["status"], "passed")
                analyze.assert_not_called()
                latest.write_text(json.dumps({**current, "candidate_digest": "sha256:stale"}), encoding="utf-8")
                self.assertEqual(self.audit.current_record()["summary"]["status"], "failed")
                analyze.assert_called_once()
            with patch.object(self.audit, "HERE", Path(tmp)), patch.object(self.audit, "analyze",
                    side_effect=subprocess.CalledProcessError(128, ["git", "clone"])):
                latest.unlink()
                self.assertEqual(self.audit.current_record()["summary"]["status"], "inconclusive")


class QEDLeafGateTests(unittest.TestCase):
    """`run_experiments.qed_leaf_gate` for each [assurance] mode; the audit itself is mocked."""

    def test_gate_records_requires_or_skips(self):
        import run_experiments
        text = (ROOT / "experiment_config.toml").read_text(encoding="utf-8")
        self.assertIn('qed_leaf = "record"', text)
        cases = [("record", "lean", "failed", "recorded"), ("require", "lean", "passed", "recorded"),
                 ("require", "lean", "failed", "refused"), ("require", "python", "failed", "skipped"),
                 ("off", "lean", "failed", "skipped")]
        for mode, checker, status, expected in cases:
            with self.subTest(mode=mode, checker=checker, status=status), tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / "config.toml"
                config.write_text(text.replace('qed_leaf = "record"', f'qed_leaf = "{mode}"'), encoding="utf-8")
                audit = SimpleNamespace(current_record=lambda: {"summary": {"status": status}})
                with patch.dict(sys.modules, {"run_audit": audit}):
                    if expected == "refused":
                        with self.assertRaises(RuntimeError):
                            run_experiments.qed_leaf_gate({"checker": checker}, Path(tmp), config)
                    else:
                        run_experiments.qed_leaf_gate({"checker": checker}, Path(tmp), config)
                written = Path(tmp) / "qed_leaf.json"
                self.assertEqual(written.exists(), expected != "skipped")
                if written.exists():
                    self.assertEqual(json.loads(written.read_text(encoding="utf-8"))["summary"]["status"], status)
