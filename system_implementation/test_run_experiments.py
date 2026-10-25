"""Benchmark selection of `run_experiments` (subprocesses mocked; no model or dataset needed).

Run from system_implementation/:
    python -m unittest test_run_experiments
"""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_experiments


class BenchmarkSelectionTests(unittest.TestCase):

    def test_all_runs_four_benchmarks_and_dexa_only_for_mas(self):
        runs = [{"name": "mas", "system": "mas"}, {"name": "react", "system": "react"}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(run_experiments.subprocess, "run",
                return_value=SimpleNamespace(returncode=0)) as run:
            rows = run_experiments.run_experiments(benchmark="all", runs=runs, output_dir=run_experiments.Path(tmp) / "out", timestamped=False)
        self.assertEqual(sorted((r["benchmark"], r["run"]) for r in rows),
                         sorted([("dexa", "mas"), ("homebench", "mas"), ("homebench", "react"), ("simuhome", "mas"),
                                 ("simuhome", "react"), ("homeassistant", "mas"), ("homeassistant", "react")]))
        self.assertEqual(rows[0]["benchmark"], "dexa")  # the included benchmark runs first
        dexa = run.call_args_list[0].args[0]
        self.assertEqual(dexa[dexa.index("--benchmark") + 1], "dexa")
        # The matrix runs the extended DEXA edition, whose path the runner passes explicitly.
        self.assertEqual(dexa[dexa.index("--dataset") + 1], "extended")
        self.assertTrue(dexa[dexa.index("--benchmark-path") + 1].endswith("extension_tasks.json"))

    def test_configured_python_path_matches_child_sys_executable(self):
        # The child compares its sys.executable with this path; `..` would make them differ.
        from configuration import config_path
        path = config_path("../external/venv/Scripts/python.exe", executable=True)
        self.assertNotIn("..", run_experiments.Path(path).parts)
        self.assertEqual(path, config_path(path, executable=True))

    def test_resume_keeps_finished_dexa_run_without_resume_flag(self):
        runs = [{"name": "mas", "system": "mas"}]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_experiments.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                run_experiments.run_experiments(benchmark="dexa", runs=runs, output_dir=run_experiments.Path(tmp) / "out", timestamped=False)
            (run_experiments.Path(tmp) / "out/mas_1/summary.json").write_text('{"overall": {"cases": 1}}', encoding="utf-8")
            with patch.object(run_experiments.subprocess, "run") as run:
                rows = run_experiments.run_experiments(benchmark="dexa", runs=runs, output_dir=run_experiments.Path(tmp) / "out",
                                                       timestamped=False, resume=True)
            run.assert_not_called()
            self.assertEqual(rows[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
