"""classification_eval without models, GPUs or network (Ollama is mocked).

Run from system_implementation/:
    python -m unittest test_classification_eval
"""
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classification_eval import gemma, metrics, modernbert, tasks


class MetricsTests(unittest.TestCase):

    def test_hand_computed_multi_label(self):
        gold = [["a", "b"], ["a"], ["c"]]
        pred = [["a"], ["a", "c"], ["c"]]
        s = metrics.scores(gold, pred, ["a", "b", "c"])
        # a: tp 2; b: fn 1; c: tp 1, fp 1 -> micro P 3/4, R 3/4; macro F1 (1 + 0 + 2/3) / 3
        self.assertAlmostEqual(s["precision_micro"], 0.75)
        self.assertAlmostEqual(s["recall_micro"], 0.75)
        self.assertAlmostEqual(s["f1_macro"], (1 + 0 + 2 / 3) / 3)
        self.assertAlmostEqual(s["accuracy"], 1 / 3)
        self.assertEqual(s["per_class"]["b"]["support"], 1)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed")
    def test_matches_scikit_learn(self):
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.preprocessing import MultiLabelBinarizer
        rng, labels = random.Random(1), ["a", "b", "c", "d"]
        for multi in (True, False):
            gold = [rng.sample(labels, rng.randint(1, 2)) if multi else [rng.choice(labels)] for _ in range(200)]
            pred = [rng.sample(labels, rng.randint(0, 3)) if multi else [rng.choice(labels)] for _ in range(200)]
            ours = metrics.scores(gold, pred, labels)
            binarizer = MultiLabelBinarizer(classes=labels).fit([labels])
            Y, P = binarizer.transform(gold), binarizer.transform(pred)
            for average in ("micro", "macro"):
                p, r, f, _ = precision_recall_fscore_support(Y, P, average=average, zero_division=0)
                self.assertAlmostEqual(ours[f"precision_{average}"], p)
                self.assertAlmostEqual(ours[f"recall_{average}"], r)
                self.assertAlmostEqual(ours[f"f1_{average}"], f)

    def test_paired_verdict(self):
        labels = ["x", "y"]
        gold = [["x"], ["y"]] * 50
        Y = metrics.binarize(gold, labels)
        perfect, wrong = metrics.binarize(gold, labels), metrics.binarize([["y"], ["x"]] * 50, labels)
        samples = metrics.bootstrap(Y, [perfect, wrong], n=200)
        self.assertTrue(((samples[:, 0] - samples[:, 1]) > 0).all())
        same = metrics.bootstrap(Y, [perfect, perfect], n=50)
        self.assertTrue(((same[:, 0] - same[:, 1]) == 0).all())

    def test_dexa_mas_predictions_from_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs, out = Path(tmp) / "logs", Path(tmp) / "pred"
            logs.mkdir(), out.mkdir()
            (logs / "simple_01.json").write_text(json.dumps({"id": "simple_01", "latency_s": 3.0, "episode_result": {
                "tasks": {"t1": {"agent_id": "search_engine"}, "t2": {"agent_id": "fridge"}, "t3": {"agent_id": "search_engine"}}}}))
            (logs / "summary.json").write_text(json.dumps({"overall": {}}))
            self.assertEqual(metrics.dexa_mas_predictions(logs, out), 1)
            row = json.loads((out / "t1_dexa_device_types.gemma4-31b-mas.jsonl").read_text())
            self.assertEqual(row["pred"], ["fridge", "search"])


class GemmaTests(unittest.TestCase):

    def test_parse(self):
        t1, t2 = "t1_dexa_device_types", "t2_homebench_validity"
        self.assertEqual(gemma.parse(t1, '{"labels": ["Light", "clock"]}'), (["light", "clock"], None))
        self.assertEqual(gemma.parse(t1, '{"labels": ["light", "toaster"]}'), (["light"], "unknown"))
        self.assertEqual(gemma.parse(t2, '{"labels": ["normal", "unexist_device"]}'), (["normal"], None))
        self.assertEqual(gemma.parse(t2, '{"labels": "normal"}'), (["normal"], None))
        self.assertEqual(gemma.parse(t2, "not json"), ([], "unparseable"))

    def test_classify_records_and_resumes(self):
        items = [tasks.item("a", "turn on the light", {"light"}), tasks.item("b", "set a timer", {"clock"})]
        reply = lambda *_, **__: io.BytesIO(json.dumps({"message": {"content": '{"labels": ["light"]}'},
                                                        "prompt_eval_count": 10, "eval_count": 5}).encode())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "p.jsonl"
            with patch.object(gemma, "urlopen", side_effect=reply) as call:
                gemma.classify("t1_dexa_device_types", items[:1], "gemma4:31b", "http://x", out)
                gemma.classify("t1_dexa_device_types", items, "gemma4:31b", "http://x", out)
            self.assertEqual(call.call_count, 2)  # the second run only asked for item b
            body = json.loads(call.call_args_list[0].args[0].data)
            self.assertEqual((body["think"], body["options"]["temperature"], body["format"]), (False, 0, "json"))
            self.assertEqual([r["id"] for r in tasks.read_jsonl(out)], ["a", "b"])


class TaskTests(unittest.TestCase):

    def test_dexa_items(self):
        items = tasks.t1_dexa_device_types()
        self.assertEqual(len(items), 100)
        allowed = set(tasks.TASKS["t1_dexa_device_types"]["labels"])
        self.assertTrue(all(i["labels"] and set(i["labels"]) <= allowed for i in items))

    def test_cross_validation_folds_partition_and_stratify(self):
        items = [tasks.item(f"e{n}", "q", {"feasible" if n % 2 else "infeasible"}, group=f"qt{n % 3}") for n in range(60)]
        parts = modernbert.folds(items, 5)
        self.assertEqual(sorted(i["id"] for p in parts for i in p), sorted(i["id"] for i in items))
        self.assertTrue(all(len(p) == 12 for p in parts))
        self.assertEqual([[i["id"] for i in p] for p in parts], [[i["id"] for i in p] for p in modernbert.folds(items, 5)])


if __name__ == "__main__":
    unittest.main()
