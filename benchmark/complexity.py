"""Complexity vector C(task) of the benchmark test cases (DRAFT, see EXTENSION.md).

C(task) = (intents, devices, graph_depth, graph_width, ambiguity,
           temporal_constraints, spatial_constraints, conflicts)

Usage:
    python complexity.py            # table of all cases, and averages per category / dimension
    python complexity.py --json     # the vectors as JSON
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DEVICE_TYPES = {"search", "fridge", "display", "audio", "clock", "thermostat", "light", "calendar"}
FEATURES = ["intents", "devices", "graph_depth", "graph_width", "ambiguity",
            "temporal_constraints", "spatial_constraints", "conflicts"]


def intents_of(case):
    """All listed intents: required ones first, then acceptable ones (the maximal interpretation)."""
    return case.get("required_intents", []) + case.get("acceptable_intents", [])


def device_types(intent):
    """The device types of an intent ('clock or calendar' counts as one alternative)."""
    return intent["device_type"].split(" or ")


def check_annotation(case, ann):
    """Raise ValueError if the annotation does not fit the case."""
    n = len(intents_of(case))
    def node_ok(x):
        return (isinstance(x, int) and 0 <= x < n) or (isinstance(x, str) and x in DEVICE_TYPES)
    for a, b in ann.get("deps", []) + ann.get("inferred_deps", []):
        if not (node_ok(a) and node_ok(b)) or a == b:
            raise ValueError(f"{case['id']}: bad dependency {[a, b]}")
    for a, b in ann.get("conflicts", []):
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b < n):
            raise ValueError(f"{case['id']}: bad conflict {[a, b]}")
    failure = case.get("injected_failure")
    if failure is not None:
        i = failure["intent"]
        if not (0 <= i < n and failure["device_type"] in device_types(intents_of(case)[i])):
            raise ValueError(f"{case['id']}: injected failure does not match an intent")
        rec = case["expected_recovery"]
        listed = rec["must_complete"] + rec["must_report"] + rec["must_replan"]
        if sorted(listed) != list(range(n)):
            raise ValueError(f"{case['id']}: expected_recovery must list every intent once")


def levels(n, edges):
    """Level of each node in the dependency graph: 1 for nodes without prerequisites, otherwise
    one more than the highest prerequisite. Nodes are intents 0..n-1 and helper devices."""
    nodes = set(range(n)) | {x for e in edges for x in e}
    preds = defaultdict(set)
    for a, b in edges:
        preds[b].add(a)
    level = {}
    def visit(v, path):
        if v in level:
            return level[v]
        if v in path:
            raise ValueError("cyclic dependencies")
        level[v] = 1 + max((visit(u, path | {v}) for u in preds[v]), default=0)
        return level[v]
    for v in nodes:
        visit(v, frozenset())
    return level


def complexity(case, ann):
    """The complexity vector of one test case."""
    check_annotation(case, ann)
    intents = intents_of(case)
    n = len(intents)
    level = levels(n, [tuple(e) for e in ann.get("deps", []) + ann.get("inferred_deps", [])])
    width = defaultdict(int)
    for v in level.values():
        width[v] += 1
    return {
        "intents": n,
        "devices": len({intent["device_type"] for intent in intents}),
        "graph_depth": max(level.values(), default=0),
        "graph_width": max(width.values(), default=0),
        "ambiguity": len(case.get("acceptable_intents", [])),
        "temporal_constraints": len(ann.get("temporal", [])),
        "spatial_constraints": len(ann.get("spatial", [])),
        "conflicts": len(ann.get("conflicts", [])),
    }


def load():
    """All test cases with their annotations: (case, group, vector)."""
    original = json.loads((HERE / "benchmark_data.json").read_text())["test_cases"]
    annotations = json.loads((HERE / "complexity_annotations.json").read_text())["cases"]
    extension = json.loads((HERE / "extension_tasks.json").read_text())["test_cases"]
    rows = []
    for case in original:
        rows.append((case, case["category"], complexity(case, annotations[case["id"]])))
    for case in extension:
        rows.append((case, case["dimension"], complexity(case, case["annotation"])))
    return rows


def main():
    rows = load()
    if "--json" in sys.argv:
        print(json.dumps({case["id"]: vec for case, _, vec in rows}, indent=1))
        return
    short = ["int", "dev", "dep", "wid", "amb", "tmp", "spa", "con"]
    print(f"{'id':16}" + "".join(f"{s:>5}" for s in short))
    for case, _, vec in rows:
        print(f"{case['id']:16}" + "".join(f"{vec[f]:>5}" for f in FEATURES))
    print("\nAverages")
    groups = defaultdict(list)
    for _, group, vec in rows:
        groups[group].append(vec)
    print(f"{'group':22}{'n':>4}" + "".join(f"{s:>6}" for s in short))
    for group, vecs in groups.items():
        means = [sum(v[f] for v in vecs) / len(vecs) for f in FEATURES]
        print(f"{group:22}{len(vecs):>4}" + "".join(f"{m:>6.1f}" for m in means))


if __name__ == "__main__":
    main()
