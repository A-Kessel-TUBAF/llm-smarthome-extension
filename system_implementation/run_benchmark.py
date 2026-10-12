"""python run_benchmark.py [all|simple|moderate|complex] [benchmark.json]"""
import json
import sys
import time
from pathlib import Path
from smart_home_langgraph import run_request


def run_benchmark(category="all", benchmark_path=None):
    path = Path(benchmark_path) if benchmark_path else Path(__file__).resolve().parent.parent / "benchmark/benchmark_data.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["test_cases"]
    cases = [case for case in cases if category == "all" or case["category"] == category]
    Path("logs").mkdir(exist_ok=True)
    print(f"Running {category.upper()} test cases ({len(cases)} total)")
    for case in cases:
        lines = [f"Test Case ID: {case['id']}", f"Category: {case['category']}", f"User Input: {case['user_input']}", ""]
        started = time.perf_counter()
        for event in run_request(case["user_input"]):
            for node, state in event.items():
                if node == "human":
                    continue
                print(f"Processing Node: {node}")
                lines.append(f"Node: {node}")
                state = state or {}
                for key, label in (("collaboration_request", "COLLABORATION REQUEST"),
                                   ("pending_task", "PENDING TASK"), ("task_queue", "Task Queue")):
                    if state.get(key):
                        lines.append(f"{label}: {json.dumps(state[key], ensure_ascii=False)}")
                for result in state.get("task_results", []):
                    lines.append(f"{result['agent_id']} RESULT: {json.dumps(result, ensure_ascii=False)}")
                if state.get("final_response"):
                    lines.append(f"Final Response: {state['final_response']}")
                lines.append("")
        elapsed = time.perf_counter() - started
        lines.append(f"Execution Time: {elapsed:.2f}s")
        destination = Path("logs") / f"{case['id']}.txt"
        destination.write_text("\n".join(lines), encoding="utf-8")
        print(f"Log saved to: {destination}; Time: {elapsed:.2f}s")


if __name__ == "__main__":
    run_benchmark(sys.argv[1] if len(sys.argv) > 1 else "all", sys.argv[2] if len(sys.argv) > 2 else None)
