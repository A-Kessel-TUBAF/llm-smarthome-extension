"""Small, system-independent boundary between a benchmark and a participant."""
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Callable
from model_client import summarize_calls

SYSTEMS = {"mas": "MAS", "react": "ReAct", "sage": "SAGE", "dsia": "DSIA"}
BENCHMARKS = ("homebench", "simuhome", "homeassistant")


@dataclass
class Episode:
    benchmark: str
    request: str
    context: dict
    tools: list[dict]
    invoke: Callable
    environment: object


def create_system(system=None, settings=None, config_path=None):
    """Only the selected wrapper imports its implementation and dependencies."""
    if hasattr(system, "run_episode"):
        return system
    if system is None or system == "mas" or hasattr(system, "run_request"):
        from .systems.mas import MAS
        return MAS(settings=settings, config_path=config_path,
                   module=system if hasattr(system, "run_request") else None)
    if isinstance(system, str) and system in SYSTEMS:
        wrapper = getattr(importlib.import_module(".systems." + system, __package__), SYSTEMS[system])
        return wrapper(settings=settings, config_path=config_path)
    if isinstance(system, str) and ":" in system:
        module, factory = system.rsplit(":", 1)
        participant = getattr(importlib.import_module(module), factory)(settings=settings or {}, config_path=config_path)
        if not hasattr(participant, "run_episode"):
            raise TypeError("A system factory must return an object with run_episode(Episode).")
        return participant
    raise ValueError(f"Unknown system {system!r}; choose {', '.join(SYSTEMS)} or module:factory.")


def source_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py")) if "tests" not in p.parts} | {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root.parent / "configuration.py", root.parent / "model_client.py")}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def prepare_output(path, metadata, resume=False):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, "python": sys.version, "adapter_sha256": source_hashes()}
    config = output / "config.json"
    if config.exists():
        if not resume:
            raise FileExistsError(f"Results already exist in {output}; use --resume or a new --output-dir.")
        if json.loads(config.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Resume refused: system, settings, selected cases, data or source code changed.")
    elif resume and any(p.name not in {"command.json", "failure.json"} for p in output.glob("*.json")):
        raise ValueError("Cannot resume results without config.json.")
    else:
        write_json(config, metadata)
    return output


def case_file(output, case_id):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(case_id))
    if name != str(case_id) or name in {".", "..", "config", "summary", "failure"}:
        name = name[:100] + "_" + hashlib.sha256(str(case_id).encode()).hexdigest()[:12]
    return Path(output) / (name + ".json")


def cached_case(output, case_id, resume):
    path = case_file(output, case_id)
    if resume and path.exists():
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("completed") and row.get("id") == case_id:
            return row
    return None


def system_report(system, episode):
    """A wrapper failure is visible and must not stop subsequent independent cases."""
    try:
        report = system.run_episode(episode)
        if not isinstance(report, dict):
            raise TypeError("run_episode must return a report dictionary.")
        if not isinstance(report.get("answer", ""), str):
            raise TypeError("run_episode report.answer must be a string.")
    except Exception as exc:
        report = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                  "exception": type(exc).__name__, "technical_error": True}
    report.setdefault("answer", "")
    report.setdefault("status", "error" if report.get("error") else "finished")
    report.setdefault("telemetry_available", "llm_calls" in report)
    report.setdefault("llm_calls", [])
    report.setdefault("model_usage", summarize_calls(report["llm_calls"]))
    return report
