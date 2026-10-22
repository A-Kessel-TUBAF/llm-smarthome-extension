"""Original SAIC-MONTREAL/SAGE in an isolated worker with native benchmark I/O."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
from threading import Thread
import time

from ..common import summarize_calls
from .react import BackendUnavailable, ChatClient
from configuration import config_path as configured_path, system_settings

RECOMMENDED_REVISION = "5f43df0a7de2383e4f082a69c591d56d613d1b2f"
DEFAULTS = system_settings("sage")
REQUIRED = ("sage/__init__.py", "sage/base.py", "sage/coordinators/sage_coordinator.py",
            "sage/coordinators/base.py", "sage/coordinators/prompts.py", "sage/smartthings/smartthings_tool.py",
            "sage/smartthings/docmanager.py", "sage/utils/common.py", "sage/utils/llm_utils.py",
            "sage/utils/logging_utils.py", "sage/retrieval/tools.py", "sage/human_interaction/tools.py")


class SAGE:
    name = "sage"

    def __init__(self, settings=None, config_path=None):
        selected = system_settings("sage", config_path)
        options = {"sage_path": os.getenv("SAGE_PATH") or configured_path(selected["sage_path"]),
                   "sage_python": os.getenv("SAGE_PYTHON") or configured_path(selected["sage_python"], executable=True)}
        options.update(settings or {})
        self.client = ChatClient(options, env_prefix="SAGE", config_path=config_path)
        cfg = self.client.settings
        if not cfg["sage_path"]:
            raise ValueError("Set --sage-path or SAGE_PATH to the original SAIC-MONTREAL/SAGE checkout.")
        root = Path(cfg["sage_path"]).expanduser().resolve()
        missing = [name for name in REQUIRED if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError("Incomplete original SAGE checkout: " + ", ".join(missing))
        python = shutil.which(cfg["sage_python"])
        if python is None:
            raise FileNotFoundError("SAGE Python not found. Set --sage-python or SAGE_PYTHON to its Python 3.10 environment.")
        python = str(Path(python).absolute())
        cfg.update(sage_path=str(root), sage_python=python)
        self.settings = cfg
        self.worker = Path(__file__).with_name("sage_worker.py")
        probe = subprocess.run([python, str(self.worker), "--probe", str(root)], capture_output=True, text=True, timeout=30)
        if probe.returncode:
            raise RuntimeError("SAGE environment setup failed: " + probe.stderr[-2000:])
        self.runtime = json.loads(probe.stdout)
        if self.runtime["langchain"] != "0.0.306":
            raise ValueError("Original SAGE requires langchain==0.0.306 in its separate environment.")
        self.hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((root / "sage").rglob("*.py"))}
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        self.commit = commit.stdout.strip() if commit.returncode == 0 else None

    def metadata(self):
        return {"name": self.name, "settings": self.settings, "backend": self.client.metadata(),
                "repository": "https://github.com/SAIC-MONTREAL/SAGE", "commit": self.commit,
                "recommended_revision": RECOMMENDED_REVISION, "source_sha256": self.hashes, "runtime": self.runtime,
                "implementation": "Original SAGECoordinator, SmartThingsTool and API planner; native benchmark service bridge.",
                "protocol": "Cold memory; no photos or external personal services. All original command dispatches retained, including double POST. No MAS or ReAct agent loop."}

    def run_episode(self, episode):
        report = {"answer": "", "rejected_requests": [], "trace": [], "llm_calls": [], "status": "error", "error": None}
        events, stderr = Queue(), deque(maxlen=80)
        process = subprocess.Popen([self.settings["sage_python"], str(self.worker), self.settings["sage_path"]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        def read_output():
            try:
                for line in process.stdout:
                    events.put(line)
            finally:
                events.put(None)

        def read_errors():
            for line in process.stderr:
                stderr.append(line.rstrip()[-2000:])

        readers = [Thread(target=read_output, daemon=True), Thread(target=read_errors, daemon=True)]
        for thread in readers:
            thread.start()
        deadline = time.monotonic() + self.settings["episode_timeout"]

        def send(data):
            process.stdin.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
            process.stdin.flush()

        try:
            send({"benchmark": episode.benchmark, "request": episode.request, "context": episode.context,
                  "tools": episode.tools, "settings": self.settings})
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("SAGE episode time budget exhausted.")
                try:
                    line = events.get(timeout=remaining)
                except Empty:
                    raise TimeoutError("SAGE episode time budget exhausted.") from None
                if line is None:
                    raise RuntimeError("SAGE worker exited without a result.")
                message = json.loads(line)
                kind = message["kind"]
                if kind == "result":
                    report.update(message["report"])
                    break
                if kind == "event":
                    report["trace"].append(message["event"])
                    continue
                if kind == "model":
                    if len(report["llm_calls"]) >= self.settings["max_llm_calls"]:
                        report.update(status="max_steps", error="SAGE total model-call budget exhausted.")
                        break
                    try:
                        text, call = self.client.generate(message["messages"], timeout=remaining, stop=message.get("stop"))
                    except BackendUnavailable as exc:
                        report["llm_calls"].append(exc.call)
                        report.update(status="timeout" if exc.episode_timeout else "backend_error",
                                      error=str(exc), backend_unavailable=not exc.episode_timeout)
                        break
                    report["llm_calls"].append(call)
                    send({"result": text})
                elif kind in {"tool", "reject"}:
                    trace = {key: value for key, value in message.items() if key != "kind"}
                    trace["type"] = kind
                    try:
                        if kind == "tool":
                            value = episode.invoke(message["name"], message["arguments"])
                        else:
                            value = {key: message[key] for key in ("request", "reason")}
                            if not all(isinstance(v, str) and v.strip() for v in value.values()):
                                raise ValueError("Rejection requires nonempty request and reason strings.")
                            if value not in report["rejected_requests"]:
                                report["rejected_requests"].append(value)
                        trace["observation"] = value
                        send({"result": value})
                    except Exception as exc:
                        trace["error"] = f"{type(exc).__name__}: {exc}"
                        send({"error": trace["error"]})
                    report["trace"].append(trace)
                else:
                    raise ValueError("Unknown SAGE bridge message: " + str(kind))
        except Exception as exc:
            report.update(status="timeout" if isinstance(exc, TimeoutError) else "error",
                          error=f"{type(exc).__name__}: {exc}", exception=type(exc).__name__,
                          technical_error=not isinstance(exc, TimeoutError))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for thread in readers:
                thread.join(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        report["worker_stderr_tail"] = list(stderr)[-20:]
        report["model_usage"] = summarize_calls(report["llm_calls"])
        return report
