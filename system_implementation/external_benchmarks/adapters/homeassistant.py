"""Native pytest/Synthetic Home lifecycle and Assist evaluation."""
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from configuration import config_path as configured_path, load_config

from external_benchmarks.common import (Episode, cached_case, case_file, create_system,
    prepare_output, summarize_calls, system_report, write_json)


@pytest.fixture
def hass(verify_cleanup, hass):
    # Start upstream cleanup tracking before its autouse simulator fixtures create HA.
    return hass


@pytest.fixture
def synthetic_home_yaml(eval_task, allow_network):
    from home_assistant_datasets.plugins.pytest_data_loader import mock_synthetic_home_content
    return mock_synthetic_home_content.__wrapped__(eval_task)


@pytest.fixture(autouse=True)
def allow_network(socket_enabled):
    from home_assistant_datasets.plugins.pytest_agent import mock_allow_sockets
    yield from mock_allow_sockets.__wrapped__(socket_enabled)


def load(path=None, category="all", case_ids=(), limit=0):
    root = Path(path or os.getenv("HOMEASSISTANT_DATASET_PATH") or
                configured_path(load_config()["benchmarks"]["homeassistant"]["path"])).resolve()
    dataset = root if (root / "dataset_card.yaml").exists() else root / "datasets/assist"
    root = dataset.parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "home-assistant-synthetic-home"))
    from home_assistant_datasets.datasets.assist_data_loader import read_dataset_records
    from home_assistant_datasets.datasets.assist_eval_task import generate_assist_eval_tasks
    from home_assistant_datasets.datasets.dataset_card import read_dataset_card
    card = read_dataset_card(dataset)
    tasks = [task for record in read_dataset_records(card, set()) for task in generate_assist_eval_tasks(record)]
    counts = Counter(task.task_id for task in tasks)
    seen, cases = Counter(), []
    for task in tasks:
        seen[task.task_id] += 1
        # Identical sentences can have different requesting devices/setup in one record.
        case_id = task.task_id + (f"-{seen[task.task_id]}" if counts[task.task_id] > 1 else "")
        if (category == "all" or category in task.categories) and (not case_ids or case_id in case_ids):
            cases.append({"id": case_id, "task": task})
    cases = cases[:limit] if limit else cases
    if not cases:
        raise ValueError("No Home Assistant cases match the selected category/IDs.")
    return root, card, cases


class Session:
    def __init__(self, hass, api, entities, rooms, user_location, reference_time):
        self.hass, self.api, self.loop = hass, api, asyncio.get_running_loop()
        self.entities, self.rooms, self.user_location = entities, rooms, user_location
        self.reference_time, self.calls = reference_time, []
        self.timeout = load_config()["benchmarks"]["homeassistant"]["tool_timeout"]

    def wait(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=self.timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def episode(self, request):
        import probatio
        from homeassistant.helpers.json import JSONEncoder
        tools = [{"name": t.name, "description": t.description or t.name,
                  "parameters": probatio.to_json_schema(t.parameters, custom_serializer=self.api.custom_serializer)}
                 for t in self.api.tools]
        context = {"user_location": self.user_location, "reference_time": self.reference_time,
                   "rooms": self.rooms, "entities": self.entities, "states": await self.states(),
                   "tool_instructions": self.api.api_prompt}
        context = json.loads(json.dumps(context, cls=JSONEncoder))
        return Episode("homeassistant", request, context, tools, self.call, self)

    async def states(self):
        return {key: {"state": self.hass.states.get(key).state,
                      "attributes": dict(self.hass.states.get(key).attributes)} for key in self.entities}

    def snapshot(self):
        states = self.wait(self.states())
        return self.reference_time, {f"device.{key}.{field}": value
                                    for key, state in states.items() for field, value in state.items()}

    def read(self, target):
        return self.wait(self.states())[target]

    def call(self, name, arguments):
        from homeassistant.helpers.llm import ToolInput
        from homeassistant.helpers.json import JSONEncoder
        async def invoke():
            entry = {"event_type": "llm_tool_call", "data": {"tool_name": name, "tool_args": arguments}}
            self.calls.append(entry)
            try:
                result = await self.api.async_call_tool(ToolInput(tool_name=name, tool_args=arguments))
                # Native IntentResponseDict retains a live hass reference; checkpoints need plain JSON.
                data = json.loads(json.dumps(result.data, cls=JSONEncoder))
                entry["result"] = data
                if result.error or data.get("response_type") == "error":
                    raise ValueError(json.dumps(data))
                await self.hass.async_block_till_done()
                return data
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                raise
        return self.wait(invoke())


async def setup(hass, config_entry, task):
    from homeassistant.core import Context
    from homeassistant.helpers import area_registry as ar, device_registry as dr, entity_registry as er, llm
    from homeassistant.setup import async_setup_component
    from homeassistant.util import dt
    for component in ("intent", "conversation", "llm"):
        assert await async_setup_component(hass, component, {})
    areas, devices, entities = ar.async_get(hass), dr.async_get(hass), er.async_get(hass)
    rooms = {area.id: {"name": area.name, "floor_id": area.floor_id} for area in areas.async_list_areas()}
    requesting = None
    if task.action.context_device:
        requesting = next(device for device in dr.async_entries_for_config_entry(devices, config_entry.entry_id)
                          if ("synthetic_home", task.action.context_device) in device.identifiers)
    location = None if requesting is None else {"device_id": requesting.id, "name": requesting.name,
        "room_id": requesting.area_id, "room": rooms.get(requesting.area_id, {}).get("name")}
    inventory = {}
    for entity in er.async_entries_for_config_entry(entities, config_entry.entry_id):
        device = devices.async_get(entity.device_id) if entity.device_id else None
        inventory[entity.entity_id] = {"domain": entity.domain,
            "name": hass.states.get(entity.entity_id).name,
            "room_id": entity.area_id or (device.area_id if device else None)}
    api = await llm.async_get_api(hass, "assist", llm.LLMContext(platform="conversation", context=Context(),
        language="en", assistant="conversation", device_id=requesting.id if requesting else None))
    reference = (task.action.context_now or dt.now()).isoformat()
    return Session(hass, api, inventory, rooms, location, reference)


def evaluate(task, answer, unexpected, calls):
    from home_assistant_datasets.tool.assist.eval import test_eval
    output = SimpleNamespace(task=asdict(task.action), response=answer, task_id=task.task_id,
        context={"unexpected_states": unexpected, "conversation_trace": calls})
    checks = {}
    for name in ("expected_states", "expect_response", "expect_llm_tool_call_name", "expect_llm_tool_call_args"):
        try:
            getattr(test_eval, "test_" + name)(output)
            checks[name] = {"passed": True}
        except pytest.skip.Exception:
            continue
        except (AssertionError, ValueError) as exc:
            checks[name] = {"passed": False, "error": str(exc)}
    if task.action.expect_changes == {}:
        # Upstream skips this empty mapping; its stated ambiguity case requires no changes.
        checks["unchanged_states"] = {"passed": not unexpected}
    return {"success": all(check["passed"] for check in checks.values()) if checks else None, "checks": checks}


@pytest.mark.parametrize("expected_lingering_timers", [True])
@pytest.mark.parametrize("expected_lingering_tasks", [True])
async def test_assist_case(hass, synthetic_home_config_entry, entity_registry, eval_task, assist_runner, assist_case):
    """Native pytest entry point; case failures are saved as benchmark results."""
    from home_assistant_datasets.entity_state import EntityStateFixture
    from home_assistant_datasets.entity_state.diff import EntityStateDiffFixture
    difference = EntityStateDiffFixture(EntityStateFixture(hass, synthetic_home_config_entry, entity_registry))
    difference.prepare(eval_task.action.expect_changes or {}, eval_task.action.ignore_changes or {})
    session = await setup(hass, synthetic_home_config_entry, eval_task)
    episode = await session.episode(eval_task.input_text)
    report, error = {}, None
    def execute():
        nonlocal report, error
        report = system_report(assist_runner.system, episode)
        error = report.get("exception")
    started = time.perf_counter()
    with patch("homeassistant.util.dt.now", return_value=datetime.fromisoformat(session.reference_time)):
        await asyncio.to_thread(execute)
        await hass.async_block_till_done()
    row = {"id": assist_case["id"], "native_task_id": eval_task.task_id, "category": eval_task.categories,
        "input": eval_task.input_text, "answer": report.get("answer", ""), "report": report,
        "error": error or report.get("error"), "exception": error, "latency_s": time.perf_counter() - started,
        "reference_time": session.reference_time, "user_location": session.user_location,
        "expected": asdict(eval_task.action), "native_calls": session.calls,
        "model_usage": report.get("model_usage", summarize_calls([]))}
    try:
        row["unexpected_states"] = difference.get_unexpected_changes()
        row["score"] = evaluate(eval_task, row["answer"], row["unexpected_states"], session.calls)
    except Exception as exc:
        row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        row["score"] = {"success": None, "checks": {}}
    if row["error"]:
        row["score"]["success"] = False
    row["completed"] = not (error or row.get("evaluation_error") or report.get("backend_unavailable"))
    assist_runner.save(row)
    if report.get("backend_unavailable"):
        pytest.exit("Model backend unavailable; case saved, configure endpoint and --resume.", returncode=2)


def summarize(rows):
    scored = [row for row in rows if row["score"]["success"] is not None]
    return {"cases": len(rows), "evaluated_cases": len(scored),
        "success_rate": sum(row["score"]["success"] for row in scored) / len(scored) if scored else None,
        "errors": sum(bool(row.get("error")) for row in rows),
        "evaluation_errors": sum(bool(row.get("evaluation_error")) for row in rows),
        "mean_latency_s": sum(row["latency_s"] for row in rows) / len(rows) if rows else 0,
        **summarize_calls([call for row in rows for call in row.get("report", {}).get("llm_calls", [])])}


def run(path=None, *, category="all", limit=0, case_ids=(), output_dir=None, settings=None, system=None, config_path=None,
        resume=False):
    system = create_system(system, settings, config_path)
    root, card, cases = load(path, category, case_ids, limit)
    sources = set(card.dataset_files) | {card.dataset_path / "dataset_card.yaml"}
    sources |= {case["task"].record_source.record_path.parent / "_fixtures.yaml" for case in cases}
    metadata = {"benchmark": "homeassistant", "dataset": card.name, "dataset_version": card.version,
        "system": system.metadata(), "case_ids": [case["id"] for case in cases], "repetitions_per_sentence": 1,
        "native_default_repetitions": card.count,
        "tool_timeout": load_config()["benchmarks"]["homeassistant"]["tool_timeout"],
        "runtime_versions": {name: version(name) for name in ("homeassistant", "synthetic-home", "pytest-homeassistant-custom-component")},
        "data_sha256": {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(sources)},
        "native_source_sha256": {str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest()
            for folder in (root / "home_assistant_datasets", root / "home-assistant-synthetic-home/custom_components")
            for file in sorted(folder.rglob("*.py"))}}
    output = prepare_output(Path(output_dir or f"logs/homeassistant/{system.name}").resolve(), metadata, resume)
    rows = []
    selected_count = len(cases)
    pending = []
    for case in cases:
        cached = cached_case(output, case["id"], resume)
        if cached is not None:
            rows.append(cached)
        else:
            pending.append(case)
    cases = pending
    class AssistRunner:
        @pytest.fixture(params=cases, ids=[case["id"] for case in cases])
        def assist_case(self, request):
            return request.param

        @pytest.fixture
        def eval_task(self, assist_case):
            return assist_case["task"]

        @pytest.fixture
        def assist_runner(self):
            return self

        def save(self, row):
            rows.append(row)
            write_json(case_file(output, row["id"]), row)
            write_json(output / "summary.json", {"overall": summarize(rows),
                "complete": len(rows) == selected_count and all(r["completed"] for r in rows)})
            print(f"[{len(rows)}/{selected_count}] {row['id']}: success={row['score']['success']}, {row['latency_s']:.2f}s"
                  + (f" ERROR: {row['error']}" if row.get("error") else "")
                  + (f" EVALUATION ERROR: {row['evaluation_error']}" if row.get("evaluation_error") else ""), flush=True)

    runner = AssistRunner()
    runner.system = system
    result = pytest.main([str(Path(__file__).resolve()), "-q", "-s", "--import-mode=importlib", "--timeout=0",
        "-o", "asyncio_mode=auto", "-o", "log_level=WARNING",
        "-o", "markers=eval_model_outputs: upstream output evaluation",
        "-p", "home_assistant_datasets.plugins.pytest_synthetic_home"], plugins=[runner]) if cases else 0
    summary = {"overall": summarize(rows), "by_category": {name: summarize([row for row in rows if name in row["category"]])
        for name in sorted({name for row in rows for name in row["category"]})}}
    summary["complete"] = len(rows) == selected_count and all(r["completed"] for r in rows)
    write_json(output / "summary.json", summary)
    if result or len(rows) != selected_count:
        raise RuntimeError(f"Home Assistant test environment failed (pytest={result}); see console.log.")
    return summary
