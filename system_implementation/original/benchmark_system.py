"""DEXA entry point around the unchanged original. No invented device executor."""
import importlib.util
from pathlib import Path
from uuid import uuid4
from langgraph.types import Command
from llm_config import get_llm

SUPPORTED_BENCHMARKS = {"dexa"}
_spec = importlib.util.spec_from_file_location("_original_source", Path(__file__).with_name("smart_home_langgraph.py"))
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)


def run_request(text, **settings):
    # Shared model configuration for the architecture comparison. Original prompts,
    # routing and text outputs are unchanged; the source still contains gemma2.
    if settings:
        raise ValueError("The original does not implement extended-system switches.")
    _original.llm = get_llm()
    config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 200}
    initial = {"messages": [], "task_queue": [], "collaboration_request": {}, "task_history": []}
    yield {"experiment_config": {"settings": {"system": "original", "recursion_limit": 200}}}
    try:
        for _ in _original.graph.stream(initial, config):
            pass
        yield from _original.graph.stream(Command(resume=text), config)
    finally:
        _original.graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
