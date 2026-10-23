import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from agent_registry import AGENT_REGISTRY

from configuration import CONFIG_PATH, load_config
from model_client import summarize_calls

load_dotenv(Path(__file__).with_name(".env"))
_models = {}
MODEL_SETTINGS = {}


def configure_models(config_path=None):
    global DEFAULT_MODEL, SAME_LLM_FOR_ALL_AGENTS, ROLE_MODELS, MODEL_SETTINGS
    data = load_config(config_path)
    defaults = load_config(CONFIG_PATH)
    shared = {**defaults["model"], **(data.get("model", {}) if isinstance(data.get("model"), dict) else {})}
    roles = {**defaults["mas_models"], **data.get("mas_models", {})}
    shared["model"] = os.getenv("LLM_MODEL") or shared["model"]
    if {"model": shared, "roles": roles} != MODEL_SETTINGS:
        _models.clear()
    MODEL_SETTINGS = {"model": shared, "roles": roles}
    DEFAULT_MODEL = shared["model"]
    SAME_LLM_FOR_ALL_AGENTS = roles["same_for_all"]
    ROLE_MODELS = dict.fromkeys(["intent_analysis", "planner", "scheduler", *AGENT_REGISTRY], DEFAULT_MODEL)
    ROLE_MODELS.update(roles.get("roles", {}))


configure_models()

API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

def model_for_role(role=None):
    return DEFAULT_MODEL if SAME_LLM_FOR_ALL_AGENTS else ROLE_MODELS.get(role, DEFAULT_MODEL)


def get_llm(role=None):
    """Reuse one model client per model ID."""
    model = model_for_role(role)
    if model in _models:
        return _models[model]
    provider = model.split(":", 1)[0]
    cfg = MODEL_SETTINGS["model"]
    kwargs = {"temperature": cfg["temperature"]}

    key_var = API_KEY_VARS.get(provider)
    if key_var and not os.getenv(key_var):
        raise RuntimeError(f"LLM_MODEL={model} needs {key_var} to be set (e.g. in .env).")
    if provider == "ollama":
        kwargs.update(format="json", seed=cfg["seed"],
                      num_predict=cfg["max_tokens"], num_ctx=cfg["context_window"],
                      client_kwargs={"timeout": cfg["request_timeout"]})
        if cfg.get("think") is not None:  # Thinking models only; omitted keeps the model's default.
            kwargs["reasoning"] = cfg["think"]
        base = os.getenv("OLLAMA_BASE_URL") or cfg["api_base"]
        if base:
            kwargs["base_url"] = base
    else:
        kwargs.update(max_tokens=cfg["max_tokens"], timeout=cfg["request_timeout"])
        if provider in API_KEY_VARS:
            kwargs["max_retries"] = 0
        if provider == "openai":
            kwargs["seed"] = cfg["seed"]
            base = os.getenv("OPENAI_BASE_URL") or cfg["api_base"]
            if base:
                kwargs["base_url"] = base
    _models[model] = init_chat_model(model, **kwargs)
    return _models[model]
