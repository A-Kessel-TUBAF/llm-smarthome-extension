import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from agent_registry import AGENT_REGISTRY

from configuration import CONFIG_PATH, load_config

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
SETUP_HINT = (
    "Either install Ollama (https://ollama.com/download) and run `ollama pull gemma2`, "
    "or configure model.model in experiment_config.toml and set the provider API key. " 
)

def model_for_role(role=None):
    return DEFAULT_MODEL if SAME_LLM_FOR_ALL_AGENTS else ROLE_MODELS.get(role, DEFAULT_MODEL)


def summarize_calls(calls):
    usage = {"llm_calls": len(calls), "llm_call_errors": sum(c.get("error") is not None for c in calls),
             "calls_with_usage": sum(c.get("input_tokens") is not None and c.get("output_tokens") is not None for c in calls)}
    for key in ("input_tokens", "output_tokens"):
        values = [c[key] for c in calls if c.get(key) is not None]
        usage[key] = sum(values) if values or not calls else None
    usage["model_call_latency_s"] = sum(c["latency_s"] for c in calls)
    return usage


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
        kwargs.update(format="json", validate_model_on_init=True, seed=cfg["seed"],
                      num_predict=cfg["max_tokens"], num_ctx=cfg["context_window"],
                      client_kwargs={"timeout": cfg["request_timeout"]})
        base = os.getenv("OLLAMA_BASE_URL") or cfg["api_base"]
        if base:
            kwargs["base_url"] = base
        try:
            _models[model] = init_chat_model(model, **kwargs)
            return _models[model]
        except (ValueError, ConnectionError) as e:
            raise RuntimeError(f"Local model '{model}' is not available.\n{e}\n\n{SETUP_HINT}") from None
    kwargs.update(max_tokens=cfg["max_tokens"], timeout=cfg["request_timeout"])
    if provider == "openai":
        kwargs["seed"] = cfg["seed"]
        base = os.getenv("OPENAI_BASE_URL") or cfg["api_base"]
        if base:
            kwargs["base_url"] = base
    _models[model] = init_chat_model(model, **kwargs)
    return _models[model]
