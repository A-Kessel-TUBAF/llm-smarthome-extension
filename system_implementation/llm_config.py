import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from agent_registry import AGENT_REGISTRY

# Reads API keys and settings from a .env file in this folder (if present)
load_dotenv()

DEFAULT_MODEL = os.getenv("LLM_MODEL") or "ollama:gemma2"
SAME_LLM_FOR_ALL_AGENTS = True
ROLE_MODELS = dict.fromkeys(
    ["intent_analysis", "planner", "scheduler", *AGENT_REGISTRY],
    DEFAULT_MODEL,
)
# For individual models: set SAME_LLM_FOR_ALL_AGENTS = False and override entries:
# ROLE_MODELS["planner"] = "openai:gpt-4o-mini"
# ROLE_MODELS["lighting"] = "ollama:gemma2"
_models = {}  # Reuse one client per model; initialize only when needed.

API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
SETUP_HINT = (
    "Either install Ollama (https://ollama.com/download) and run `ollama pull gemma2`, "
    "or use an online API: set LLM_MODEL and its API key in system_implementation/.env " 
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
    """
    Build the chat model from env vars, so local and online models are interchangeable.
    LLM_MODEL = "provider:model", e.g.
        ollama:gemma2                  default, local Ollama
        openai:gpt-4o-mini             needs OPENAI_API_KEY
        anthropic:claude-sonnet-5      needs ANTHROPIC_API_KEY
    OPENAI_BASE_URL = optional, for OpenAI-compatible services 
    (GWDG Chat AI, OpenRouter, Groq, vLLM, ...)
    """
    model = model_for_role(role)
    if model in _models:
        return _models[model]
    provider = model.split(":", 1)[0]
    kwargs = {"temperature": 0.0}  # fixed for repeatable benchmark runs

    # Fail at startup with a clear message, instead of at the first LLM call
    key_var = API_KEY_VARS.get(provider)
    if key_var and not os.getenv(key_var):
        raise RuntimeError(f"LLM_MODEL={model} needs {key_var} to be set (e.g. in .env).")
    if provider == "ollama":
        kwargs["format"] = "json"  # All roles in the current system return JSON.
        kwargs["validate_model_on_init"] = True  # checks Ollama is running and the model is pulled        
        try:
            _models[model] = init_chat_model(model, **kwargs)
            return _models[model]
        except (ValueError, ConnectionError) as e:
            raise RuntimeError(f"Local model '{model}' is not available.\n{e}\n\n{SETUP_HINT}") from None
    _models[model] = init_chat_model(model, **kwargs)
    return _models[model]