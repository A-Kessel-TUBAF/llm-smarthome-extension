import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Reads API keys and settings from a .env file in this folder (if present)
load_dotenv()

API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
SETUP_HINT = (
    "Either install Ollama (https://ollama.com/download) and run `ollama pull gemma2`, "
    "or use an online API: set LLM_MODEL and its API key in system_implementation/.env " 
)

def get_llm():
    """
    Build the chat model from env vars, so local and online models are interchangeable.
    LLM_MODEL = "provider:model", e.g.
        ollama:gemma2                  default, local Ollama
        openai:gpt-4o-mini             needs OPENAI_API_KEY
        anthropic:claude-sonnet-5      needs ANTHROPIC_API_KEY
    OPENAI_BASE_URL = optional, for OpenAI-compatible services 
    (GWDG Chat AI, OpenRouter, Groq, vLLM, ...)
    """
    model = os.getenv("LLM_MODEL") or "ollama:gemma2"
    provider = model.split(":", 1)[0]
    kwargs = {"temperature": 0.0}  # fixed for repeatable benchmark runs

    # Fail at startup with a clear message, instead of at the first LLM call
    key_var = API_KEY_VARS.get(provider)
    if key_var and not os.getenv(key_var):
        raise RuntimeError(f"LLM_MODEL={model} needs {key_var} to be set (e.g. in .env).")
    if provider == "ollama":
        kwargs["validate_model_on_init"] = True  # checks Ollama is running and the model is pulled        
        try:
            return init_chat_model(model, **kwargs)
        except (ValueError, ConnectionError) as e:
            raise RuntimeError(f"Local model '{model}' is not available.\n{e}\n\n{SETUP_HINT}") from None
    return init_chat_model(model, **kwargs)