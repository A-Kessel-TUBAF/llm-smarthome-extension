"""Shared model HTTP transport, retries and usage; no agent or device execution."""
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from configuration import CONFIG_PATH, load_config, system_settings


def summarize_calls(calls):
    usage = {"llm_calls": len(calls), "llm_call_errors": sum(c.get("error") is not None for c in calls),
             "llm_attempts": sum(c.get("attempts", 1) for c in calls),
             "llm_retries": sum(len(c.get("retries", [])) for c in calls),
             "calls_with_usage": sum(c.get("input_tokens") is not None and c.get("output_tokens") is not None for c in calls)}
    for key in ("input_tokens", "output_tokens"):
        values = [c[key] for c in calls if c.get(key) is not None]
        usage[key] = sum(values) if values or not calls else None
    usage["model_call_latency_s"] = sum(c.get("latency_s", 0) for c in calls)
    return usage


def retry_model_call(request, *, call, retry_delays=None, deadline=None):
    """Retry only the same model request on HTTP 500; never an agent step or tool."""
    delays = retry_delays
    if delays is None:
        delays = load_config()["model"].get("retry_delays", load_config(CONFIG_PATH)["model"]["retry_delays"])
    if any(not math.isfinite(delay) or delay < 0 for delay in delays):
        raise ValueError("Model retry delays must be finite and nonnegative.")
    call.update(attempts=0, retries=[])
    for attempt in range(len(delays) + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Model episode time budget exhausted.")
        call["attempts"] += 1
        try:
            return request()
        except Exception as exc:
            if isinstance(exc, HTTPError):
                exc.close()
            error, status = exc, None
            while error is not None and status is None:
                status = getattr(error, "status_code", getattr(error, "code", None))
                if status is None:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                error = error.__cause__
            if status != 500:
                raise
            if attempt == len(delays):
                error = BackendUnavailable(f"Model endpoint returned HTTP 500 after {call['attempts']} attempts.")
                error.call, error.status_code = call, status
                raise error from exc
            delay = delays[attempt]
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise TimeoutError("Model episode time budget exhausted before retry.") from exc
            call["retries"].append({"http_status": status, "delay_s": delay})
            time.sleep(delay)


class BackendUnavailable(RuntimeError):
    episode_timeout = False


class ChatClient:
    """Shared HTTP transport and model settings; no agent decisions."""
    def __init__(self, settings=None, env_prefix="REACT", config_path=None):
        defaults = system_settings(env_prefix.lower())
        cfg = system_settings(env_prefix.lower(), config_path)
        cfg["model"] = os.getenv(env_prefix + "_MODEL") or os.getenv("LLM_MODEL") or cfg["model"]
        cfg["api_base"] = os.getenv(env_prefix + "_API_BASE") or cfg["api_base"]
        cfg.update(settings or {})
        unknown = set(cfg) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {env_prefix} settings: " + ", ".join(sorted(unknown)))
        for name in ("max_steps", "max_llm_calls", "max_tokens", "context_window"):
            if name in cfg and (type(cfg[name]) is not int or cfg[name] <= 0):
                raise ValueError(name + " must be a positive integer.")
        for name in ("request_timeout", "episode_timeout", "temperature"):
            if not math.isfinite(cfg[name]) or cfg[name] < 0 or (name != "temperature" and not cfg[name]):
                raise ValueError(name + " must be finite and within its allowed range.")
        if type(cfg["seed"]) is not int or ("json_mode" in cfg and type(cfg["json_mode"]) is not bool):
            raise ValueError("seed must be an integer and json_mode a boolean.")
        self.settings = cfg
        self.model = cfg["model"]
        self.provider, separator, self.model_id = self.model.partition(":")
        if not separator or self.provider not in {"ollama", "openai"} or not self.model_id:
            raise ValueError("Model must be ollama:<model> or openai:<model> (also for compatible servers).")
        default = (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") if self.provider == "ollama"
                   else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        self.base = (cfg["api_base"] or default).rstrip("/")
        parsed = urlsplit(self.base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("api_base must be an HTTP(S) URL without credentials or query parameters.")
        self.key = os.getenv(env_prefix + "_API_KEY") or (os.getenv("OPENAI_API_KEY") if self.provider == "openai" else None)

    def metadata(self):
        return {"model": self.model, "api_base": self.base, "credentials": "environment only"}

    def generate(self, messages, timeout=None, stop=None):
        cfg = self.settings
        if self.provider == "ollama":
            path = "/api/chat"
            body = {"model": self.model_id, "messages": messages, "stream": False,
                    "options": {"temperature": cfg["temperature"], "seed": cfg["seed"],
                                "num_predict": cfg["max_tokens"], "num_ctx": cfg["context_window"]}}
            if cfg.get("json_mode", False):
                body["format"] = "json"
            if stop:
                body["options"]["stop"] = stop
        else:
            path = "/chat/completions"
            body = {"model": self.model_id, "messages": messages, "temperature": cfg["temperature"],
                    "max_tokens": cfg["max_tokens"], "seed": cfg["seed"]}
            if stop:
                body["stop"] = stop
            if cfg.get("json_mode", False):
                body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        call = {"model": self.model, "input_tokens": None, "output_tokens": None, "error": None}
        started = time.perf_counter()
        deadline = time.monotonic() + timeout if timeout is not None else None
        request_timeout = cfg["request_timeout"]
        try:
            request = Request(self.base + path, data=json.dumps(body, allow_nan=False).encode(), headers=headers)
            def send():
                nonlocal request_timeout
                request_timeout = min(cfg["request_timeout"], deadline - time.monotonic()) if deadline is not None else cfg["request_timeout"]
                with urlopen(request, timeout=request_timeout) as response:
                    return json.load(response)
            data = retry_model_call(send, call=call, retry_delays=cfg["retry_delays"], deadline=deadline)
            if self.provider == "ollama":
                content = data["message"]["content"]
                call.update(input_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"))
            else:
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                call.update(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
            if not isinstance(content, str):
                raise ValueError("Model response has no text content.")
            call["raw_response"] = content
            return content, call
        except (BackendUnavailable, HTTPError, URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            # Do not include server error bodies, request headers or API keys in logs.
            status = getattr(exc, "status_code", getattr(exc, "code", None))
            reason = f"HTTP {status}" if status is not None else type(exc).__name__
            call["error"] = f"Model endpoint unavailable or invalid response ({reason})."
            error = BackendUnavailable(call["error"])
            error.call = call
            error.episode_timeout = (deadline is not None and (request_timeout < cfg["request_timeout"] or time.monotonic() >= deadline) and
                (isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)))
            raise error from None
        finally:
            call["latency_s"] = time.perf_counter() - started

