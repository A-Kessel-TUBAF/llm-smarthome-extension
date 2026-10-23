"""The optional [model] think setting reaches Ollama (mocked; no model or server needed).

Run from system_implementation/:
    python -m unittest test_model_settings
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import llm_config
from model_client import ChatClient

DGX = Path(__file__).with_name("experiment_config.dgx.toml")


class ThinkSettingTests(unittest.TestCase):

    def test_mas_passes_think_to_chat_ollama_only_when_configured(self):
        try:
            for config, expected in ((DGX, False), (None, "absent")):
                with self.subTest(config=config), patch.object(llm_config, "init_chat_model") as init:
                    llm_config.configure_models(config)
                    llm_config._models.clear()
                    llm_config.get_llm("planner")
                    self.assertEqual(init.call_args.kwargs.get("reasoning", "absent"), expected)
        finally:
            llm_config.configure_models()
            llm_config._models.clear()

    def test_react_sends_think_only_when_configured(self):
        for config, expected in ((DGX, False), (None, "absent")):
            with self.subTest(config=config), patch("model_client.urlopen",
                                                    side_effect=RuntimeError("request captured")) as urlopen:
                with self.assertRaises(Exception):
                    ChatClient(config_path=config).generate([{"role": "user", "content": "hi"}])
                self.assertEqual(json.loads(urlopen.call_args.args[0].data).get("think", "absent"), expected)

    def test_react_rejects_non_boolean_think(self):
        with self.assertRaises(ValueError):
            ChatClient({"think": "no"}, config_path=DGX)


if __name__ == "__main__":
    unittest.main()
