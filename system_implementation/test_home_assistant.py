"""Switching devices through Home Assistant, against a small fake Home Assistant (no LLM or
hardware needed).

Run from system_implementation/:
    python -m unittest test_home_assistant
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError

import smart_home_langgraph as shl
from home_assistant import HomeAssistant

TOKEN = "test-token"
NOW = "2026-01-01T12:00:00+00:00"


class FakeHomeAssistant(BaseHTTPRequestHandler):
    """The two REST endpoints the module uses: read a state, call a turn_on/turn_off service."""
    states, stuck, calls = {}, set(), []

    def reply(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        entity = self.path.removeprefix("/api/states/")
        if self.headers["Authorization"] != f"Bearer {TOKEN}":
            return self.reply(401, {"message": "Unauthorized"})
        if entity not in self.states:
            return self.reply(404, {"message": "Entity not found."})
        self.reply(200, {"entity_id": entity, "state": self.states[entity]})

    def do_POST(self):
        service = self.path.removeprefix("/api/services/").split("/")[1]
        entity = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["entity_id"]
        self.calls.append((service, entity))
        if entity not in self.stuck:
            self.states[entity] = "on" if service == "turn_on" else "off"
        self.reply(200, [])

    def log_message(self, *args):
        pass


class HomeAssistantTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHomeAssistant)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeHomeAssistant.states = {"media_player.tv": "off", "light.ceiling": "unavailable"}
        FakeHomeAssistant.stuck, FakeHomeAssistant.calls = set(), []
        self.ha = HomeAssistant(self.url, TOKEN, {"tv_lr": "media_player.tv", "light_lr_ceiling": "light.ceiling"},
                                timeout=1)

    def run_tool(self, skill, device):
        """Set up a request as the CLI does and run one operation's tool on the simulator."""
        state = shl.create_initial_state(**shl.home_assistant_setup(self.ha))
        simulator = shl.SIMULATORS.pop(state["request_id"])
        agent = shl.HOME_DEFAULTS["devices"][device]["agent_id"]
        output = simulator.tools[(agent, skill)](skill, {"device_id": device}, simulator, NOW, {"_action_id": "t1"})
        return output, simulator.snapshot()

    def test_start_state_comes_from_home_assistant(self):
        FakeHomeAssistant.states["media_player.tv"] = "playing"
        state = shl.create_initial_state(**shl.home_assistant_setup(self.ha))
        shl.SIMULATORS.pop(state["request_id"])
        self.assertIs(state["home_state"]["device.tv_lr.power"]["value"], True)
        self.assertIsNone(state["home_state"]["device.light_lr_ceiling.power"]["value"])  # unavailable

    def test_turn_on_switches_the_real_device(self):
        output, home = self.run_tool("turn_on", "tv_lr")
        self.assertEqual(FakeHomeAssistant.calls, [("turn_on", "media_player.tv")])
        self.assertEqual(output, {"power": True})
        self.assertEqual(home["device.tv_lr.power"]["source"], "home_assistant")
        self.assertIs(home["device.tv_lr.power"]["value"], True)

    def test_turn_off_keeps_the_simulated_side_effects(self):
        FakeHomeAssistant.states["media_player.tv"] = "on"
        output, home = self.run_tool("turn_off", "tv_lr")
        self.assertEqual(FakeHomeAssistant.calls, [("turn_off", "media_player.tv")])
        self.assertIs(home["device.tv_lr.power"]["value"], False)
        self.assertIsNone(home["device.tv_lr.current_content"]["value"])
        self.assertIsNone(output["current_content"])

    def test_device_that_does_not_switch_fails_the_outcome_check(self):
        FakeHomeAssistant.stuck.add("media_player.tv")
        output, home = self.run_tool("turn_on", "tv_lr")
        self.assertIs(home["device.tv_lr.power"]["value"], False)  # what Home Assistant reports
        postcondition = {"state": "device.{device_id}.power", "equals": True}
        self.assertFalse(shl.check_condition(postcondition, {"device_id": "tv_lr"}, home, {"output": output}))

    def test_unmapped_devices_stay_simulated(self):
        _, home = self.run_tool("turn_on", "light_bedroom")
        self.assertEqual(FakeHomeAssistant.calls, [])
        self.assertEqual(home["device.light_bedroom.power"]["source"], "simulation:tool")

    def test_wrong_token_is_an_error(self):
        with self.assertRaises(HTTPError):
            HomeAssistant(self.url, "wrong", {"tv_lr": "media_player.tv"}).power_states()


if __name__ == "__main__":
    unittest.main()
