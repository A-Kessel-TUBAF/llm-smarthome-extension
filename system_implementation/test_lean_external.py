"""Lean admission on external-benchmark homes: per-device limits and omitted optional values
(no LLM needed).

External benchmarks describe parameters per device (HomeBench, SimuHome) and mark optional values
as nullable (Home Assistant, SimuHome). The arguments are passed to `check_allocation` directly,
without the JSON-schema validation of `validate_action`, to show what the checker itself rejects.

Run from system_implementation/ after building the checker (cd lean && lake build):
    python -m unittest test_lean_external
"""

import sys
import time
import unittest
from pathlib import Path

import smart_home_langgraph as shl

sys.path.insert(0, str(Path(__file__).parent / "external_benchmarks"))
from agent_registry_homebench import build_home  # noqa: E402

EXE = Path(__file__).parent / "lean/.lake/build/bin/allocation_checker"

HOMEBENCH_HOME = {
    "method": [
        {"room_name": "living_room", "device_name": "light", "operation": "set_brightness",
         "parameters": [{"name": "brightness", "type": "int"}]},
        {"room_name": "bedroom", "device_name": "light", "operation": "set_brightness",
         "parameters": [{"name": "brightness", "type": "int"}]},
        {"room_name": "living_room", "device_name": "fan", "operation": "set_mode",
         "parameters": [{"name": "mode", "type": "str"}]}],
    "home_status": {
        "living_room": {
            "light": {"state": "on", "attributes": {"brightness": {"value": 50, "lowest": 0, "highest": 100}}},
            "fan": {"state": "on", "attributes": {"mode": {"value": "low", "options": ["low", "high"]}}}},
        "bedroom": {
            "light": {"state": "on", "attributes": {"brightness": {"value": 20, "lowest": 0, "highest": 50}}}}}}

OPTIONAL = {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 100}, {"type": "null"}], "default": None}
NULLABLE = {"type": ["integer", "null"], "default": None}
SET_LIGHT = {"service_id": {"type": "string"}, "name": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
             "brightness": OPTIONAL, "transition": NULLABLE}
OPTIONAL_REGISTRY = {"lighting": {
    "agent": "lighting_agent", "enabled": True, "description": "Lights.", "capabilities": ["set_light"],
    "operations": {"set_light": {"description": "Set lights.", "parameters": SET_LIGHT, "output": {},
                                 "reads": [], "writes": ["service.{service_id}.state"],
                                 "preconditions": [], "postconditions": [], "max_retries": 0}}}}
OPTIONAL_HOME = {"rooms": {}, "devices": {}, "services": {
    "light": {"agent_id": "lighting", "room_id": None, "type": "light", "skills": ["set_light"],
              "parameters": {"set_light": SET_LIGHT}}}}


def action(agent, skill, **args):
    return dict(task_id="a1", agent_id=agent, skill_id=skill, arguments=args, depends_on=[],
                status="running", dispatch_until=time.monotonic() + 30)


@unittest.skipUnless(EXE.is_file(), "build the Lean checker first: cd lean && lake build")
class LeanExternalTests(unittest.TestCase):

    def check(self, home_config, registry, batch):
        state = shl.create_initial_state(home_config=home_config, agent_registry=registry,
                                         checker="lean", lean_executable=EXE)
        try:
            shl.check_allocation(state, batch)
        finally:
            shl.SIMULATORS.pop(state["request_id"], None)

    def homebench(self, batch):
        self.check(*build_home(HOMEBENCH_HOME), batch)

    def optional(self, batch):
        self.check(OPTIONAL_HOME, OPTIONAL_REGISTRY, batch)

    def brightness(self, device, value):
        return action("lighting", "set_brightness", device_id=device, brightness=value)

    def test_device_range_is_checked(self):
        self.homebench([self.brightness("living_room__light", 40)])
        for value in (150, -1, "high"):
            with self.assertRaises(ValueError, msg=value):
                self.homebench([self.brightness("living_room__light", value)])

    def test_each_device_has_its_own_range(self):
        self.homebench([self.brightness("bedroom__light", 40)])
        with self.assertRaises(ValueError):
            self.homebench([self.brightness("bedroom__light", 60)])  # within 0-100, but this light allows 0-50

    def test_device_options_are_checked(self):
        self.homebench([action("air_quality", "set_mode", device_id="living_room__fan", mode="high")])
        with self.assertRaises(ValueError):
            self.homebench([action("air_quality", "set_mode", device_id="living_room__fan", mode="medium")])

    def test_omitted_optional_values_are_allowed(self):
        self.optional([action("lighting", "set_light", service_id="light")])
        self.optional([action("lighting", "set_light", service_id="light", name="Kitchen", brightness=None)])

    def test_given_optional_values_are_checked(self):
        self.optional([action("lighting", "set_light", service_id="light", brightness=40, transition=2)])
        for args in ({"brightness": 150}, {"brightness": "dim"}, {"transition": "slow"}):
            with self.assertRaises(ValueError, msg=args):
                self.optional([action("lighting", "set_light", service_id="light", **args)])


if __name__ == "__main__":
    unittest.main()
