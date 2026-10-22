"""Lean admission on external-benchmark homes: per-device limits, omitted optional values and
whole-home commands (no LLM needed).

External benchmarks describe parameters per device (HomeBench, SimuHome), mark optional values as
nullable (Home Assistant, SimuHome), and Home Assistant's tools may change entities other than their
target. The arguments are passed to `check_allocation` directly, without the JSON-schema validation
of `validate_action`, to show what the checker itself rejects.

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


GET_STATE = {"description": "Read.", "parameters": {"device_id": {"type": "string"}}, "output": {},
             "reads": ["device.{device_id}.state", "device.{device_id}.attributes"], "writes": [],
             "preconditions": [], "postconditions": [], "max_retries": 0}
TURN_ON = {"service_id": {"type": "string"}, "name": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}}
SET_BRIGHTNESS = {"service_id": {"type": "string"}, "brightness": {"type": "integer", "minimum": 0, "maximum": 100}}
SET_TEMPERATURE = {"service_id": {"type": "string"}, "temperature": {"type": "number", "minimum": 5, "maximum": 35}}


def home_assistant_fixture(lights=("kitchen", "living")):
    """Shaped like external_benchmarks/agent_registry_homeassistant.py: per-entity get_state, a name/area
    tool that may change every entity, and domain tools that change every entity of their domain."""
    entities = {**{f"light.{name}": "lighting" for name in lights}, "climate.hall": "thermostat"}
    keys = lambda domain: [f"device.{e}.{f}" for e in entities if e.startswith(domain) for f in ("state", "attributes")]
    tool = lambda parameters, affected: {"description": "Tool.", "parameters": parameters, "output": {},
                                         "reads": affected, "writes": affected, "preconditions": [],
                                         "postconditions": [], "max_retries": 0}
    tools = {"light": {"HassTurnOn": tool(TURN_ON, keys("")), "HassLightSet": tool(SET_BRIGHTNESS, keys("light."))},
             "climate": {"HassClimateSetTemperature": tool(SET_TEMPERATURE, keys("climate."))}}
    agents = {"light": "lighting", "climate": "thermostat"}
    registry = {agents[domain]: {"agent": agents[domain] + "_agent", "enabled": True, "description": domain,
                                 "capabilities": ["get_state", *ops], "operations": {"get_state": GET_STATE, **ops}}
                for domain, ops in tools.items()}
    home = {"rooms": {}, "devices": {e: {"agent_id": a, "room_id": None, "type": e.split(".")[0], "skills": ["get_state"]}
                                    for e, a in entities.items()},
            "services": {domain: {"agent_id": agents[domain], "room_id": None, "type": domain, "skills": list(ops),
                                  "parameters": {name: op["parameters"] for name, op in ops.items()}}
                         for domain, ops in tools.items()}}
    return home, registry


HA_HOME, HA_REGISTRY = home_assistant_fixture()


# Shaped like external_benchmarks/agent_registry_simuhome.py: a settable enumeration attribute
# (SimuHome's SystemMode: 0=OFF, 1=AUTO, 3=COOL, 4=HEAT) with the optional scheduling time.
SYSTEM_MODE = {"device_id": {"type": "string"}, "value": {"type": "integer", "enum": [0, 1, 3, 4]},
               "start_time": {"type": ["string", "null"], "default": None}}
SIMUHOME_REGISTRY = {"thermostat": {"agent": "thermostat_agent", "enabled": True, "description": "Climate.",
                                    "capabilities": ["1.Thermostat.write_SystemMode"], "operations": {
                                        "1.Thermostat.write_SystemMode": {
                                            "description": "Write.", "parameters": SYSTEM_MODE, "output": {},
                                            "reads": [], "writes": ["device.{device_id}.attributes"],
                                            "preconditions": [], "postconditions": [], "max_retries": 0}}}}
SIMUHOME_HOME = {"rooms": {}, "services": {}, "devices": {"ac_1": {
    "agent_id": "thermostat", "room_id": None, "type": "air_conditioner", "skills": ["1.Thermostat.write_SystemMode"],
    "parameters": {"1.Thermostat.write_SystemMode": SYSTEM_MODE}}}}


def action(agent, skill, task_id="a1", **args):
    return dict(task_id=task_id, agent_id=agent, skill_id=skill, arguments=args, depends_on=[],
                status="running", dispatch_until=time.monotonic() + 30)


def turn_on(task_id="a1"):
    return action("lighting", "HassTurnOn", task_id, service_id="light", name="Kitchen")


def light_set(value, task_id="a1"):
    return action("lighting", "HassLightSet", task_id, service_id="light", brightness=value)


def set_temperature(value, task_id="a1"):
    return action("thermostat", "HassClimateSetTemperature", task_id, service_id="climate", temperature=value)


def get_state(entity, task_id="a1"):
    return action("lighting" if entity.startswith("light.") else "thermostat", "get_state", task_id, device_id=entity)


@unittest.skipUnless(EXE.is_file(), "build the Lean checker first: cd lean && lake build")
class LeanExternalTests(unittest.TestCase):

    def check(self, home_config, registry, batch, action_graph=None, **setup):
        state = shl.create_initial_state(home_config=home_config, agent_registry=registry,
                                         checker="lean", lean_executable=EXE, **setup)
        state["action_graph"] = action_graph or {}
        try:
            shl.check_allocation(state, batch)
        finally:
            shl.SIMULATORS.pop(state["request_id"], None)

    def homebench(self, batch):
        self.check(*build_home(HOMEBENCH_HOME), batch)

    def optional(self, batch):
        self.check(OPTIONAL_HOME, OPTIONAL_REGISTRY, batch)

    def home_assistant(self, batch):
        self.check(HA_HOME, HA_REGISTRY, batch)

    def simuhome(self, batch):
        self.check(SIMUHOME_HOME, SIMUHOME_REGISTRY, batch)

    def system_mode(self, value):
        return action("thermostat", "1.Thermostat.write_SystemMode", device_id="ac_1", value=value)

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

    def test_whole_home_commands_alone_are_accepted(self):
        self.home_assistant([turn_on()])
        self.home_assistant([set_temperature(21)])

    def test_whole_home_command_conflicts_with_other_writers(self):
        # Turning on "Kitchen" may change any entity, including the thermostat.
        with self.assertRaises(ValueError):
            self.home_assistant([turn_on("a1"), set_temperature(21, "a2")])

    def test_whole_home_command_conflicts_with_readers(self):
        with self.assertRaises(ValueError):
            self.home_assistant([turn_on("a1"), get_state("light.kitchen", "a2")])

    def test_independent_reads_and_domain_writes_stay_parallel(self):
        self.home_assistant([get_state("light.kitchen", "a1"), get_state("climate.hall", "a2")])
        self.home_assistant([set_temperature(21, "a1"), get_state("light.kitchen", "a2")])
        with self.assertRaises(ValueError):  # reads what the thermostat command writes
            self.home_assistant([set_temperature(21, "a1"), get_state("climate.hall", "a2")])

    def test_domain_commands_of_different_domains_stay_parallel(self):
        self.home_assistant([set_temperature(21, "a1"), light_set(50, "a2")])
        with self.assertRaises(ValueError):  # "Kitchen" may be any entity, including the lights
            self.home_assistant([turn_on("a1"), light_set(50, "a2")])

    def test_large_homes_fit_leans_key_lists(self):
        # Lean lists at most 64 keys per operation; forty lights and a thermostat have 82.
        home, registry = home_assistant_fixture([f"l{i}" for i in range(40)])
        self.check(home, registry, [turn_on("a1")])
        self.check(home, registry, [set_temperature(21, "a1"), get_state("light.l7", "a2")])
        with self.assertRaises(ValueError):
            self.check(home, registry, [turn_on("a1"), get_state("light.l7", "a2")])

    def test_state_stays_readable_after_whole_home_commands(self):
        # A finished whole-home command must not hide an entity's state from later preconditions.
        home, registry = home_assistant_fixture()
        registry["lighting"]["operations"]["get_state"] = {
            **GET_STATE, "preconditions": [{"state": "device.{device_id}.state", "known": True}]}
        self.check(home, registry, [get_state("light.kitchen")],
                   observations={"device.light.kitchen.state": shl.make_observation("on", source="test")},
                   action_graph={"h1": {**turn_on("h1"), "status": "success", "output": {}}})

    def test_whole_home_command_values_are_checked(self):
        with self.assertRaises(ValueError):
            self.home_assistant([set_temperature(50)])

    def test_whole_number_options_are_checked(self):
        self.simuhome([self.system_mode(3)])
        for value in (2, "3", True, None):  # not an option; text; yes/no; missing
            with self.assertRaises(ValueError, msg=value):
                self.simuhome([self.system_mode(value)])

    def test_rejection_reports_leans_reason(self):
        for check, batch, reason in [
                (self.simuhome, [self.system_mode(2)], "is not one of"),
                (self.homebench, [self.brightness("living_room__light", 150)], "argument_out_of_bounds")]:
            with self.assertRaises(ValueError) as caught:
                check(batch)
            self.assertIn(reason, str(caught.exception))
            self.assertNotIn("reference value", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
