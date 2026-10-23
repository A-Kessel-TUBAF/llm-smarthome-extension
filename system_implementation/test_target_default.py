"""An omitted routing argument (service_id) takes its declared default (no LLM, simulator or Lean).

Regression for the Home Assistant runs in which 81 of 95 cases failed with PlanError:
'service_id' before planning could complete: the adapter adds service_id for routing, Home
Assistant's own tool schema does not have it, and agents often omitted it.

Run from system_implementation/:
    python -m unittest test_target_default
"""
import unittest

import smart_home_langgraph as shl


def state(service_schema):
    operation = {"parameters": {"service_id": service_schema, "name": {"type": "string"}},
                 "reads": ["device.x.state"], "writes": ["device.x.state"], "preconditions": []}
    return {"agent_registry": {"lighting": {"operations": {"HassTurnOn": operation}}}, "action_graph": {}}


def action(**arguments):
    return {"task": {"agent_id": "lighting"}, "skill_id": "HassTurnOn", "arguments": arguments}


class ServiceDefaultTests(unittest.TestCase):

    def test_omitted_service_takes_declared_default(self):
        s, a = state({"type": "string", "enum": ["light"], "default": "light"}), action(name="Kitchen Light")
        shl.fill_service_ids(s, "lighting", [a])
        self.assertEqual(a["arguments"], {"service_id": "light", "name": "Kitchen Light"})
        self.assertEqual(shl.action_footprint(s, a)[2], ("service_id", "light"))

    def test_given_service_is_kept(self):
        s, a = state({"type": "string", "default": "light"}), action(service_id="other", name="x")
        shl.fill_service_ids(s, "lighting", [a])
        self.assertEqual(a["arguments"]["service_id"], "other")

    def test_without_default_behaviour_is_unchanged(self):
        s, a = state({"type": "string"}), action(name="x")
        shl.fill_service_ids(s, "lighting", [a])
        self.assertNotIn("service_id", a["arguments"])
        with self.assertRaises(KeyError):
            shl.action_footprint(s, a)

    def test_non_action_entries_are_ignored(self):
        s = state({"type": "string", "default": "light"})
        shl.fill_service_ids(s, "lighting", ["not an action", {"skill_id": "HassTurnOn"}])  # no arguments dict


if __name__ == "__main__":
    unittest.main()
