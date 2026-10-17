"""Admission of typed arguments by the Lean checker (no LLM needed).

The arguments are passed to `check_allocation` directly, without the JSON-schema validation of
`validate_action`, to show that the checker itself rejects wrongly typed values.

Run from system_implementation/ after building the checker (cd lean && lake build):
    python -m unittest test_typed_arguments
"""

import time
import unittest
from pathlib import Path

import smart_home_langgraph as shl

EXE = Path(__file__).parent / "lean/.lake/build/bin/allocation_checker"


def state():
    powered = {f"device.{d}.power": shl.make_observation(True, source="test")
               for d in ("light_lr_ceiling", "light_bedroom")}
    return shl.create_initial_state(powered, checker="lean", lean_executable=EXE)


def action(i, agent, skill, **args):
    return dict(task_id=f"a{i}", agent_id=agent, skill_id=skill, arguments=args, depends_on=[],
                status="running", dispatch_until=time.monotonic() + 30)


def brightness(value):
    return action(1, "lighting", "set_brightness", device_id="light_lr_ceiling", brightness_pct=value)


@unittest.skipUnless(EXE.is_file(), "build the Lean checker first: cd lean && lake build")
class TypedArgumentTests(unittest.TestCase):

    def assertAccepted(self, batch):
        shl.check_allocation(state(), batch)

    def assertRejected(self, batch):
        with self.assertRaises(ValueError):
            shl.check_allocation(state(), batch)

    def test_well_typed_batch_accepted(self):
        self.assertAccepted([
            brightness(40),
            action(2, "thermostat", "set_temperature", device_id="thermostat_lr", temperature_c=21.5),
            action(3, "thermostat", "set_mode", device_id="thermostat_bedroom", mode="heat")])

    def test_text_instead_of_number_rejected(self):
        self.assertRejected([brightness("high")])
        self.assertRejected([action(1, "thermostat", "set_temperature", device_id="thermostat_lr",
                                    temperature_c="21")])

    def test_boolean_instead_of_number_rejected(self):
        self.assertRejected([brightness(True)])

    def test_unknown_enum_value_rejected(self):
        self.assertRejected([action(1, "thermostat", "set_mode", device_id="thermostat_lr", mode="warm")])

    def test_out_of_range_rejected(self):
        self.assertRejected([brightness(150)])


if __name__ == "__main__":
    unittest.main()
