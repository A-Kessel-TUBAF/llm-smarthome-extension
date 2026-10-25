"""Switch real devices on and off through Home Assistant (https://www.home-assistant.io).

Only `turn_on` and `turn_off` go to Home Assistant; every other operation, and every device
without an entry in HA_ENTITIES, stays simulated. Settings in `.env`:

    HA_URL=http://homeassistant.local:8123
    HA_TOKEN=<long-lived access token, created on your Home Assistant profile page>
    HA_ENTITIES=tv_lr=media_player.living_room_tv,light_lr_ceiling=light.living_room

Then run `python smart_home_langgraph.py --home-assistant`.
"""
import json
import os
import time
from urllib.request import Request, urlopen

# Home Assistant states that mean switched on or off; anything else (e.g. unavailable) is unknown.
ON = {"on", "playing", "paused", "idle", "buffering"}
OFF = {"off", "standby"}


class HomeAssistant:
    def __init__(self, url, token, entities, timeout=15.0):
        self.url, self.token, self.entities, self.timeout = url.rstrip("/"), token, dict(entities), timeout

    @classmethod
    def from_env(cls):
        pairs = (pair.split("=", 1) for pair in os.environ["HA_ENTITIES"].split(",") if pair.strip())
        return cls(os.environ["HA_URL"], os.environ["HA_TOKEN"],
                   {device.strip(): entity.strip() for device, entity in pairs},
                   float(os.getenv("HA_TIMEOUT", "15")))

    def _call(self, path, body=None):
        request = Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                          headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read() or b"null")

    def power(self, entity):
        """True or False for a switched-on or -off entity, None if Home Assistant does not know."""
        state = self._call(f"/api/states/{entity}")["state"]
        return True if state in ON else False if state in OFF else None

    def switch(self, entity, on):
        """Switch the entity and wait until Home Assistant reports the new state, or the timeout."""
        domain = entity.split(".", 1)[0]
        self._call(f"/api/services/{domain}/{'turn_on' if on else 'turn_off'}", {"entity_id": entity})
        deadline = time.monotonic() + self.timeout
        while (power := self.power(entity)) != on and time.monotonic() < deadline:
            time.sleep(0.5)
        return power

    def power_states(self):
        return {device: self.power(entity) for device, entity in self.entities.items()}

    def power_tool(self, fallback):
        """A device tool that switches mapped devices for real and records the power state that
        Home Assistant reports, so the usual outcome check catches a device that did not switch.
        Everything else, including the simulated side effects of switching, is left to `fallback`."""
        def tool(skill, args, simulator, now, simulation):
            entity = self.entities.get(args.get("device_id"))
            if entity is None or skill not in {"turn_on", "turn_off"}:
                return fallback(skill, args, simulator, now, simulation)
            power = self.switch(entity, skill == "turn_on")
            output = fallback(skill, args, simulator, now, simulation)
            simulator.write({f"device.{args['device_id']}.power": power}, now,
                            source="home_assistant", ref=simulation.get("_action_id"))
            return output
        return tool
