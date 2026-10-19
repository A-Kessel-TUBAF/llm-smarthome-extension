"""HomeBench device families and contracts, built from each home's method list."""
from copy import deepcopy

GROUPS = {
    "lighting": "Lights and their brightness or RGB colour.",
    "thermostat": "Air conditioners, heating and water heaters.",
    "audio_system": "Media playback, songs, artists, styles and volume.",
    "air_quality": "Fans, air purifiers, humidifiers, dehumidifiers and aromatherapy.",
    "covers": "Curtains, blinds and garage doors.",
    "appliances": "Robot vacuums and packing the trash.",
}
DEVICE_GROUPS = {
    "light": "lighting",
    **dict.fromkeys(("air_conditioner", "heating", "water_heater"), "thermostat"),
    "media_player": "audio_system",
    **dict.fromkeys(("fan", "air_purifiers", "humidifier", "dehumidifiers", "aromatherapy"), "air_quality"),
    **dict.fromkeys(("curtain", "blinds", "garage_door"), "covers"),
    **dict.fromkeys(("vacuum_robot", "trash"), "appliances"),
}
STATE_VALUES = {"turn_on": "on", "turn_off": "off", "open": "open", "close": "closed",
                "play": "playing", "pause": "paused", "stop": "stopped", "pack": "packed"}
PARAM_TYPES = {"int": {"type": "integer"}, "str": {"type": "string"},
               "typing.Tuple[int, int, int]": {"type": "array", "items": {"type": "integer"},
                                              "minItems": 3, "maxItems": 3}}


def device_id(method):
    room, device = method["room_name"], method["device_name"]
    return device if room == "None" else f"{room}__{device}"


def device_state(home, method):
    # The released data use a special top-level name for the room-independent vacuum.
    if method["room_name"] == "None":
        return home["home_status"].get("VacuumRobot", {})
    return home["home_status"].get(method["room_name"], {}).get(method["device_name"], {})


def parameters(method, attributes=None):
    attributes = {k.strip(): v for k, v in (attributes or {}).items()}
    schemas = {"device_id": {"type": "string", "minLength": 1}}
    for parameter in method["parameters"]:
        name = parameter["name"]
        schema = deepcopy(PARAM_TYPES[parameter["type"]])
        attribute = "cleaning_area" if name == "area" else name
        metadata = attributes.get(attribute, {})
        if "options" in metadata:
            schema["enum"] = metadata["options"]
        if schema["type"] == "integer":
            for source, target in (("lowest", "minimum"), ("highest", "maximum")):
                if source in metadata:
                    schema[target] = int(metadata[source])
        schemas[name] = schema
    return schemas


def operation(method):
    skill = method["operation"]
    field = skill[4:] if skill.startswith("set_") else "state"
    key = "device.{device_id}." + field
    postcondition = {"state": key, "equals_arg": method["parameters"][0]["name"]} if method["parameters"] else {
        "state": key, "equals": STATE_VALUES[skill]}
    return {
        "description": f"HomeBench {skill}; use this device's parameter limits and options.",
        "parameters": parameters(method),
        "output": {"device_id": {"type": "string"}, "applied": {"const": True}},
        "reads": [], "writes": [key], "preconditions": [], "postconditions": [postcondition],
        "max_retries": 0,
    }


def build_home(home):
    """Return the system's home_config and registry; never read task gold outputs."""
    config = {"rooms": {}, "devices": {}, "services": {}}
    registry = {}
    for method in home["method"]:
        room, kind, skill = method["room_name"], method["device_name"], method["operation"]
        agent, target = DEVICE_GROUPS[kind], device_id(method)
        if room != "None":
            config["rooms"][room] = {"name": "dining room" if room == "ding_room" else room.replace("_", " ")}
        card = registry.setdefault(agent, {"agent": agent + "_agent", "enabled": True,
                                          "description": GROUPS[agent], "capabilities": [], "operations": {}})
        if skill not in card["operations"]:
            card["capabilities"].append(skill)
            card["operations"][skill] = operation(method)
        original = device_state(home, method)
        device = config["devices"].setdefault(target, {
            "agent_id": agent, "room_id": None if room == "None" else room, "type": kind,
            "homebench_name": kind if room == "None" else f"{room}.{kind}",
            "skills": [], "parameters": {}, "initial": {
                **({"state": (original["state"], None)} if "state" in original else {}),
                **{key.strip(): (value["value"], None) for key, value in original.get("attributes", {}).items()},
            },
        })
        device["skills"].append(skill)
        device["parameters"][skill] = parameters(method, original.get("attributes"))
    return config, registry


def execute(skill, arguments, simulator, now, simulation):
    """Lightweight local effects for controller checks; HomeBench scores commands, not physics."""
    field = skill[4:] if skill.startswith("set_") else "state"
    value = next(value for key, value in arguments.items() if key != "device_id") if skill.startswith("set_") else STATE_VALUES[skill]
    simulator.write({f"device.{arguments['device_id']}.{field}": value}, now, ref=simulation.get("_action_id"))
    return {"applied": True}
