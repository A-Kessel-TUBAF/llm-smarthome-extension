"""Domain agents using Home Assistant's native Assist tools and schemas."""
from copy import deepcopy
from jsonschema import Draft202012Validator

GROUPS = {"light": "lighting", "climate": "thermostat", "media_player": "audio_system",
          "fan": "air_quality", "cover": "covers", "lock": "locks", "todo": "lists",
          "vacuum": "appliances", "valve": "valves", "switch": "switches"}


def build_home(session):
    import probatio
    from homeassistant.components.intent.llm import DEVICE_CONTROL_TOOL_USAGE_PROMPT

    config = {"rooms": session.rooms, "devices": {}, "services": {}}
    registry, bindings = {}, {}
    domains = {entity["domain"] for entity in session.entities.values()}
    all_keys = [f"device.{target}.{field}" for target in session.entities for field in ("state", "attributes")]
    for domain in sorted(domains):
        agent = GROUPS.get(domain, domain)
        targets = [key for key, entity in session.entities.items() if entity["domain"] == domain]
        keys = [f"device.{target}.{field}" for target in targets for field in ("state", "attributes")]
        card = registry.setdefault(agent, {"agent": agent + "_agent", "enabled": True,
            "description": f"Home Assistant {domain} devices. Use native Assist argument names and units. "
                "Resolve implicit locations from requesting_device and pass an explicit native name/area/floor. "
                "A domain-only call can affect matching devices in multiple rooms. "
                "For an ambiguous target ask for clarification; do not choose arbitrarily.",
            "context": {"requesting_device": session.user_location,
                        "tool_instructions": DEVICE_CONTROL_TOOL_USAGE_PROMPT}, "capabilities": [], "operations": {}})
        card["operations"]["get_state"] = {
            "description": "Read this entity's live Home Assistant state and attributes without changing it.",
            "parameters": {"device_id": {"type": "string"}},
            "output": {"device_id": {"type": "string"}, "data": {}},
            "reads": ["device.{device_id}.state", "device.{device_id}.attributes"], "writes": [],
            "preconditions": [], "postconditions": [], "max_retries": 0}
        card["capabilities"].append("get_state")
        for target in targets:
            entity = session.entities[target]
            config["devices"][target] = {"agent_id": agent, "room_id": entity["room_id"],
                "type": domain, "name": entity["name"], "skills": ["get_state"]}
        def read_state(skill, arguments, simulator, now, simulation):
            return {"data": session.read(arguments["device_id"])}
        bindings[(agent, "get_state")] = read_state

        for tool in session.api.tools:
            schema = probatio.to_json_schema(tool.parameters, custom_serializer=session.api.custom_serializer)
            props = schema.get("properties", {})
            # Generic intents address multiple domains; domain-specific tools retain their native owner.
            if tool.integration != domain and not (tool.integration in {"intent", "homeassistant"} and "name" in props):
                continue
            if tool.integration == "intent" and tool.name.endswith(("HassSetPosition", "HassStopMoving")) and domain not in {"cover", "valve"}:
                continue
            if tool.integration == "intent" and (domain not in GROUPS or domain == "todo"):
                continue
            parameters = deepcopy(props)
            for name, definition in list(parameters.items()):
                # The system's parameter format marks optional arguments with a default.
                if name not in schema.get("required", []):
                    default = definition.get("default")
                    if default is None and not Draft202012Validator(definition).is_valid(None):
                        parameters[name] = {"anyOf": [definition, {"type": "null"}], "default": None}
                    else:
                        parameters[name].setdefault("default", default)
            parameters = {"service_id": {"type": "string", "enum": [domain], "default": domain}, **parameters}
            read_only = tool.annotations.read_only
            affected = all_keys if tool.integration in {"intent", "homeassistant"} else keys
            card["operations"][tool.name] = {"description": tool.description or tool.name,
                "parameters": parameters,
                "output": {"service_id": {"type": "string"}, "data": {}},
                "reads": affected, "writes": [] if read_only else affected,
                "preconditions": [], "postconditions": [], "max_retries": 0}
            card["capabilities"].append(tool.name)
            def execute(skill, arguments, simulator, now, simulation):
                args = {key: value for key, value in arguments.items() if key != "service_id" and value is not None}
                return {"data": session.call(skill, args)}
            bindings[(agent, tool.name)] = execute
        skills = [name for name in card["operations"] if name != "get_state"]
        if skills:
            config["services"][domain] = {"agent_id": agent, "room_id": None, "type": domain,
                "serves_type": domain,
                "skills": skills, "parameters": {name: card["operations"][name]["parameters"] for name in skills}}
    return config, registry, bindings
