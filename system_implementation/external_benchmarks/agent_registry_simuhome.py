"""Native SimuHome commands and writable attributes, discovered per device."""
import inspect
from enum import IntEnum


GROUPS = {
    "on_off_light": "lighting", "dimmable_light": "lighting",
    "air_conditioner": "thermostat", "heat_pump": "thermostat",
    "fan": "air_quality", "air_purifier": "air_quality",
    "humidifier": "air_quality", "dehumidifier": "air_quality",
    "tv": "audio_system", "window_covering_controller": "covers",
}
TYPES = {bool: "boolean", int: "integer", float: "number", str: "string", list: "array", dict: "object"}
TYPE_NAMES = {kind.__name__: name for kind, name in TYPES.items()}
ENVIRONMENT_STATES = {"thermostat": ["temperature"], "lighting": ["illuminance"], "covers": ["illuminance"],
                      "air_quality": ["humidity", "air_quality"]}


def read_operation(target, reads=(), parameters=None):
    return {"description": "Read current simulator information without changing devices.",
        "parameters": {target: {"type": "string"}, **(parameters or {})},
        "output": {target: {"type": "string"}, "data": {}},
        "reads": list(reads), "writes": [], "preconditions": [], "postconditions": [], "max_retries": 0}


def add_operation(card, device, skill, definition):
    device["skills"].append(skill)
    device["parameters"][skill] = definition["parameters"]
    if skill not in card["operations"]:
        card["capabilities"].append(skill)
        card["operations"][skill] = definition


def command_parameters(command):
    """Signatures come from SimuHome itself; do not guess Matter argument names."""
    parameters = {}
    for name, param in inspect.signature(command).parameters.items():
        kind = TYPES.get(param.annotation, TYPE_NAMES.get(param.annotation))
        schema = {"type": kind} if kind else {}
        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
            if param.default is None and "type" in schema:
                schema["type"] = [schema["type"], "null"]
        parameters[name] = schema
    return parameters


def cycle_durations(kind, prototype):
    """Derive cycle capabilities from isolated native devices, never the episode's goals."""
    from src.simulator.application.device_factory import create_device
    selectors = {"laundry_washer": ("LaundryWasherMode", "CurrentMode", "SupportedModes"),
                 "dishwasher": ("DishwasherMode", "CurrentMode", "SupportedModes"),
                 "laundry_dryer": ("LaundryDryerControls", "SelectedDrynessLevel", "SupportedDrynessLevels")}
    if kind not in selectors:
        return {}
    cluster, attribute, choices = selectors[kind]
    choices = prototype.endpoints[1][cluster].attributes[choices]
    durations = {}
    for choice in choices:
        value = choice["mode"] if attribute == "CurrentMode" else int(choice)
        device = create_device(kind, "duration_probe")
        endpoint = device.endpoints[1]
        endpoint["OnOff"].commands["On"]()
        endpoint[cluster].write_attribute(attribute, value)
        result = endpoint["OperationalState"].commands["Start"]()
        if result.success:
            durations[str(value)] = endpoint["OperationalState"].attributes["CountdownTime"]
    return {"selector": f"1.{cluster}.{attribute}", "seconds_by_value": durations}


def build_home(session):
    from src.simulator.application.device_factory import create_device

    config, registry, bindings = {"rooms": {}, "devices": {}, "services": {}}, {}, {}
    prototypes, durations = {}, {}
    rules = {state: session.call("get_environment_control_rules", {"state": state})["data"]
             for state in ("temperature", "humidity", "illuminance", "air_quality")}
    registry["home_information"] = {"agent": "home_information_agent", "enabled": True,
        "description": "Read room conditions (temperature, humidity, illuminance/lux, PM10) and environment control rules. No device changes.",
        "capabilities": ["get_room_states", "get_environment_control_rules", "get_current_time"], "operations": {
            "get_room_states": read_operation("service_id", ["room.{service_id}." + key
                                                             for key in ("temperature", "humidity", "illuminance", "pm10")]),
            "get_environment_control_rules": read_operation("service_id", parameters={
                "state": {"type": "string", "enum": list(rules)}}),
            "get_current_time": read_operation("service_id")}}
    for tool in registry["home_information"]["capabilities"]:
        def read_room(skill, arguments, simulator, now, simulation, *, tool=tool):
            params = ({"room_id": arguments["service_id"]} if tool == "get_room_states" else
                      {"state": arguments["state"]} if tool == "get_environment_control_rules" else {})
            return {"data": session.call(tool, params, origin="agent")["data"]}
        bindings[("home_information", tool)] = read_room
    for room in session.call("get_rooms", {})["data"]["rooms"]:
        room_id = room["room_id"]
        config["rooms"][room_id] = {"name": room["display_name"]}
        config["services"][room_id] = {"agent_id": "home_information", "room_id": room_id, "type": "room",
                                      "skills": registry["home_information"]["capabilities"]}
        # Discovery is performed for every room, independent of hidden evaluation goals.
        devices = session.call("get_room_devices", {"room_id": room_id})["data"]
        for device_id, info in devices.items():
            kind = info["device_type"]
            agent = GROUPS.get(kind, "appliances")
            card = registry.setdefault(agent, {"agent": agent + "_agent", "enabled": True,
                "description": "SimuHome native device control. Keep native units and enum values; "
                    "temperature values are hundredths of degrees Celsius. Method names encode endpoint.cluster.command "
                    "or endpoint.cluster.write_attribute. Optional start_time schedules the same method in virtual time; "
                    "omit it for immediate execution. Command/workflow acceptance does not imply a completed physical effect.",
                "context": {"environment_control_rules": {key: rules[key] for key in ENVIRONMENT_STATES.get(agent, [])}},
                "capabilities": [], "operations": {}})
            if kind not in prototypes:
                prototypes[kind] = create_device(kind, "schema_device")
                durations[kind] = cycle_durations(kind, prototypes[kind])
            structure = session.call("get_device_structure", {"device_id": device_id})["data"]
            device = {"agent_id": agent, "room_id": room_id, "type": kind, "method_family": kind,
                      "skills": [], "parameters": {}}
            if durations[kind]:
                device["cycle_duration"] = durations[kind]
            for endpoint, endpoint_info in structure["endpoints"].items():
                for cluster_id, cluster in endpoint_info["clusters"].items():
                    native_cluster = prototypes[kind].endpoints[int(endpoint)][cluster_id]
                    methods = [(command, "execute_command", command_parameters(native_cluster.commands[command]))
                               for command in cluster["commands"]]
                    for attribute, metadata in cluster["attributes"].items():
                        if metadata["readonly"]:
                            continue
                        schema = {"type": TYPE_NAMES[metadata["type"]]} if metadata["type"] in TYPE_NAMES else {}
                        if "enum_values" in metadata:
                            schema = {"enum": list(metadata["enum_values"].values()),
                                      "description": str(metadata["enum_values"])}
                        elif isinstance(native_cluster.attributes.get(attribute), IntEnum):
                            # Allowed values from SimuHome's own enumeration, e.g. SystemMode.
                            members = list(type(native_cluster.attributes[attribute]))
                            schema = {"type": "integer", "enum": [int(m) for m in members],
                                      "description": ", ".join(f"{int(m)}={m.name}" for m in members)}
                        methods.append(("write_" + attribute, "write_attribute", {"value": schema}))
                    for name, tool, parameters in methods:
                        skill = f"{endpoint}.{cluster_id}.{name}"
                        parameters = {"device_id": {"type": "string"}, **parameters,
                            "start_time": {"type": ["string", "null"], "default": None,
                                "description": "Optional future virtual YYYY-MM-DD HH:MM:SS; null executes now."}}
                        native = {"endpoint_id": int(endpoint), "cluster_id": cluster_id,
                                  "command_id" if tool == "execute_command" else "attribute_id":
                                      name if tool == "execute_command" else name.removeprefix("write_")}
                        definition = {"description": f"Native {tool}: {name}.", "parameters": parameters,
                            "output": {"device_id": {"type": "string"}, "accepted": {"const": True}, "native_result": {}},
                            "reads": [], "writes": ["device.{device_id}.attributes"],
                            "preconditions": [], "postconditions": [{"output": "accepted", "equals": True}],
                            "max_retries": 0}
                        add_operation(card, device, skill, definition)

                        def execute(skill, arguments, simulator, now, simulation, *, tool=tool, native=native):
                            args = dict(arguments)
                            target = args.pop("device_id")
                            start_time = args.pop("start_time", None)
                            params = {"device_id": target, **native}
                            params.update({"args": args} if tool == "execute_command" else args)
                            if start_time is not None:
                                params = {"start_time": start_time, "steps": [{"tool": tool, "args": params}]}
                                tool = "schedule_workflow"
                            result = session.call(tool, params, origin="agent")
                            return {"accepted": True, "native_result": result["data"]}

                        bindings[(agent, skill)] = execute
            for tool in ("get_device_structure", "get_all_attributes", "get_workflow_status"):
                definition = read_operation("device_id", ["device.{device_id}.attributes"],
                    {"workflow_id": {"type": "string"}} if tool == "get_workflow_status" else None)
                add_operation(card, device, tool, definition)
                def read_device(skill, arguments, simulator, now, simulation, *, tool=tool):
                    params = {"workflow_id": arguments["workflow_id"]} if tool == "get_workflow_status" else {"device_id": arguments["device_id"]}
                    return {"data": session.call(tool, params, origin="agent")["data"]}
                bindings[(agent, tool)] = read_device
            if "OperationalState" in structure["endpoints"].get("1", {}).get("clusters", {}):
                skill = "get_cycle_status"
                definition = read_operation("device_id", ["device.{device_id}.cycle_timing"])
                definition["description"] = "Read observed cycle state, remaining seconds and predicted ends_at in virtual time."
                add_operation(card, device, skill, definition)
                def read_cycle(skill, arguments, simulator, now, simulation):
                    home = session.call("get_home_state", {}, origin="agent")["data"]
                    target = next(d for room in home["rooms"].values() for d in room["devices"]
                                  if d["device_id"] == arguments["device_id"])
                    return {"data": {**session.cycle_timing(home, target),
                        "observed_at": home["current_time"],
                        "remaining_seconds": target["attributes"]["1.OperationalState.CountdownTime"]}}
                bindings[(agent, skill)] = read_cycle
            config["devices"][device_id] = device
    return config, registry, bindings
