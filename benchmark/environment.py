"""Independent DEXA device model. No planner, agent registry or system imports."""
from copy import deepcopy
from datetime import datetime, timedelta
from threading import RLock
import json
import time


NOW = "2026-10-22T08:00:00+02:00"
ROOMS = ("living_room", "bedroom", "kitchen", "bathroom", "guest_room", "kids_room", "study", "dining_room")
DOMAINS = {"light": "lighting", "thermostat": "thermostat", "audio": "audio_system",
           "display": "tv_display", "fridge": "fridge", "calendar": "calendar", "clock": "clock", "search": "search_engine"}
PREFERENCES = {
    "default_room": "living_room", "wake_up": "07:00", "reminder_lead_minutes": 60,
    "default_devices": {"light": "light_living_room", "thermostat": "thermostat_living_room",
                        "audio": "audio_living_room", "display": "display_living_room"},
    "search_location": "Berlin",
    "relative_temperature_delta_c": 2, "relative_brightness_delta_pct": 20,
    "relative_volume_delta_pct": 10, "cold_below_c": 15, "event_duration_minutes": 60,
    "scenes": {"study": {"brightness_pct": 90, "temperature_c": 20, "track": "Focus"}},
    "dim_brightness_pct": 20, "bright_brightness_pct": 90, "normal_brightness_pct": 60,
    "quiet_volume_pct": 15, "warm_white_k": 2700, "tonight": "20:00", "afternoon": ["13:00", "17:00"],
}
RECIPES = [
    {"name": "Tomato omelette", "ingredients": ["eggs", "tomatoes", "milk"], "cooking_minutes": 10,
     "instructions": "Beat eggs with milk; add chopped tomatoes and cook for 10 minutes."},
    {"name": "Italian pasta", "ingredients": ["pasta", "tomatoes", "oil"], "cooking_minutes": 10,
     "instructions": "Boil pasta; simmer tomatoes with oil; combine after 10 minutes."},
    {"name": "Chicken rice", "ingredients": ["chicken breast", "rice", "tomatoes"], "cooking_minutes": 20,
     "instructions": "Cook rice and chicken thoroughly; add tomatoes. Total cooking time: 20 minutes."},
    {"name": "Chicken omelette", "ingredients": ["chicken breast", "eggs", "tomatoes"], "cooking_minutes": 20,
     "instructions": "Cook chicken thoroughly, add chopped tomatoes and beaten eggs; cook until set. Total: 20 minutes."},
    {"name": "Lasagna", "ingredients": ["ground beef", "lasagna sheets", "tomatoes", "milk", "cheese"], "cooking_minutes": 60,
     "instructions": "Brown beef, make tomato sauce, layer with pasta and cheese, bake. Total: 60 minutes."},
    {"name": "Cookies", "ingredients": ["flour", "butter", "sugar", "eggs"], "cooking_minutes": 12,
     "instructions": "Mix, shape cookies and bake at 180 C for 12 minutes."},
    {"name": "Chicken Alfredo", "ingredients": ["chicken breast", "pasta", "cream", "parmesan", "butter"], "cooking_minutes": 25,
     "instructions": "Boil pasta, cook chicken, combine with cream, butter and parmesan."},
    {"name": "Pancakes", "ingredients": ["eggs", "flour", "milk"], "cooking_minutes": 15,
     "instructions": "Mix ingredients and fry small pancakes for 15 minutes."},
    {"name": "Popcorn", "ingredients": ["corn kernels", "oil"], "cooking_minutes": 5,
     "instructions": "Heat kernels with oil in a covered pan for 5 minutes."},
    {"name": "Roast turkey", "ingredients": ["turkey", "butter", "potatoes"], "cooking_minutes": 180,
     "instructions": "Roast the prepared turkey until thoroughly cooked, approximately 180 minutes."},
]


def initial_home():
    home = {"rooms": {r: {"name": r.replace("_", " ")} for r in ROOMS}, "devices": {}, "services": {}}
    for room in ROOMS:
        home["devices"]["light_" + room] = {"type": "light", "room_id": room,
            "state": {"power": True, "brightness_pct": 60, "color_temperature_k": 4000, "color": "white"}}
        home["devices"]["thermostat_" + room] = {"type": "thermostat", "room_id": room,
            "state": {"target_temperature_c": 21, "mode": "heat"}}
    for room in ("living_room", "bedroom"):
        home["devices"]["audio_" + room] = {"type": "audio", "room_id": room,
            "state": {"power": False, "volume_pct": 30, "playback_status": "stopped", "current_track": None}}
    for room in ("living_room", "kitchen"):
        home["devices"]["display_" + room] = {"type": "display", "room_id": room,
            "state": {"power": False, "brightness_pct": 60, "volume_pct": 30, "current_content": None}}
    inventory = [{"item": n, "quantity": q, "unit": u, "expires": e} for n, q, u, e in
        [("milk", 1, "l", "2026-10-23"), ("eggs", 6, "piece", "2026-10-29"),
         ("tomatoes", 3, "piece", "2026-10-23"), ("chicken breast", 250, "g", "2026-10-24")]]
    home["devices"]["fridge_kitchen"] = {"type": "fridge", "room_id": "kitchen", "state": {"inventory": inventory}}
    events = [{"event_id": key, "title": title, "start": day + "T" + start + ":00+02:00",
               "end": day + "T" + end + ":00+02:00", "location": location} for key, title, day, start, end, location in
        [("today-meeting", "Team meeting", "2026-10-22", "10:00", "11:00", "Office"),
         ("today-deadline", "Project deadline", "2026-10-22", "18:00", "18:30", "Home"),
         ("dentist", "Dentist appointment", "2026-10-23", "10:00", "11:00", "Dentist"),
         ("tomorrow-meeting", "Team meeting", "2026-10-23", "13:00", "14:00", "Office"),
         ("doctor", "Doctor appointment", "2026-10-23", "15:00", "16:00", "Clinic"),
         ("training", "Training", "2026-10-24", "06:00", "07:00", "Park")]]
    home["services"] = {
        "clock_main": {"type": "clock", "state": {"timers": [], "alarms": [], "schedules": []}},
        "calendar_personal": {"type": "calendar", "state": {"events": events,
            "todo": ["Finish project report", "Buy groceries"]}},
        "search_main": {"type": "search", "state": {}},
    }
    return home


def operation_specs(kind):
    s = {"type": "string", "minLength": 1}
    n = {"type": "number"}
    pct = {"type": "integer", "minimum": 0, "maximum": 100}
    specs = {"get_state": ({}, [])} if kind in {"light", "thermostat", "audio", "display"} else {}
    if kind in {"light", "audio", "display"}:
        specs.update(turn_on=({}, ["power"]), turn_off=({}, ["power"]))
    if kind in {"light", "display"}:
        specs["set_brightness"] = ({"brightness_pct": pct}, ["brightness_pct"])
    if kind == "light":
        specs.update(set_color_temperature=({"color_temperature_k": {"type": "integer", "minimum": 2000, "maximum": 6500}}, ["color_temperature_k"]),
                     set_color=({"color": s}, ["color"]))
    if kind == "thermostat":
        specs.update(set_temperature=({"temperature_c": {**n, "minimum": 5, "maximum": 35}}, ["target_temperature_c"]),
                     set_mode=({"mode": {"enum": ["heat", "cool", "off"]}}, ["mode"]))
    if kind in {"audio", "display"}:
        specs["set_volume"] = ({"volume_pct": pct}, ["volume_pct"])
    if kind == "audio":
        specs.update(play=({"track": s}, ["playback_status", "current_track"]),
                     stop=({}, ["playback_status", "current_track"]))
    if kind == "display":
        specs["show_content"] = ({"content": s, "content_type": {"type": "string", "default": "text"}}, ["current_content"])
    if kind == "fridge":
        specs["get_inventory"] = ({}, [])
    if kind == "clock":
        specs.update(get_time=({}, []),
            set_timer=({"duration_seconds": {**n, "exclusiveMinimum": 0}, "label": {"type": "string", "default": ""},
                       "repeat_seconds": {**n, "minimum": 0, "default": 0}}, ["timers"]),
            set_alarm=({"at": {**s, "format": "date-time"}, "label": {"type": "string", "default": ""}}, ["alarms"]),
            schedule=({"at": {**s, "format": "date-time"}, "tool": s, "arguments": {"type": "object"}}, ["schedules"]))
    if kind == "calendar":
        specs.update(list_events=({"start": {"type": ["string", "null"], "default": None},
                                  "end": {"type": ["string", "null"], "default": None}}, []),
            get_todo=({}, []), create_event=({"title": s, "start": {**s, "format": "date-time"},
                "end": {**s, "format": "date-time"}, "location": {"type": "string", "default": ""}}, ["events"]))
    if kind == "search":
        specs["search"] = ({"query": s}, [])
    return specs


class Session:
    """Fresh state and one recorder for every case, irrespective of the participant."""
    def __init__(self, case):
        self.now = case["scenario"]["now"]
        self.home = initial_home()
        rooms = case["scenario"].get("rooms", list(ROOMS))
        self.home["rooms"] = {k: v for k, v in self.home["rooms"].items() if k in rooms}
        self.home["devices"] = {k: v for k, v in self.home["devices"].items() if v["room_id"] in rooms}
        self.preferences = deepcopy(PREFERENCES)
        self.preferences.update(case["scenario"].get("preferences", {}))
        self.weather = {"location": "Berlin", "temperature_c": 8, "rain_today": True,
                        "rain_tomorrow": True, "sunset": "2026-10-22T18:00:00+02:00",
                        **case["scenario"].get("weather", {})}
        self.state = {}
        self.entities = {**self.home["devices"], **self.home["services"]}
        for target, entity in self.entities.items():
            prefix = ("device." if target in self.home["devices"] else "service.") + target + "."
            self.state.update({prefix + k: deepcopy(v) for k, v in entity["state"].items()})
        self.state.update(deepcopy(case["scenario"].get("state", {})))
        self.initial = deepcopy(self.state)
        self.visible = {k: deepcopy(v) for k, v in self.state.items() if not k.endswith((".inventory", ".events", ".todo"))}
        self.fault = deepcopy(case.get("injected_failure"))
        self.calls, self.injections, self.versions = [], [], {}
        self.lock, self.fault_count, self.sequence = RLock(), 0, 0
        self.tools, self.methods = [], {}
        for target, entity in self.entities.items():
            prefix = ("device." if target in self.home["devices"] else "service.") + target + "."
            for skill, (parameters, writes) in operation_specs(entity["type"]).items():
                name = target + "_" + skill
                power = skill in {"play", "show_content", "set_brightness", "set_color_temperature", "set_color", "set_volume"}
                if skill == "turn_off":
                    writes = list(entity["state"] if entity["type"] in {"audio", "display"} else writes)
                    writes = [k for k in writes if k in {"power", "current_track", "playback_status", "current_content"}]
                self.methods[name] = (target, skill, prefix, parameters)
                self.tools.append({"name": name, "description": f"{entity['type']} in {entity.get('room_id', 'shared services')}: {skill}. "
                    + ("Requires power=true. " if power else "")
                    + ("Replaces the current display content; combine information requested together in one payload. " if skill == "show_content" else "")
                    + ("Schedule an existing tool at a future ISO time; the simulator executes it after this episode." if skill == "schedule" else ""),
                    "parameters": {"type": "object", "properties": parameters,
                        "required": [k for k, p in parameters.items() if "default" not in p], "additionalProperties": False},
                    "target_id": target, "skill_id": skill, "domain": DOMAINS[entity["type"]],
                    "contract": {"reads": [prefix + "power"] if power else [],
                        "writes": [prefix + k for k in writes],
                        "preconditions": [{"state": prefix + "power", "equals": True}] if power else []}})

    def public_context(self):
        return {"now": self.now, "reference_time": self.now, "timezone": "Europe/Berlin", "user_location": "living_room",
                "rooms": self.home["rooms"], "entities": {k: {p: v for p, v in e.items() if p != "state"} for k, e in self.entities.items()},
                "state": deepcopy(self.visible), "preferences": deepcopy(self.preferences),
                "instructions": "Use the supplied preferences to resolve implicit requests. Read inventory, calendar and to-do data through tools. "
                    "Explicit targets and unique referents in the observed state take precedence over default_devices. "
                    "Relative adjustments use the saved increments; dim, bright and quiet use the named preferences. "
                    "For 'too bright' use the dim setting; optimal work lighting/temperature use the study scene. "
                    "Use default_pasta_recipe for unspecified pasta when supplied. Execute clear independent goals; "
                    "ask about unresolved choices and leave only those parts and their dependent actions untouched. "
                    "Retain canonical catalogue recipe names in selected/displayed recipes so they can be checked automatically. "
                    "Scheduling is simulated; tools use ISO datetimes. Final temperature means thermostat setpoint, not physical room temperature. "
                    "Times without a date refer to the next applicable occurrence. No external internet access is required."}

    def snapshot(self):
        # Compatibility hook for adapters: only observations already public or obtained by tools.
        with self.lock:
            return self.now, deepcopy(self.visible)

    def update(self, values):
        for k, v in values.items():
            if self.state.get(k) != v:
                self.versions[k] = self.versions.get(k, 0) + 1
            self.state[k] = deepcopy(v)
            self.visible[k] = deepcopy(v)

    def call(self, name, arguments, *, origin="system"):
        from jsonschema import Draft202012Validator, FormatChecker
        started = time.perf_counter()
        with self.lock:
            self.sequence += 1
            entry = {"sequence": self.sequence, "tool": name, "arguments": deepcopy(arguments), "origin": origin,
                     "started_at": time.time(), "virtual_time": self.now, "status": "error"}
            before = deepcopy(self.state)
            try:
                target, skill, prefix, parameters = self.methods[name]
                json.dumps(arguments, allow_nan=False)
                tool = next(t for t in self.tools if t["name"] == name)
                Draft202012Validator(tool["parameters"], format_checker=FormatChecker()).validate(arguments)
                args = {**{k: deepcopy(p["default"]) for k, p in parameters.items() if "default" in p}, **arguments}
                entry.update(target_id=target, skill_id=skill)
                fault = self.fault or {}
                mode = fault.get("mode")
                matches = origin == "system" and target == fault.get("target_id") and (skill == fault.get("skill_id") or mode == "unavailable")
                if matches and mode in {"unavailable", "transient"} and (mode == "unavailable" or self.fault_count == 0):
                    self.fault_count += 1
                    self.injections.append({"mode": mode, "sequence": entry["sequence"], "target_id": target, "skill_id": skill})
                    raise RuntimeError("DeviceUnavailable: configured device operation is unreachable.")
                if matches and mode == "state_changed" and skill == "play" and not self.fault_count and self.state[prefix + "power"]:
                    self.fault_count += 1
                    self.update({prefix + "power": False})
                    self.injections.append({"mode": mode, "sequence": entry["sequence"], "target_id": target,
                        "changes": {prefix + "power": False}, "before_skill": skill})
                for condition in tool["contract"]["preconditions"]:
                    if self.state[condition["state"]] != condition["equals"]:
                        raise ValueError("PreconditionFailed: device must be powered on.")
                visible_before, versions_before = deepcopy(self.visible), dict(self.versions)
                result = self.execute(target, skill, prefix, args)
                if matches and mode == "no_effect":
                    self.state, self.visible, self.versions = before, visible_before, versions_before
                    self.injections.append({"mode": mode, "sequence": entry["sequence"], "target_id": target, "skill_id": skill})
                entry.update(status="success", output=deepcopy(result))
                return result
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                entry.update(ended_at=time.time(), latency_s=time.perf_counter() - started,
                    changes={k: deepcopy(v) for k, v in self.state.items() if before.get(k) != v})
                self.calls.append(entry)

    def execute(self, target, skill, prefix, args):
        get = lambda field: deepcopy(self.state[prefix + field])
        changes = {}
        if skill == "get_state":
            result = {k[len(prefix):]: deepcopy(v) for k, v in self.state.items() if k.startswith(prefix)}
            self.visible.update({prefix + k: v for k, v in result.items()})
            return result
        if skill == "get_time":
            return {"current_time": self.now, "timezone": "Europe/Berlin"}
        if skill == "get_inventory":
            self.visible[prefix + "inventory"] = get("inventory")
            return {"inventory": get("inventory")}
        if skill in {"list_events", "get_todo"}:
            field = "events" if skill == "list_events" else "todo"
            values = get(field)
            self.visible[prefix + field] = deepcopy(values)
            if skill == "list_events":
                values = [e for e in values if (not args["start"] or datetime.fromisoformat(e["end"]) > datetime.fromisoformat(args["start"]))
                    and (not args["end"] or datetime.fromisoformat(e["start"]) < datetime.fromisoformat(args["end"]))]
            return {field: values}
        if skill == "search":
            # A fixed, public world catalogue, not a case-dependent answer generator.
            return {"recipes": deepcopy(RECIPES), "weather": deepcopy(self.weather),
                "media": ["Focus", "Sleep", "Relax", "Cozy", "Romantic", "Jazz", "Classical", "Upbeat", "Holiday", "News podcast",
                    "Workout", "Meditation", "Morning", "Family comedy", "Kids animation", "Fireplace", "30-minute workout", "Stranger Things Season 3"],
                "places": [{"name": "Demo Salon", "address": "Example Street 1", "opening_hours": "09:00-18:00",
                            "category": "barbershop", "location": "Berlin", "distance_m": 400}],
                "activities": ["drawing", "board games"], "sleep_tips": ["dim lights", "keep a regular bedtime"],
                "cocktails": [{"name": "Citrus spritz", "ingredients": ["orange juice", "sparkling water"],
                               "preparation_minutes": 5, "instructions": "Mix orange juice and sparkling water over ice."}],
                "meal_times": ["07:30", "12:30", "18:30"], "query": args["query"]}
        if skill in {"turn_on", "turn_off"}:
            changes["power"] = skill == "turn_on"
            if skill == "turn_off" and self.entities[target]["type"] == "audio":
                changes.update(playback_status="stopped", current_track=None)
            if skill == "turn_off" and self.entities[target]["type"] == "display":
                changes["current_content"] = None
        elif skill in {"set_brightness", "set_volume", "set_color", "set_color_temperature", "set_temperature", "set_mode"}:
            field = {"set_temperature": "target_temperature_c"}.get(skill, next(iter(args)))
            changes[field] = next(iter(args.values()))
        elif skill in {"play", "stop"}:
            changes.update(playback_status="playing" if skill == "play" else "stopped", current_track=args.get("track"))
        elif skill == "show_content":
            changes["current_content"] = {"type": args["content_type"], "value": args["content"]}
        elif skill in {"set_timer", "set_alarm", "schedule", "create_event"}:
            field = {"set_timer": "timers", "set_alarm": "alarms", "schedule": "schedules", "create_event": "events"}[skill]
            item = {**deepcopy(args), "id": f"{target}-{self.sequence}"}
            if skill == "create_event" and datetime.fromisoformat(args["start"]) >= datetime.fromisoformat(args["end"]):
                raise ValueError("Event start must precede its end.")
            if skill in {"set_alarm", "schedule"} and datetime.fromisoformat(args["at"]) <= datetime.fromisoformat(self.now):
                raise ValueError("Time must be in the future.")
            if skill == "schedule" and (args["tool"] not in self.methods or self.methods[args["tool"]][1] == "schedule"):
                raise ValueError("Schedule must reference an existing non-scheduling tool.")
            if skill == "create_event":
                item["event_id"] = item.pop("id")
            changes[field] = get(field) + [item]
        else:
            raise ValueError("Unknown operation: " + skill)
        self.update({prefix + k: v for k, v in changes.items()})
        return deepcopy(changes)

    def finish_schedules(self):
        # Run the registered, finite schedule after the system returns, not in real time.
        schedules = sorted(self.state["service.clock_main.schedules"], key=lambda s: datetime.fromisoformat(s["at"]))
        for item in schedules:
            self.now = item["at"]
            try:
                self.call(item["tool"], item["arguments"], origin="scheduled")
            except Exception:
                pass  # The recorder retains the failure for independent evaluation.
