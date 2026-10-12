"""Beispielhaus: Inventar, unterstuetzte Skills und Anfangswerte (Wert, Einheit)."""

HOME_DEFAULTS = {
    "rooms": {
        'living_room': {"name": 'Wohnzimmer', "initial": {"temperature_c": (20.0, 'degC')}},
        'bedroom': {"name": 'Schlafzimmer', "initial": {"temperature_c": (19.0, 'degC')}},
        'kitchen': {"name": 'Kueche', "initial": {"temperature_c": (21.0, 'degC')}},
        'bathroom': {"name": 'Badezimmer', "initial": {"temperature_c": (22.0, 'degC')}},
    },
    'devices': {
        'light_lr_ceiling': {
            "agent_id": 'lighting', "room_id": 'living_room',
            "skills": ['turn_on', 'turn_off', 'set_brightness', 'set_color_temperature'],
            "initial": {'power': (False, None), 'brightness_pct': (0, '%'), 'color_temperature_k': (2700, 'K')},
        },
        'light_lr_floor': {
            "agent_id": 'lighting', "room_id": 'living_room',
            "skills": ['turn_on', 'turn_off'],
            "initial": {'power': (False, None)},
        },
        'light_bedroom': {
            "agent_id": 'lighting', "room_id": 'bedroom',
            "skills": ['turn_on', 'turn_off', 'set_brightness'],
            "initial": {'power': (False, None), 'brightness_pct': (0, '%')},
        },
        'light_kitchen': {
            "agent_id": 'lighting', "room_id": 'kitchen',
            "skills": ['turn_on', 'turn_off'],
            "initial": {'power': (False, None)},
        },
        'light_bathroom': {
            "agent_id": 'lighting', "room_id": 'bathroom',
            "skills": ['turn_on', 'turn_off'],
            "initial": {'power': (False, None)},
        },
        'thermostat_lr': {
            "agent_id": 'thermostat', "room_id": 'living_room',
            "skills": ['set_temperature', 'set_mode'],
            "initial": {'current_temperature_c': (20.0, 'degC'), 'target_temperature_c': (21.0, 'degC'), 'mode': ('heat', None)},
        },
        'thermostat_bedroom': {
            "agent_id": 'thermostat', "room_id": 'bedroom',
            "skills": ['set_temperature', 'set_mode'],
            "initial": {'current_temperature_c': (19.0, 'degC'), 'target_temperature_c': (21.0, 'degC'), 'mode': ('heat', None)},
        },
        'tv_lr': {
            "agent_id": 'tv_display', "room_id": 'living_room',
            "skills": ['turn_on', 'turn_off', 'show_content'],
            "initial": {'power': (False, None), 'current_content': (None, None)},
        },
        'fridge_kitchen': {
            "agent_id": 'fridge', "room_id": 'kitchen',
            "skills": ['get_inventory'],
            "initial": {'inventory': ([{'item': 'milk', 'quantity': 1.0, 'unit': 'l'}, {'item': 'eggs', 'quantity': 6, 'unit': 'piece'}, {'item': 'tomatoes', 'quantity': 3, 'unit': 'piece'}], None)},
        },
        'audio_lr': {
            "agent_id": 'audio_system', "room_id": 'living_room',
            "skills": ['turn_on', 'turn_off', 'set_volume', 'play', 'pause', 'stop'],
            "initial": {'power': (False, None), 'volume_pct': (20, '%'), 'playback_status': ('stopped', None), 'current_track': (None, None)},
        },
        'audio_bedroom': {
            "agent_id": 'audio_system', "room_id": 'bedroom',
            "skills": ['turn_on', 'turn_off', 'set_volume', 'play', 'pause', 'stop'],
            "initial": {'power': (False, None), 'volume_pct': (20, '%'), 'playback_status': ('stopped', None), 'current_track': (None, None)},
        },
    },
    'services': {
        'clock_main': {
            "agent_id": 'clock',
            "skills": ['get_time', 'set_timer', 'set_alarm', 'start_stopwatch', 'stop_stopwatch'],
            "initial": {'timers': ([], None), 'alarms': ([], None), 'stopwatch_running': (False, None), 'stopwatch_elapsed_seconds': (0.0, 's')},
        },
        'calendar_personal': {
            "agent_id": 'calendar',
            "skills": ['list_events', 'create_event', 'delete_event'],
            "initial": {'events': ([], None)},
        },
        'search_engine_main': {
            "agent_id": 'search_engine',
            "skills": ['search'],
            "initial": {},
        },
    },
}
