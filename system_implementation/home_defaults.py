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

# Synthetic, deterministic information for benchmark runs. These are NOT live web results.
# Caller-supplied simulation_data["search_results"] replaces this catalogue completely.
SEARCH_FIXTURES = {alias: rows for aliases, rows in [
    (('Italian pasta recipe', 'pasta recipe', 'pasta ingredients'), [
        {
            'title': 'Italian tomato pasta',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic recipe fixture. Ingredients for 2: 200 g pasta, 3 tomatoes, 1 tbsp oil, salt. Steps: boil pasta for 10 minutes; chop tomatoes and simmer with oil for 8 minutes; combine and season.',
        },
    ]),
    (('cookie ingredients', 'cookie recipe', 'cookies'), [
        {
            'title': 'Simple cookies',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic recipe fixture. Ingredients: 150 g flour, 80 g butter, 60 g sugar, 1 egg. Mix butter and sugar, add egg and flour, shape cookies, bake at 180 C for 12 minutes.',
        },
    ]),
    (('chicken recipes', 'low-calorie meals with chicken breast', 'chicken recipe', 'chicken'), [
        {
            'title': 'Chicken with tomatoes',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic recipe fixture. Ingredients for 2: 250 g chicken breast, 3 tomatoes, 1 tbsp oil, 100 g rice. Steps: prepare rice; cook chicken thoroughly; simmer chopped tomatoes for 10 minutes and combine. The recipe is information, not evidence that chicken exists in the fridge.',
        },
    ]),
    (('lasagna recipe', 'lasagna'), [
        {
            'title': 'Beef lasagna',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic recipe fixture. Ingredients: 300 g ground beef, 6 lasagna sheets, 400 g tomatoes, 200 ml milk, 100 g cheese. Brown beef, simmer with tomatoes for 15 minutes, layer with pasta and sauce, bake at 180 C for 40 minutes.',
        },
    ]),
    (('dinner party recipes', 'easy meals', 'simple and quick meals', 'quick recipes', 'dinner recipes', 'meal ideas', 'recipes'), [
        {
            'title': 'Tomato omelette',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic recipe fixture. Ingredients for 1: 2 eggs, 1 tomato, 30 ml milk, 1 tsp oil. Chop tomato, whisk eggs with milk, heat oil, add tomato and eggs and cook until set. Preparation takes about 10 minutes.',
        },
    ]),
    (('Thanksgiving dinner recipes with ingredients from fridge', 'Thanksgiving recipes', 'Thanksgiving'), [
        {
            'title': 'Holiday vegetable bake',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic recipe fixture. Ingredients: 3 tomatoes, 2 potatoes, 100 ml milk, 50 g cheese. Slice vegetables, layer in a dish, add milk and cheese and bake at 180 C for 35 minutes.',
        },
    ]),
    (('kid-friendly food', 'kid-friendly activities', 'kid-friendly entertainment'), [
        {
            'title': 'Family activity fixture',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic fixture: a simple egg-and-tomato meal, drawing and board games, and a fictional short family animation. Personal preferences and nap times must come from the user/calendar.',
        },
    ]),
    (('meal times for marathon training', 'marathon training schedule'), [
        {
            'title': 'Example training organisation',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic organisational fixture: training 06:00-07:00, breakfast 07:30, lunch 12:30, dinner 18:30. This is an example plan, not a personal calendar or nutrition prescription.',
        },
    ]),
    (('weather', 'weather tomorrow', 'outdoor temperature', 'weather forecast'), [
        {
            'title': 'Synthetic Berlin weather',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Weather fixture, not live weather: outdoor temperature 8 C, rain expected today and tomorrow. Use these controlled values only for the simulation.',
            'data': {'location': 'Berlin', 'temperature_c': 8, 'rain_expected': True, 'tomorrow_rain_expected': True},
        },
    ]),
    (('barbershops', 'hair salon'), [
        {
            'title': 'Fictional local salon',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic fixture: Demo Salon, Example Street 1, open 09:00-18:00. No real business, live availability or confirmed appointment is represented.',
        },
    ]),
    (('movies', 'yoga practice videos', 'Stranger Things Season 3', 'tv shows'), [
        {
            'title': 'Media fixture',
            'url': None,
            'snippet': 'Synthetic benchmark fixture; no live source.',
            'content': 'Synthetic media catalogue entry. This fixture contains descriptive text only; it is not a playable video, streaming subscription or live recommendation.',
        },
    ]),
    (('Chicken Alfredo',), [
        {
            'title': 'Chicken Alfredo',
            'url': None,
            'snippet': 'Synthetic benchmark recipe fixture; no live source.',
            'content': 'Synthetic recipe fixture. Ingredients for 2: 200 g pasta, 200 g chicken breast, 100 ml cream, 50 g Parmesan and 1 tbsp butter. Boil pasta, cook chicken thoroughly, melt butter with cream, add cheese and combine. The fridge inventory determines which ingredients are missing.',
        },
    ]),
] for alias in aliases}
