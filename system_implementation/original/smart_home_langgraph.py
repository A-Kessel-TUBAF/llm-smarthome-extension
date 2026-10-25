from langchain_ollama import ChatOllama
from typing import List, Optional, Dict, Any, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="gemma2", temperature=0.0)

# State
class SmartHomeState(TypedDict, total=False):

    messages: Annotated[list, add_messages]

    # Intent processing
    complexity_score: int
    infos: List[str]
    original_user_input: str
    task_queue: List[dict]
    key_modifiers: List[str]

    # Collaboration
    collaboration_request: Dict
    pending_task: Optional[Dict[str, Any]]
    task_history:List[dict]

    # Agent responses
    clock_response: Optional[str]
    clock_result: Optional[str]
    calendar_response: Optional[str]
    calendar_result: Optional[str]
    search_engine_response: Optional[str]
    search_engine_result: Optional[str]
    tv_display_response: Optional[str]
    tv_display_result: Optional[str]
    fridge_response: Optional[str]
    fridge_result: Optional[str]

    lighting_response: Optional[str]
    lighting_result: Optional[str]
    thermostat_response: Optional[str]
    thermostat_result: Optional[str]
    audio_system_response: Optional[str]
    audio_system_result: Optional[str]


def human(state: SmartHomeState) -> Command:

    user_input = interrupt(value="wait for user input...")
    if user_input.strip().lower() in {"q", "quit", "exit"}:
        raise SystemExit

    return Command(
        update={
            "messages": state["messages"] + [
                {"role": "human", "content": user_input}
            ]
        },
        goto= "intent_analysis"
    )

def get_user_input(state: SmartHomeState) -> str:
    """
    Get user input from state
    """
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "human":
            return msg.get("content")

    raise ValueError("No user input found")

def intent_analysis(state: SmartHomeState) -> Command:

    user_message= get_user_input(state)

    if not user_message:
        raise ValueError("No user message found for intent classification")

    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""Analyze the user's smart home request.

        User input: {user_message}

        Task 1: Split into separate information units
        - One info = one intent, feeling, or fact
        - Keep all details: what, how, when, why, where
        - If "and" connects independent intents or requests, split them (e.g., 'I'm hungry and tired' = two separate feelings)"

        Task 2: Extract key modifiers
        Find words that specify HOW, WHEN, WHERE, HOW MUCH. These are easy to miss:

        Time-related:
        - "gradually", "slowly", "immediately"
        - "at 10pm", "for 30 minutes", "until tonight"
        - "today", "tomorrow", "this afternoon"

        Location:
        - "in the bedroom", "in the living room"

        Manner/degree:
        - "very", "extremely", "slightly"
        - "quietly", "brightly", "warmly"
        - "dim", "bright", "loud"

        Quantity/negation:
        - "all", "some", "half"
        - "no music", "don't", "without"

        Examples:

        Input: "I want the lights to dim gradually starting at 10pm"
        {{
        "infos": ["I want the lights to dim gradually starting at 10pm"],
        "key_modifiers": ["gradually", "starting at 10pm", "dim"]
        }}

        Input: "Play some quiet music in the bedroom for 30 minutes"
        {{
        "infos": ["Play some quiet music in the bedroom for 30 minutes"],
        "key_modifiers": ["quiet", "in the bedroom", "for 30 minutes"]
        }}

        Input: "I'm tired and need to relax"
        {{
        "infos": ["I'm tired", "need to relax"],
        "key_modifiers": []
        }}

        Input: "Turn on very bright lights, no music please"
        {{
        "infos": ["Turn on very bright lights", "no music please"],
        "key_modifiers": ["very bright", "no music"]
        }}

        {format_instructions}

        Output ONLY valid JSON, no markdown code blocks, no explanations, no extra comma

        Output format:
        {{
        "infos": ["info1", "info2"],
        "key_modifiers": ["modifier1", "modifier2"]
        }}
        """,
        input_variables=["user_message"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    result = chain.invoke({
        "user_message": user_message,
    })

    infos = result.get("infos")
    complexity_score = len(infos)
    key_modifiers = result.get("key_modifiers",[])

    return Command(
        update={
            "complexity_score": complexity_score,
            "infos": infos,
            "key_modifiers": key_modifiers,
            "original_user_input": user_message,
        },
        goto="task_planner"
    )

def task_planner(state: SmartHomeState) -> Command:

    original_input = state.get('original_user_input')
    task_queue = state.get('task_queue', [])

    # Continue executing the remaining tasks
    if task_queue:
        current_task = task_queue[0]
        return Command(
            update={
                "task_queue": task_queue # new for log
            },
            goto=f"{current_task['device']}_agent"
        )

    # Fresh start, original_input is not empty
    if original_input:

        infos = state.get('infos', [])
        key_modifiers = state.get('key_modifiers', [])

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are the task planner for a smart home system.

            User input: {original_input}

            Reference: Infos = {infos}, Key modifiers = {key_modifiers} (Use as reference, but trust original user input if conflict)

            Your responsibility is to identify the user's main goal and assign it to the most appropriate device as a high-level task.

            Rule for task planner:
            1. Assign each goal to ONE primary device
            2. Describe WHAT needs to be done, PRESERVING ALL KEY DETAILS from user's request
            - Time references: "today", "tonight", "next Monday", "this weekend", "tomorrow"
            - Quantities: "for 6 people", "quick", "large"
            - Constraints: "vegan", "easy", "outdoor", "under 30 minutes"
            - Locations: "near me", "in the bedroom"
            3. DO NOT predict or specify collaboration between devices
            4. Trust individual agents to determine if they need help from other devices

            Available devices and their capabilities:
            - lighting: adjust lights, create atmosphere through lighting
            - thermostat: temperature control, create atmosphere through temperature
            - audio_system: play music, adjust volume, create atmosphere through audio_system
            - clock: provide current time, set alarms and timers, start or stop stopwatch
            - calendar: add schedule/reminders, provide information about schedule
            - fridge: provide food inventory (doesn't know any recipes)
            - search_engine: general information, recipes information, weather
            - tv_display: show/display visual content

            EXAMPLES:

            # === CALENDAR & SCHEDULE SCENARIOS ===
            Input: "What's on my calendar today?"
            {{"task_queue": [
                {{"device": "calendar", "action": "What's on my calendar today?"}}
            ]}}

            Input: "Where is the location for my next appointment?"
            {{"task_queue": [
                {{"device": "calendar", "action": "check the location of my next appointment"}}
            ]}}

            Input: "What time is my next appointment?"
            {{"task_queue": [
                {{"device": "calendar", "action": "check the start time of my next appointment"}}
            ]}}

            # === TIME & ALARM SCENARIOS ===

            # Set Alarm
            Input: "I need to wake up at tomorrow 7am"
            {{"task_queue": [
                {{"device": "clock", "action": "set alarm at 7am for wake up"}}
            ]}}

            # === MUSIC SCENARIOS ===

            "play relaxing music"
            Note: User only mentioned music
            {{"task_queue": [{{"device": "audio_system", "action": "play relaxing music"}}]}}

            # === FOOD & COOKING SCENARIOS ===

            # Hungry
            Input: "Do we have any milk? Is it about to expire?"
            {{"task_queue": [
                {{"device": "fridge", "action": "Is there any milk in the fridge? If there is milk, is it about to expire?"}}
            ]}}

            Input: "What dishes can I make based on the ingredients I have in the fridge?"
            {{"task_queue": [
                {{"device": "fridge", "action": "what's in the fridge"}},
                {{"device": "search_engine", "action": "suggest recipes using ingredients you already have"}}
            ]}}

            # === INFORMATION SEARCH SCENARIOS ===

            Input: "recommend some songs"
            {{"task_queue": [{{"device": "search_engine", "action": "recommend songs"}}]}}

            Input: "what music should I listen to?"
            {{"task_queue": [{{"device": "search_engine", "action": "suggest music recommendations"}}]}}

            # General Recipe Search
            Input: "Find me pasta recipe"
            {{"task_queue": [
                {{"device": "search_engine", "action": "find a pasta recipe"}}
            ]}}

            # General Information Search
            Input: "At what time does the New Year typically begin?"
            {{"task_queue": [
                {{"device": "search_engine", "action": "At what time does the New Year typically begin?"}}
            ]}}

            Input: "I want to make 'Fried Rice'"
            NOTE: When a user mentions a specific dish name, it should be assigned to the search engine rather than the fridge, as the fridge does not know the recipe for that particular dish.
            {{"task_queue": [
                {{"device": "search_engine", "action": "find a 'Fried Rice' recipe"}}
            ]}}

             # Hungry
            Input: "I'm hungry"
            {{"task_queue": [
                {{"device": "search_engine", "action": "user is hungry, suggest quick meal options with available food"}}
            ]}}

            # Meal Planning
            Input: "What should I cook tonight?"
            {{"task_queue": [
                {{"device": "search_engine", "action": "suggest dinner recipes using available ingredients"}}
            ]}}

            # === DISPLAY SCENARIOS ===

            Input: "I want to watch TV shows"
            {{"task_queue": [
                {{"device": "tv_display", "action": "display TV shows content"}}
            ]}}

            Input: "display something on the screen"
            {{"task_queue": [
                {{"device": "tv_display", "action": "display something on the screen"}}
            ]}}

            # === MULTI-ASPECT SCENARIOS ===

            Input: "play relaxing music for 30 minutes"
            {{"task_queue": [
                {{"device": "audio_system", "action": "play relaxing music"}},
                {{"device": "clock", "action": "set timer for 30 minutes to stop music"}}
            ]}}

            Input: "play music for 1 hour"
            {{"task_queue": [
                {{"device": "audio_system", "action": "play music"}},
                {{"device": "clock", "action": "set timer for 1 hour to stop music"}}
            ]}}

            Input: "show me a movie until 10pm"
            {{"task_queue": [
                {{"device": "tv_display", "action": "show me a movie"}},
                {{"device": "clock", "action": "set reminder at 10pm to stop watching"}}
            ]}}

            # Schedule + Environment Setup

            Input: "Show me my schedule and prepare the room for meetings"
            {{"task_queue": [
                {{"device": "calendar", "action": "display today's schedule"}},
                {{"device": "lighting", "action": "set bright lighting for meetings"}},
                {{"device": "thermostat", "action": "set comfortable temperature for meetings"}}
            ]}}

            # Context Preservation - Event Type
            Input: "I'm hosting a baby shower this afternoon. Get everything ready."
            {{"task_queue": [
                {{"device": "lighting", "action": "create welcoming atmosphere for baby shower with soft cheerful lighting"}},
                {{"device": "thermostat", "action": "create comfortable temperature for baby shower guests"}},
                {{"device": "audio_system", "action": "create pleasant atmosphere for baby shower with gentle background music"}}
            ]}}

            # Context Preservation - Activity Purpose
            Input: "I'm preparing for an important job interview via video call soon. Help me get ready."
            {{"task_queue": [
                {{"device": "lighting", "action": "create professional atmosphere for video interview with optimal lighting"}},
                {{"device": "thermostat", "action": "create comfortable temperature for job interview preparation"}},
                {{"device": "clock", "action": "set an alarm 1 hour before job interview for preparation time"}}
            ]}}

            # === CREATE ATMOSPHERE SCENARIOS ===
            # Comfortable scenario
            Input:"I'm tired and need relax"
            {{"task_queue": [
                {{"device": "audio_system", "action": "play relaxing music"}},
                {{"device": "lighting", "action": "dim the lighting to help users relax"}},
                {{"device": "thermostat", "action": "set a comfortable temperature for users to relax better"}}
            ]}}

            # Work Environment
            Input: "Make the room comfortable for working"
            {{"task_queue": [
                {{"device": "lighting", "action": "create bright lighting for a comfortable work environment"}},
                {{"device": "thermostat", "action": "set comfortable temperature for work"}},
                {{"device": "audio_system", "action": "play calming sounds for work"}}
            ]}}

            # Sleep Environment
            Input: "I'm going to bed soon"
            {{"task_queue": [
                {{"device": "lighting", "action": "prepare turn off the light for sleep"}},
                {{"device": "thermostat", "action": "set comfortable temperature for sleep"}},
                {{"device": "audio_system", "action": "play calming sounds for sleep"}}
            ]}}

            # Semantic Atmosphere Recognition - User describes feeling/state
            Input: "I just woke up and feel groggy. Help me get energized for the day ahead."
            {{"task_queue": [
                {{"device": "lighting", "action": "create energizing morning atmosphere with bright lighting to help wake up"}},
                {{"device": "thermostat", "action": "create comfortable temperature for active morning"}},
                {{"device": "audio_system", "action": "create motivating atmosphere with upbeat morning music"}}
            ]}}

            # Semantic Atmosphere Recognition - User describes desired outcome
            Input: "I want to create the perfect reading nook atmosphere in the living room."
            {{"task_queue": [
                {{"device": "lighting", "action": "create cozy reading atmosphere with warm focused lighting in living room"}},
                {{"device": "thermostat", "action": "create comfortable temperature for extended reading"}},
                {{"device": "audio_system", "action": "create calm reading atmosphere with soft instrumental background music"}}
            ]}}

            {format_instructions}

            Format rules:
            1. Every task MUST have both "device" and "action" fields
            2. Output ONLY valid JSON, no markdown code blocks, no explanations
            3. Format: {{"task_queue": [{{"device": "device_name", "action": "complete description with context"}}]}}
            4. Include relevant details from user input in the action description

            Output format: {{"task_queue": [{{"device": "device_name", "action": "what to do with full context"}}]}}

            """,

            input_variables=["original_input", "infos", "key_modifiers"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_input": original_input,
            "infos": infos,
            "key_modifiers": key_modifiers,
        })
        new_task_queue = result.get("task_queue")

        #print(f"DEBUG: LLM result = {result}")
        current_task = new_task_queue[0]
        return Command(
            update={
                "task_queue": new_task_queue,
                "original_user_input": ""
            },
            goto=f"{current_task['device']}_agent"
        )

    #print("DEBUG: All tasks completed")
    return Command(
        update={"task_queue": []} # new for log
    )
def clock_agent(state: SmartHomeState) -> Command:
    task_queue = state.get("task_queue", [])
    collaboration_request = state.get("collaboration_request")
    pending_task = state.get("pending_task")
    task_history = state.get("task_history",[])

    # Branch 1: Respond to collaboration requests from other agents
    if collaboration_request and collaboration_request.get("target") == "clock":
        requester = collaboration_request.get("requester")
        request = collaboration_request.get("request")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Clock Agent.

            Your capabilities:
            1. Provide current time
            2. Set or cancel alarms with default alarm sound
            3. Set or cancel timers
            4. Start or stop a stopwatch

            You received a collaboration request from {requester} agent.
            Request: {request}

            Provide the requested information directly. Simulate reasonable time data.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your response"}}
            """,
            input_variables=["requester", "request"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "requester": requester,
            "request": request,
        })
        clock_response = result.get("response")

        new_entry = {
            "device": "clock",
            "type": "collaboration_response",
            "action_taken": request,
            "result": clock_response,
        }
        return Command(
            update={
                "clock_response": clock_response,
                "collaboration_request": {},
                "task_history": task_history + [new_entry],
            },
            goto=f"{requester}_agent"
        )

    # Branch 2: Handling Collaborative Responses
    elif pending_task and pending_task.get("device") == "clock":
        collaborator = pending_task.get("waiting_for")
        response_key = f"{collaborator}_response"
        collaborator_response = state.get(response_key)
        original_action = pending_task.get("action")


        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Clock Agent completing a task with collaboration information.

            Your capabilities:
            1. Provide current time
            2. Set or cancel alarms with default alarm sound
            3. Set or cancel timers
            4.Start or stop a stopwatch

            Original task: {original_action}
            Task history (what happened before this):{task_history}
            Collaboration request：{collaboration_request}
            Response from {collaborator}: {collaborator_response}

            Now complete the task using these information without asking user. Simulate reasonable time data.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "task completion message"}}
            """,
            input_variables=["original_action","task_history","collaboration_request", "collaborator", "collaborator_response"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_action": original_action,
            "task_history": task_history,
            "collaboration_request": collaboration_request,
            "collaborator": collaborator,
            "collaborator_response": collaborator_response
        })

        clock_result = result.get("response")
        remaining_tasks = task_queue[1:]
        new_entry = {
            "device": "clock",
            "type": "task_completion",
            "action_taken": original_action,
            "result": clock_result,
        }
        return Command(
            update={
                "clock_result": clock_result,
                "task_queue": remaining_tasks,
                "pending_task": None,
                "collaboration_request": {},
                f"{collaborator}_response": None,  # 清空临时协作响应
                "task_history": task_history + [new_entry],
            },
            goto="task_planner"
        )

    # Branch 3: Handling New Tasks
    elif task_queue and task_queue[0].get("device") == "clock":
        action = task_queue[0].get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Clock Agent.

            Your capabilities:
            1. Provide current time
            2. Set or cancel alarms with default alarm sound
            3. Set or cancel timers
            4. Start or stop a stopwatch

            Current task: {action}
            Task history: {task_history} which you will know what other device already done

            Important: Check task_history first before requesting collaboration
            1. Review the task_history carefully
            2. Check if another agent has already provided the information you need
            3. Only request collaboration if the required information is genuinely NOT in task_history

            Decide: Can you complete this independently with your capabilities and task history?
            If YES: Complete the task directly without asking the user
            If NO: Identify what you need and request help from appropriate agent

            Don't ask the user for clarification. Make reasonable assumptions when needed.

            Other agents available for collaboration:
            calendar (provide information(time,location,with who) about schedule/events), audio_system (music), lighting (adjust lights and create lighting scenes),tv_display(show information),thermostat(temperature control),search_engine(provide external information),fridge(food inventory)

            Examples:

            action: "set alarm for tomorrow 7am"
            {{"response": "Alarm set for 7:00 AM on Saturday, September 27, 2025", "collaboration_request": {{}}}}

            action: "remind me 10 minutes before my next meeting"
            {{"response": "", "collaboration_request": {{"target": "calendar", "request": "It is now 2 PM. What time is my next scheduled meeting today?"}}}}

            action: "remind me next event after 30 minutes"
            {{"response": "", "collaboration_request": {{"target": "calendar", "request": "It is now 1 PM. What time is my next scheduled meeting today?"}}}}

            {format_instructions}

            Output ONLY pure JSON.
            Output format: {{"response": "your result" or "", "collaboration_request": {{"target": "agent_name", "request": "what you need"}} or {{}}}}
            """,
            input_variables=["action","task_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "action": action,
            "task_history": task_history,
        })

        if result.get("collaboration_request") and result["collaboration_request"].get("target"):
            collaboration = result["collaboration_request"]

            new_entry = {
                "device": "clock",
                "type": "collaboration_request",
                "action_taken": action,
                "result": {
                    "target": collaboration["target"],
                    "request": collaboration["request"],
                },
            }
            return Command(
                update={
                    "collaboration_request": {
                        "requester": "clock",
                        "target": collaboration["target"],
                        "request": collaboration["request"],
                    },
                    "pending_task": {
                        "device": "clock",
                        "action": action,
                        "waiting_for": collaboration["target"]
                    },
                    "task_history": task_history + [new_entry],
                },
                goto=f"{collaboration['target']}_agent"
            )
        else:
            remaining_tasks = task_queue[1:]
            clock_result = result.get("response")
            new_entry = {
                "device": "clock",
                "type": "task_completion",
                "action_taken": action,
                "result": clock_result,
            }
            return Command(
                update={
                    "clock_result": clock_result,
                    "task_queue": remaining_tasks,
                    "task_history": task_history + [new_entry],
                },
                goto="task_planner"
            )

def search_engine_agent(state: SmartHomeState) -> Command:
    task_queue = state.get("task_queue", [])
    collaboration_request = state.get("collaboration_request")
    pending_task = state.get("pending_task")
    task_history = state.get("task_history",[])

    # branch 1
    if collaboration_request and collaboration_request.get("target") == "search_engine":
        requester = collaboration_request.get("requester")
        request = collaboration_request.get("request")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Search Engine Agent.

            Your capabilities:
            1. Provide weather information (any time: past, present, future)
            2. Provide recipes and cooking information
            3. Provide general information and knowledge
            4. Provide home management tips and advice

            You received a collaboration request from {requester} agent.
            Request: {request}

            Provide the information they need directly. Simulate a reasonable search result.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            Response format: Plain text, items separated by commas, no quotation marks or special formatting.

            Examples:
            Request: "find restaurants"
            {{"response": "Restaurants nearby: Luigi's Pizza at Main Street 10, Sushi House at Park Ave 25, Burger Palace at Market Square 5"}}

            {format_instructions}

            Output format: {{"response": "your simulated search result"}}
            """,
            input_variables=["requester", "request"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "requester": requester,
            "request": request
        })
        search_engine_response = result.get("response")
        new_entry = {
            "device": "search_engine",
            "type": "collaboration_response",
            "action_taken": request,
            "result": search_engine_response,
        }
        return Command(
            update={
                "search_engine_response": search_engine_response,
                "collaboration_request": {},
                "task_history": task_history + [new_entry],
            },
            goto=f"{requester}_agent"
        )

    # branch 2
    elif pending_task and pending_task.get("device") == "search_engine":
        collaborator = pending_task.get("waiting_for")
        response_key = f"{collaborator}_response"
        collaborator_response = state.get(response_key)
        original_action = pending_task.get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Search Engine Agent completing a task with collaboration information.

            Your capabilities:
            1. Provide weather information (any time: past, present, future)
            2. Provide recipes and cooking information
            3. Provide general information and knowledge
            4. Provide home management tips and advice

            Original task: {original_action}
            Task history (what happened before this): {task_history}
            The content of your collaboration request：{collaboration_request}
            Information received from {collaborator}: {collaborator_response}

            Response format: Plain text, items separated by commas, no quotation marks or special formatting.

            Now complete the task using these information without asking user.
            Provide a simulated search result.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "search result"}}
            """,
            input_variables=["original_action","task_history","collaboration_request", "collaborator", "collaborator_response"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_action": original_action,
            "task_history": task_history,
            "collaboration_request": collaboration_request,
            "collaborator": collaborator,
            "collaborator_response": collaborator_response
        })
        search_engine_result = result.get("response")
        remaining_tasks = task_queue[1:]
        new_entry = {
            "device": "search_engine",
            "type": "task_completion",
            "action_taken": original_action,
            "result": search_engine_result,
        }
        return Command(
            update={
                "search_engine_result": search_engine_result,
                "task_queue": remaining_tasks,
                "pending_task": None,
                "collaboration_request": {},
                f"{collaborator}_response": None,
                "task_history": task_history + [new_entry],
            },
            goto="task_planner"
        )

    # branch 3
    elif task_queue and task_queue[0].get("device") == "search_engine":
        action = task_queue[0].get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Search Engine Agent.

            Your capabilities:
            1. Provide weather information (any time: past, present, future)
            2. Provide recipes and cooking information
            3. Provide general information and knowledge
            4. Provide home management tips and advice

            Current task: {action}
            Default location: Hamburg, Germany
            Task history: {task_history} which you will know what other device already done

            Important:
            1. Always provide simulated results - never say information is 'unavailable'
            2. Never ask the user questions or offer to do additional searches
            3. Make reasonable assumptions and provide complete answers directly
            4. For general advice questions, provide answers without needing other agents
            5. When advice involves weather, include simulated weather data directly

            SPECIAL RULE FOR RECIPE SEARCHES:
            When the task involves finding recipes 'based on available ingredients' or 'based on what's in fridge':
            Check task_history first: Has fridge already provided ingredient information?
            If YES: Use those ingredients to suggest recipes
            If NO: Request collaboration to get available ingredients

            Response format: Plain text, items separated by commas, no quotation marks or special formatting.

            Decide: Can you complete this independently with your capabilities and task history?

            If YES: Provide complete simulated results
            If NO: Only collaborate if you need SPECIFIC data you cannot simulate

            Other agents available for collaboration:
            tv_display (show visual content on screens), calendar(check/add appointments and schedule),clock (check time, alarms, timers), fridge (food inventory), lighting (lights control), thermostat (temperature control), audio_system (music or volume control)

            Examples:

            action: "find current weather in Hamburg"
            {{"response": "Hamburg weather: 18°C, partly cloudy, light breeze. Expected high of 22°C today.", "collaboration_request": {{}}}}

            action: "find simple recipes using chicken and rice"
            {{"response": "Found 3 simple recipes: 1) Chicken Rice Bowl, 2) One-Pot Chicken and Rice, 3) Asian Chicken Fried Rice. Each takes 30-40 minutes.", "collaboration_request": {{}}}}

            action: "recommend music"
            {{"response": "Music recommendations: Blinding Lights by The Weeknd, As It Was by Harry Styles, Break My Heart by Dua Lipa, Industry Baby by Lil Nas X, Heat Waves by Glass Animals", "collaboration_request": {{}}}}

            action: "recommend cozy atmosphere music"
            {{"response": "For a cozy atmosphere, try instrumental music with mellow tempos and warm tones. Suggestions: Weightless by Marconi Union, Clair de Lune by Claude Debussy, Nuvole Bianche by Ludovico Einaudi, Watermark by Enya", "collaboration_request": {{}}}}

            action: "suggest recipes based on ingredients"
            Note: This requires current available food information in fridge
            {{"response": "", "collaboration_request": {{"target": "fridge", "request": "list available ingredients for meal planning"}}}}

            action: "suggest recipes using available ingredients"
            {{"response": "", "collaboration_request": {{"target": "fridge", "request": "check what food items are available"}}}}

            action: "what's the weather like at my next scheduled location?"
            {{"response": "", "collaboration_request": {{"target": "calendar", "request": "check the location for the next schedule"}}}}

            {format_instructions}

            Output ONLY JSON
            Output format: {{"response": "your search result", "collaboration_request": {{}} }}
            """,
            input_variables=["action","task_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "action": action,
            "task_history": task_history,
        })

        if result.get("collaboration_request") and result["collaboration_request"].get("target"):
            collaboration = result["collaboration_request"]

            new_entry = {
                "device": "search_engine",
                "type": "collaboration_request",
                "action_taken": action,
                "result": {
                    "target" : collaboration["target"],
                    "request" : collaboration["request"],
                },
            }
            return Command(
                update={
                    "collaboration_request": {
                        "requester": "search_engine",
                        "target": collaboration["target"],
                        "request": collaboration["request"],
                    },
                    "pending_task": {
                        "device": "search_engine",
                        "action": action,
                        "waiting_for": collaboration["target"]
                    },
                    "task_history": task_history + [new_entry],
                },
                goto=f"{collaboration['target']}_agent"
            )
        else:
            remaining_tasks = task_queue[1:]
            search_engine_result = result.get("response")
            new_entry = {
                "device": "search_engine",
                "type": "task_completion",
                "action_taken": action,
                "result": search_engine_result,
            }
            return Command(
                update={
                    "search_engine_result": search_engine_result ,
                    "task_queue": remaining_tasks,
                    "task_history": task_history + [new_entry],
                },
                goto="task_planner"
            )
def calendar_agent(state: SmartHomeState) -> Command:
    task_queue = state.get("task_queue", [])
    collaboration_request = state.get("collaboration_request")
    pending_task = state.get("pending_task")
    task_history = state.get("task_history",[])

    if collaboration_request and collaboration_request.get("target") == "calendar":
        requester = collaboration_request.get("requester")
        request = collaboration_request.get("request")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Calendar Agent.

            Your capabilities:
            1. Add appointments/reminders/meeting
            2. Cancel or reschedule appointments
            3. Provide information about schedule/appointments/reminders (time, location, with who)

            You received a collaboration request from {requester} agent.
            Request: {request}

            Provide schedule information directly. Simulate reasonable calendar data.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your simulated schedule information"}}
            """,
            input_variables=["requester", "request"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "requester": requester,
            "request": request,
        })

        calendar_response = result.get("response")
        new_entry = {
            "device": "calendar",
            "type": "collaboration_response",
            "action_taken": request,
            "result": calendar_response,
        }
        return Command(
            update={
                "calendar_response": calendar_response,
                "collaboration_request": {},
                "task_history": task_history + [new_entry],
            },
            goto=f"{requester}_agent"
        )

    elif pending_task and pending_task.get("device") == "calendar":

        collaborator = pending_task.get("waiting_for")
        response_key = f"{collaborator}_response"
        collaborator_response = state.get(response_key)
        original_action = pending_task.get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Calendar Agent completing a task with collaboration information.

            Your capabilities:
            1. Add appointments/reminders/meeting
            2. Cancel or reschedule appointments
            3. Provide information about schedule/appointments/reminders (time, location, with who)

            Original task: {original_action}
            Task history (what happened before this task):{task_history}
            Collaboration request：{collaboration_request}
            Request from {collaborator}: {collaborator_response}

            Now simulate the calendar operation and provide the result and make reasonable assumptions.
            Don't ask the user questions

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "task completion message"}}
            """,
            input_variables=["original_action","task_history","collaboration_request","collaborator", "collaborator_response"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_action": original_action,
            "task_history": task_history,
            "collaboration_request": collaboration_request,
            "collaborator": collaborator,
            "collaborator_response": collaborator_response
        })
        calendar_result = result.get("response")
        remaining_tasks = task_queue[1:]
        new_entry = {
            "device": "calendar",
            "type": "task_completion",
            "action_taken": original_action,
            "result": calendar_result,
        }
        return Command(
            update={
                "calendar_result": calendar_result,
                "task_queue": remaining_tasks,
                "pending_task": None,
                "collaboration_request": {},
                f"{collaborator}_response": None,
                "task_history": task_history + [new_entry],
            },
            goto="task_planner"
        )

    elif task_queue and task_queue[0].get("device") == "calendar":
        action = task_queue[0].get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Calendar Agent.

            Your capabilities:
            1. Add appointments/reminders/meeting
            2. Cancel or reschedule appointments
            3. Provide information about schedule/appointments/reminders (event time, location, with who)

            Current task: {action}
            Task history : {task_history} which you will know what other device already done

            Important: Check task_history first before requesting collaboration
            1. Review what other agents have already done
            2. Only request collaboration if needed information is not in task_history

            Decide: Can you complete this independently with your capabilities and task history?

            If YES: Complete the task directly without asking user
            If NO: Identify what you need and request help from appropriate agent

            Other agents available for collaboration:
            tv_display (show/display content on screens), search_engine (look up information),clock (get current time, alarms, timers), fridge (food), lighting (lights), thermostat (temperature),audio_system (music)

            Examples:

            # Adding appointments
            action: "add a dentist appointment for next Tuesday at 3pm"
            {{"response": "Added 'Dentist Appointment' for next Tuesday at 3:00 PM", "collaboration_request": {{}}}}

            action: "check availability for this weekend"
            {{"response": "This weekend is free", "collaboration_request": {{}}}}

            # Checking schedule
            action: "what's on my calendar today?"
            {{"response": "", "collaboration_request": {{"target": "tv_display", "request": "Display today's schedule: Online meeting at 9:00 AM, Have lunch with Sarah at 'Mama's Burger' at 1 PM, Project review at 3 PM in your office"}}}}

            action: "check the location of my next appointment"
            Note: calendar doesn't know what time it is, so it can't directly provide information about the next event
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "What time is it now? That way I can confirm what the next schedule is"}}}}

            action: "do I have any meetings tomorrow?"
            Note：There's no need to ask what time it is now
            {{"response": "Yes, you have 2 meetings tomorrow: 9 AM Team Meeting and 2 PM Client Call", "collaboration_request": {{}}}}

            action: "check if I am free this Friday night"
            {{"response": "You are free this Friday night", "collaboration_request": {{}}}}

            action: "show my schedule on the screen"
            {{"response": "", "collaboration_request": {{"target": "tv_display", "request": "Display today's calendar: 9am Team Standup, 1pm Lunch, 3pm Review"}}}}

            action: "What time does my next meeting start?"
            Note: calendar doesn't know what time it is, so it can't directly provide information about the next event
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "what time is it now?"}}}}

            action: "display schedule"
            {{"response": "", "collaboration_request": {{"target": "tv_display", "request": "Display today's calendar: 9am Team Standup, 1pm Lunch, 3pm Review"}}}}

            action: "show my schedule"
            {{"response": "", "collaboration_request": {{"target": "tv_display", "request": "Display today's calendar: 9am Team Standup, 1pm Lunch, 3pm Review"}}}}

            # Need external information
            action: "find a good time to eat dinner next week and add it to my calendar"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "What are the operating hours of the restaurants near me for dinner?"}}}}

            # Canceling
            action: "cancel tomorrow's dentist appointment"
            {{"response": "Cancelled the dentist appointment for tomorrow", "collaboration_request": {{}}}}

            {format_instructions}

            CRITICAL: Output ONLY pure JSON.

            Output format: {{"response": "your result" or "", "collaboration_request": {{"target": "agent_name", "request": "what you need"}} or {{}}}}
            """,
            input_variables=["action","task_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "action": action,
            "task_history": task_history,
        })

        if result.get("collaboration_request") and result["collaboration_request"].get("target"):
            collaboration = result["collaboration_request"]
            new_entry = {
                "device": "calendar",
                "type": "collaboration_request",
                "action_taken": action,
                "result":{
                    "target": collaboration["target"],
                    "request": collaboration["request"],
                }
            }
            return Command(
                update={
                    "collaboration_request": {
                        "requester": "calendar",
                        "target": collaboration["target"],
                        "request": collaboration["request"],
                    },
                    "pending_task": {
                        "device": "calendar",
                        "action": action,
                        "waiting_for": collaboration["target"]
                    },
                    "task_history": task_history + [new_entry],
                },
                goto=f"{collaboration['target']}_agent"
            )
        else:
            remaining_tasks = task_queue[1:]
            calendar_result = result.get("response")
            new_entry = {
                "device": "calendar",
                "type": "task_completion",
                "action_taken": action,
                "result": calendar_result,
            }
            return Command(
                update={
                    "calendar_result": calendar_result,
                    "task_queue": remaining_tasks,
                    "task_history": task_history + [new_entry],
                },
                goto="task_planner"
            )

def tv_display_agent(state: SmartHomeState) -> Command:
    task_queue = state.get("task_queue", [])
    collaboration_request = state.get("collaboration_request")
    pending_task = state.get("pending_task")
    task_history = state.get("task_history",[])

    if collaboration_request and collaboration_request.get("target") == "tv_display":
        requester = collaboration_request.get("requester")
        request = collaboration_request.get("request")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Display Agent.

            Your capability: Display/Show information in a clear, visual format on the screen

            You received a collaboration request from {requester} agent.
            Request: {request}

            Simulate displaying the requested content.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "confirmation of what you displayed"}}
            """,
            input_variables=["requester", "request"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "requester": requester,
            "request": request
        })
        tv_display_response = result.get("response")
        new_entry = {
            "device": "tv_display",
            "type": "collaboration_response",
            "action_taken": request,
            "result": tv_display_response,
        }
        return Command(
            update={
                "tv_display_response": tv_display_response,
                "collaboration_request": {},
                "task_history": task_history + [new_entry],
            },
            goto=f"{requester}_agent"
        )

    # 分支2: 处理协作响应（有pending_task）
    elif pending_task and pending_task.get("device") == "tv_display":
        collaborator = pending_task.get("waiting_for")
        response_key = f"{collaborator}_response"
        collaborator_response = state.get(response_key)
        original_action = pending_task.get("action")


        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Display Agent completing a task with collaboration information.

            Your capability: Display/Show information in a clear, visual format on the screen

            Original task: {original_action}
            Task history (what happened before this):{task_history}
            Collaboration request：{collaboration_request}
            Request from {collaborator}: {collaborator_response}

            Now display the content using this information. Simulate the display operation.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "confirmation of what was displayed"}}
            """,
            input_variables=["original_action","task_history","collaboration_request", "collaborator", "collaborator_response"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_action": original_action,
            "task_history": task_history,
            "collaboration_request": collaboration_request,
            "collaborator": collaborator,
            "collaborator_response": collaborator_response
        })

        remaining_tasks = task_queue[1:]
        tv_display_result = result.get("response")
        new_entry = {
            "device": "tv_display",
            "type": "task_completion",
            "action_taken": original_action,
            "result": tv_display_result,
        }
        return Command(
            update={
                "tv_display_result": tv_display_result ,
                "task_queue": remaining_tasks,
                "pending_task": None,
                "collaboration_request": {},
                f"{collaborator}_response":None,
                "task_history": task_history + [new_entry],
            },
            goto="task_planner"
        )

    # 分支3: 处理新任务
    elif task_queue and task_queue[0].get("device") == "tv_display":
        action = task_queue[0].get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home TV Display Agent.

            Your capability: Display ANY visual content on the TV screen (entertainment, information, schedules, recipes, etc.)

            Current task: {action}
            Task history: {task_history} which you will know what other device already done

            Decision rules:
            Step 1: What TYPE of content does the user want to display?

            1. Entertainment content (movies, shows, videos, etc.)
            2. Information content (schedules, recipes, food inventory, weather, etc.)
            3. Simple messages (welcome, notifications, etc.)

            Step 2: Can you complete this task independently using your own capabilities ?
            First check task history, please note whether there are any other needs.
            If YES: Display it directly
            If NO: Request collaboration from the appropriate agent

            Other agents available for collaboration:
            search_engine (look up information, weather, recipes), calendar (schedules, events), clock (time, alarms, set timers), fridge (available food), lighting (light control), thermostat (temperature control), audio_system (playlist info)

            Examples:

            # Entertainment: need recommendations
            action: "show me a comedy"
            Note: Need recommendations first
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "recommend popular comedy"}}}}

            action: "show something"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "recommend popular content to watch"}}}}

            action: "play The Stranger Things for 3 hours"
            Note: Need collaboration from clock agent
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "Set a timer for the three-hour watch of The Stranger Things"}}}}

            action: "find and show a good action movie"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "recommend a good action movie"}}}}

            # Entertainment - exact title
            action: "play Titanic"
            {{"response": "Now playing: Titanic", "collaboration_request": {{}}}}

            # Schedule/calendar display
            action: "display today's schedule"
            {{"response": "", "collaboration_request": {{"target": "calendar", "request": "get today's schedule and appointments"}}}}

            # Information/Recipe display
            action: "show cooking instructions on TV"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "get cooking instructions"}}}}

            # Fridge/food inventory display
            action: "display available ingredients"
            {{"response": "", "collaboration_request": {{"target": "fridge", "request": "get available ingredients"}}}}

            # Time/timer display
            action: "display timer on TV"
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "get timer status"}}}}

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your result" or "", "collaboration_request": {{"target": "agent_name", "request": "what you need"}} or {{}}}}
            """,
            input_variables=["action","task_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "action": action,
            "task_history": task_history,
        })

        if result.get("collaboration_request") and result["collaboration_request"].get("target"):
            collaboration = result["collaboration_request"]

            new_entry = {
                "device": "tv_display",
                "type": "collaboration_request",
                "action_taken": action,
                "result": {
                    "target": collaboration["target"],
                    "request": collaboration["request"],
                }
            }
            return Command(
                update={
                    "collaboration_request": {
                        "requester": "tv_display",
                        "target": collaboration["target"],
                        "request": collaboration["request"],
                    },
                    "pending_task": {
                        "device": "tv_display",
                        "action": action,
                        "waiting_for": collaboration["target"]
                    },
                    "task_history": task_history + [new_entry],
                },
                goto=f"{collaboration['target']}_agent"
            )
        else:
            remaining_tasks = task_queue[1:]
            tv_display_result = result.get("response")
            new_entry = {
                "device": "tv_display",
                "type": "task_completion",
                "action_taken": action,
                "result": tv_display_result,
            }
            return Command(
                update={
                    "tv_display_result": tv_display_result,
                    "task_queue": remaining_tasks,
                    "task_history": task_history + [new_entry]
                },
                goto="task_planner"
            )

def fridge_agent(state: SmartHomeState) -> Command:
    task_queue = state.get("task_queue")
    collaboration_request = state.get("collaboration_request")
    pending_task = state.get("pending_task")
    task_history = state.get("task_history",[])

    if collaboration_request and collaboration_request.get("target") == "fridge":
        requester = collaboration_request.get("requester")
        request = collaboration_request.get("request")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Fridge Agent.

            Your capabilities:
            1. Provide food inventory data (items, quantities, expiry dates)
            2. Alert about expiring items
            3. Provide available ingredients lists

            You received a collaboration request from {requester} agent.
            Request need: {request}

            Provide fridge information. Simulate reasonable food inventory data.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your simulated response"}}
            """,
            input_variables=["requester", "request"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "requester": requester,
            "request": request,
        })
        fridge_response = result.get("response")
        new_entry = {
            "device": "fridge",
            "type": "collaboration_response",
            "action_taken": request,
            "result": fridge_response,
        }
        return Command(
            update={
                "fridge_response": fridge_response,
                "collaboration_request": {},
                "task_history": task_history + [new_entry],
            },
            goto=f"{requester}_agent"
        )

    elif pending_task and pending_task.get("device") == "fridge":
        collaborator = pending_task.get("waiting_for")
        response_key = f"{collaborator}_response"
        collaborator_response = state.get(response_key)
        original_action = pending_task.get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Fridge Agent completing a task with collaboration information.

            1. Provide food inventory data (items, quantities, expiry dates)
            2. Alert about expiring items
            3. Provide available ingredients lists

            Original task: {original_action}
            Task history (what happened before this): {task_history}
            Collaboration request：{collaboration_request}
            Request from {collaborator}: {collaborator_response}

            Now complete the fridge task using this information. Simulate reasonable food inventory data.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "task completion message"}}
            """,
            input_variables=["original_action","task_history","collaboration_request","collaborator", "collaborator_response"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_action": original_action,
            "collaboration_request": collaboration_request,
            "task_history": task_history,
            #"food_inventory": food_inventory,
            "collaborator": collaborator,
            "collaborator_response": collaborator_response
        })

        fridge_result = result.get("response")
        remaining_tasks = task_queue[1:]
        new_entry = {
            "device": "fridge",
            "type": "task_completion",
            "action_taken": original_action,
            "result": fridge_result,
        }
        return Command(
            update={
                "fridge_result": fridge_result,
                "task_queue": remaining_tasks,
                "pending_task": None,
                "collaboration_request": {},
                f"{collaborator}_response":None,
                "task_history": task_history + [new_entry],
            },
            goto="task_planner"
        )

    elif task_queue and task_queue[0].get("device") == "fridge":
        action = task_queue[0].get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Fridge Agent.

            Your capabilities:
            1. Provide food inventory data (items, quantities, expiry dates)
            2. Alert about expiring items
            3. Provide available ingredients lists

            Important:
            1. NEVER say "not accessible" or "please provide inventory"
            2. NEVER ask the user for information.
            3. You DON'T know recipes or what ingredients are needed for specific dishes
            4. Simulate reasonable food inventory data.

            Current task: {action}
            Task history:{task_history} which you will know other device already done

            Important: Check task_history first before requesting collaboration.

            First, understand what this task requires.
            Then, decide based on {task_history}: Can you complete this task independently with your capabilities:
            If YES: Provide the result with current inventory data without asking the user questions
            If NO: Identify what you need help with and request help from appropriate agent

            Other agents available for collaboration:
            search_engine (information, recipes), calendar (scheduled events), clock (time, alarms, timers), lighting (light control), thermostat (temperature control), audio_system (music), tv_display(display/show content)

            Examples:

            action: "check what food items are available"
            {{"response": "Available: chicken 500g, rice 1kg, vegetables, eggs, milk", "collaboration_request": {{}}}}

            action: "list ingredients for meal planning"
            {{"response": "Current ingredients: chicken, rice, vegetables, pasta, tomato sauce", "collaboration_request": {{}}}}

            action: "alert about expiring items"
            {{"response": "Warning: milk expires in 2 days, yogurt expires tomorrow", "collaboration_request": {{}}}}

            "user is hungry, suggest quick meal options with available food"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "find quick meal recipes using beef, rice, and vegetables"}}}}

            action: "suggest recipes using available ingredients"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "find recipes using chicken, rice, and vegetables"}}}}

            action: "what's in fridge and how to cook"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "find recipes using chicken, rice, and vegetables"}}}}

            action: "check if I can make spaghetti carbonara with current ingredients"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "What ingredients are needed for spaghetti carbonara?"}}}}

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your result" or "", "collaboration_request": {{"target": "agent_name", "request": "what you need"}} or {{}} }}
            """,
            input_variables=["action","task_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "action": action,
            "task_history": task_history,
        })

        if result.get("collaboration_request") and result["collaboration_request"].get("target"):
            collaboration = result["collaboration_request"]
            new_entry = {
                "device": "fridge",
                "type": "collaboration_request",
                "action_taken": action,
                "result": {
                    "target":collaboration["target"],
                    "request":collaboration["request"],
                }
            }
            return Command(
                update={
                    "collaboration_request": {
                        "requester": "fridge",
                        "target": collaboration["target"],
                        "request": collaboration["request"]
                    },
                    "pending_task": {
                        "device": "fridge",
                        "action": action,
                        "waiting_for": collaboration["target"]
                    },
                    "task_history": task_history + [new_entry],
                },
                goto=f"{collaboration['target']}_agent"
            )
        else:
            # Task completed, remove the first task from the current task_queue
            remaining_tasks = task_queue[1:]
            fridge_result = result.get("response")
            new_entry = {
                "device": "fridge",
                "type": "task_completion",
                "action_taken": action,
                "result": fridge_result,
            }
            return Command(
                update={
                    "fridge_result": fridge_result,
                    "task_queue": remaining_tasks,
                    "task_history": task_history + [new_entry],
                },
                goto = "task_planner"
            )

def lighting_agent(state: SmartHomeState) -> Command:
    task_queue = state.get("task_queue", [])
    collaboration_request = state.get("collaboration_request")
    pending_task = state.get("pending_task")
    task_history = state.get("task_history",[])

    if collaboration_request and collaboration_request.get("target") == "lighting":
        requester = collaboration_request.get("requester")
        request = collaboration_request.get("request")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Lighting Agent.

            Your capabilities:
            1. Turn lights on/off
            2. Change light colors
            3. Adjust light brightness
            4. Set appropriate lighting for different activities (work, sleep, relaxation, eco, etc.)

            You received a collaboration request from {requester} agent.
            Request: {request}

            Simulate the lighting control operation.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your simulated lighting response"}}
            """,
            input_variables=["requester", "request"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "requester": requester,
            "request": request
        })
        lighting_response = result.get("response")
        new_entry = {
            "device": "lighting",
            "type": "collaboration_response",
            "action_taken": request,
            "result": lighting_response,
        }
        return Command(
            update={
                "lighting_response": lighting_response,
                "collaboration_request": {},
                "task_history": task_history + [new_entry],
            },
            goto=f"{requester}_agent"
        )

    elif pending_task and pending_task.get("device") == "lighting":
        collaborator = pending_task.get("waiting_for")
        response_key = f"{collaborator}_response"
        collaborator_response = state.get(response_key)
        original_action = pending_task.get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Lighting Agent completing a task with collaboration information.

            Your capabilities:
            1. Turn lights on/off
            2. Change light colors
            3. Adjust light brightness
            4. Set appropriate lighting for different activities (work, sleep, relaxation, eco, etc.)

            Original task: {original_action}
            Task history (what happened before this): {task_history}
            Collaboration request：{collaboration_request}
            Request from {collaborator}: {collaborator_response}

            Now complete the lighting task using this information. Simulate reasonable operation.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "task completion message"}}
            """,
            input_variables=["original_action","task_history","collaboration_request", "collaborator", "collaborator_response"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_action": original_action,
            "task_history": task_history,
            "collaboration_request": collaboration_request,
            "collaborator": collaborator,
            "collaborator_response": collaborator_response
        })
        lighting_result = result.get("response")
        remaining_tasks = task_queue[1:]
        new_entry = {
            "device": "lighting",
            "type": "task_completion",
            "action_taken": original_action,
            "result": lighting_result,
        }
        return Command(
            update={
                "lighting_result": lighting_result,
                "task_queue": remaining_tasks,
                "pending_task": None,
                "collaboration_request": {},
                f"{collaborator}_response": None,
                "task_history": task_history + [new_entry],
            },
            goto="task_planner"
        )

    elif task_queue and task_queue[0].get("device") == "lighting":
        action = task_queue[0].get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Lighting Agent.

            Your capabilities:
            1. Turn lights on/off
            2. Change light colors
            3. Adjust light brightness
            4. Set appropriate lighting for different activities (work, sleep, relaxation, eco, etc.)

            Current task: {action}
            Task history: {task_history} which you will know what other device already done

            IMPORTANT: Always provide complete solutions with specific settings.
            DO NOT ask users for additional information, infer reasonable values from context.

            First, understand what this task requires.
            Then, decide with {task_history}: Can you complete this task independently with your capabilities and task history?

            If YES: Simulate the lighting operation and provide the response.
            If NO: Identify what you need help with and which agent can provide it.

            Other agents available for collaboration:
            clock (time, alarms, timers), search_engine (information, weather, recipes), calendar (scheduled events),thermostat (temperature control), audio_system (music), tv_display(display/show content), fridge(food related)

            Examples:

            action: "turn on the lights"
            Note: Can do independently with power control
            {{"response": "Lights turned on", "collaboration_request": {{}}}}

            action: "set warm, bright lighting for reading"
            Note: Can do independently with brightness and color control
            {{"response": "Set warm white light at 80% brightness for comfortable reading", "collaboration_request": {{}}}}

            action: "adjust lights based on current time of day"
            Note: Need time information
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "what time is it now for appropriate lighting adjustment?"}}}}

            action: "turn on the light for 2 hour"
            Note: Need collaboration from clock agent
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "Set a timer for two hours to turn on the light"}}}}

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your result" or "", "collaboration_request": {{"target": "agent_name", "request": "what you need"}} or {{}}}}
            """,
            input_variables=["action","task_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "action": action,
            "task_history": task_history,
        })

        if result.get("collaboration_request") and result["collaboration_request"].get("target"):
            collaboration = result["collaboration_request"]
            new_entry = {
                "device": "lighting",
                "type": "collaboration_request",
                "action_taken": action,
                "result": {
                    "target": collaboration["target"],
                    "request": collaboration["request"],
                }
            }
            return Command(
                update={
                    "collaboration_request": {
                        "requester": "lighting",
                        "target": collaboration["target"],
                        "request": collaboration["request"],
                    },
                    "pending_task": {
                        "device": "lighting",
                        "action": action,
                        "waiting_for": collaboration["target"]
                    },
                    "task_history": task_history + [new_entry],
                },
                goto=f"{collaboration['target']}_agent"
            )
        else:
            remaining_tasks = task_queue[1:]
            lighting_result = result.get("response")
            new_entry = {
                "device": "lighting",
                "type": "task_completion",
                "action_taken": action,
                "result": lighting_result,
            }

            return Command(
                update={
                    "lighting_result": lighting_result,
                    "task_queue": remaining_tasks,
                    "task_history": task_history + [new_entry],
                },
                goto="task_planner"
            )

def thermostat_agent(state: SmartHomeState) -> Command:
    task_queue = state.get("task_queue", [])
    collaboration_request = state.get("collaboration_request")
    pending_task = state.get("pending_task")
    task_history = state.get("task_history",[])

    if collaboration_request and collaboration_request.get("target") == "thermostat":
        requester = collaboration_request.get("requester")
        request = collaboration_request.get("request")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Thermostat Agent.

            Your capabilities:
            1. Temperature control: Adjust heating and cooling
            2. Climate optimization: Set comfortable temperature levels
            3. Mode settings: Heat, cool, auto, eco modes

            You received a collaboration request from {requester} agent.
            Request: {request}

            Simulate the temperature control operation.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your simulated thermostat response"}}
            """,
            input_variables=["requester", "request"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "requester": requester,
            "request": request
        })
        thermostat_response = result.get("response")
        new_entry = {
            "device": "thermostat",
            "type": "collaboration_response",
            "action_taken": request,
            "result": thermostat_response,
        }
        return Command(
            update={
                "thermostat_response": thermostat_response,
                "collaboration_request": {},
                "task_history": task_history + [new_entry],
            },
            goto=f"{requester}_agent"
        )

    elif pending_task and pending_task.get("device") == "thermostat":
        collaborator = pending_task.get("waiting_for")
        response_key = f"{collaborator}_response"
        collaborator_response = state.get(response_key)
        original_action = pending_task.get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Thermostat Agent completing a task with collaboration information.

            Your capabilities:
            1. Temperature control: Adjust heating and cooling
            2. Climate optimization: Set comfortable temperature levels
            3. Mode settings: Heat, cool, auto, eco modes

            Original task: {original_action}
            Task history (what happened before this task):{task_history}
            Collaboration request：{collaboration_request}
            Request from {collaborator}: {collaborator_response}

            Now complete the thermostat task using this information. Simulate the operation.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "task completion message"}}
            """,
            input_variables=["original_action","task_history","collaboration_request", "collaborator", "collaborator_response"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_action": original_action,
            "task_history": task_history,
            "collaboration_request": collaboration_request,
            "collaborator": collaborator,
            "collaborator_response": collaborator_response
        })

        remaining_tasks = task_queue[1:]
        thermostat_result = result.get("response")
        new_entry = {
            "device": "thermostat",
            "type": "task_completion",
            "action_taken": original_action,
            "result": thermostat_result,
        }
        return Command(
            update={
                "thermostat_result": thermostat_result,
                "task_queue": remaining_tasks,
                "pending_task": None,
                "collaboration_request": {},
                f"{collaborator}_response": None,
                "task_history": task_history + [new_entry],
            },
            goto="task_planner"
        )

    elif task_queue and task_queue[0].get("device") == "thermostat":
        action = task_queue[0].get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Thermostat Agent.

            Your capabilities:
            1. Temperature control: Adjust heating and cooling
            2. Climate optimization: Set comfortable temperature levels
            3. Mode settings: Heat, cool, auto, eco modes

            Current task: {action}
            Task history: {task_history} which you will know what other device already done

            IMPORTANT: Always provide complete solutions with specific settings.
            DO NOT ask users for additional information, infer reasonable values from context.

            First, understand what this task requires.
            Then, decide with {task_history}: Can you complete this task independently with your capabilities and current data?

            If YES: Simulate the temperature control operation and provide the response.
            If NO: Identify what you need help with and which agent can provide it.

            Other agents available for collaboration:
            clock (time, alarms, timers), search_engine (information, weather, recipes), calendar (scheduled events), audio_system (music), tv_display(display/show content), fridge(food related), lighting(light control)

            Examples:

            action: "adjust temperature to 22 degrees"
            Note: Can do independently with temperature control
            {{"response": "Temperature set to 22°C", "collaboration_request": {{}}}}

            action: "create comfortable climate for relaxation"
            Note: Can do independently with climate optimization
            {{"response": "Set temperature to 21°C with gentle airflow for relaxation", "collaboration_request": {{}}}}

            action: "optimize room temperature for energy efficiency"
            Note: Need external energy efficiency knowledge
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "find optimal home temperature settings for energy efficiency"}}}}

            action: "adjust temperature to 22 degrees for an hour"
            Note: Need external energy efficiency knowledge
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "Set a timer for one hour at a temperature of 22 degrees"}}}}

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your result" or "", "collaboration_request": {{"target": "agent_name", "request": "what you need"}} or {{}}}}
            """,
            input_variables=["action","task_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "action": action,
            "task_history": task_history,
        })

        if result.get("collaboration_request") and result["collaboration_request"].get("target"):
            collaboration = result["collaboration_request"]
            new_entry = {
                "device": "thermostat",
                "type": "collaboration_request",
                "action_taken": action,
                "result": {
                    "target": collaboration["target"],
                    "request": collaboration["request"],
                }
            }
            return Command(
                update={
                    "collaboration_request": {
                        "requester": "thermostat",
                        "target": collaboration["target"],
                        "request": collaboration["request"],
                    },
                    "pending_task": {
                        "device": "thermostat",
                        "action": action,
                        "waiting_for": collaboration["target"]
                    },
                    "task_history": task_history + [new_entry],
                },
                goto=f"{collaboration['target']}_agent"
            )
        else:
            remaining_tasks = task_queue[1:]
            thermostat_result = result.get("response")
            new_entry = {
                "device": "thermostat",
                "type": "task_completion",
                "action_taken": action,
                "result": thermostat_result,
            }
            return Command(
                update={
                    "thermostat_result": thermostat_result,
                    "task_queue": remaining_tasks,
                    "task_history": task_history + [new_entry],
                },
                goto="task_planner"
            )

def audio_system_agent(state: SmartHomeState) -> Command:
    task_queue = state.get("task_queue", [])
    collaboration_request = state.get("collaboration_request")
    pending_task = state.get("pending_task")
    task_history = state.get("task_history", [])

    if collaboration_request and collaboration_request.get("target") == "audio_system":
        requester = collaboration_request.get("requester")
        request = collaboration_request.get("request")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Audio System Agent.

            Your capability: Play music and audio content, control volume

            You received a collaboration request from {requester} agent.
            Request: {request}

            Simulate the audio system operation.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your simulated audio system response"}}
            """,
            input_variables=["requester", "request"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "requester": requester,
            "request": request
        })
        audio_system_response = result.get("response")
        new_entry = {
            "device": "audio system",
            "type": "collaboration_response",
            "action_taken": request,
            "result": audio_system_response,
        }

        return Command(
            update={
                "audio_system_response": audio_system_response,
                "collaboration_request": {},
                "task_history": task_history + [new_entry],
            },
            goto=f"{requester}_agent"
        )

    elif pending_task and pending_task.get("device") == "audio_system":
        collaborator = pending_task.get("waiting_for")
        response_key = f"{collaborator}_response"
        collaborator_response = state.get(response_key)
        original_action = pending_task.get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Audio System Agent completing a task with collaboration information.

            Your capability: Play music and audio content, control volume

            Original task: {original_action}
            Task history (what happened before this task):{task_history}
            Collaboration request：{collaboration_request}
            Request from {collaborator}: {collaborator_response}

            Now complete the audio system task using this information. Simulate the operation.

            Don't ask the user for clarification or request help from other agents.
            Don't ask the user for choices or preferences.

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "task completion message"}}
            """,
            input_variables=["original_action","task_history","collaboration_request", "collaborator", "collaborator_response"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "original_action": original_action,
            "task_history": task_history,
            "collaboration_request": collaboration_request,
            "collaborator": collaborator,
            "collaborator_response": collaborator_response
        })
        audio_system_result = result.get("response")
        remaining_tasks = task_queue[1:]
        new_entry = {
            "device": "audio system",
            "type": "task_completion",
            "action_taken": original_action,
            "result": audio_system_result,
        }
        return Command(
            update={
                "audio_system_result": audio_system_result,
                "task_queue": remaining_tasks,
                "pending_task": None,
                "collaboration_request": {},
                f"{collaborator}_response": None,
                "task_history": task_history + [new_entry],
            },
            goto="task_planner"
        )

    elif task_queue and task_queue[0].get("device") == "audio_system":
        action = task_queue[0].get("action")

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="""You are a smart home Audio System Agent.

            Your capability: Play music and audio content, control volume

            Current task: {action}
            Task history: {task_history} which you will know what other device already done

            Important:
            - You can play specific songs, artists, or albums directly
            - You cannot choose music for vague requests - you need recommendations from other agents

            How to handle the task:
            1. Understand what's needed - focus on what's mentioned, don't overthink
            2. Check task_history - has another agent provided relevant information?
            3. Can you complete this **independently** with your capability and task_history info?
            - If yes: do it and provide response without asking user
            - If no: request collaboration from appropriate agent

            Other agents available for collaboration:
            search_engine (music recommendations, playlists), clock (time, alarms, timers), calendar (event-based audio), lighting (light status), thermostat (temperature control), tv_display(display content), fridge(food related)

            Examples:

            action: "play something relaxing at low volume"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "recommend relaxing music"}}}}

            action: "play classical music, not too loud"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "recommend classical music tracks"}}}}

            # Can do independently with volume control capability
            action: "adjust volume to comfortable level"
            {{"response": "Volume adjusted to 50% for comfortable listening", "collaboration_request": {{}}}}

            # No type specified - Get general recommendations
            action: "play something"
            {{"response": "", "collaboration_request": {{"target": "search_engine", "request": "recommend popular music to play"}}}}

            # Exact song/artist name
            action: "play Bohemian Rhapsody"
            {{"response": "Now playing: Bohemian Rhapsody by Queen", "collaboration_request": {{}}}}

            action: "play Taylor Swift for 2 hours"
            Note: Note: Cannot independently do - involves timing, beyond just playing music, need collaboration
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "set 2 hours timer for playing Taylor Swift's songs"}}}}

            action: "play Adele for 1 hour"
            Note: Cannot independently do - involves timing, beyond just playing music, need collaboration
            {{"response": "", "collaboration_request": {{"target": "clock", "request": "set 1 hour timer for playing Adele's songs"}}}}

            {format_instructions}

            Output only JSON.
            Output format: {{"response": "your result" or "", "collaboration_request": {{"target": "agent_name", "request": "what you need"}} or {{}}}}
            """,
            input_variables=["action","task_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        result = chain.invoke({
            "action": action,
            "task_history": task_history,
        })

        if result.get("collaboration_request") and result["collaboration_request"].get("target"):
            collaboration = result["collaboration_request"]
            new_entry = {
                "device": "audio system",
                "type": "collaboration_request",
                "action_taken": action,
                "result": {
                    "target": collaboration["target"],
                    "request": collaboration["request"],
                }
            }
            return Command(
                update={
                    "collaboration_request": {
                        "requester": "audio_system",
                        "target": collaboration["target"],
                        "request": collaboration["request"],
                    },
                    "pending_task": {
                        "device": "audio_system",
                        "action": action,
                        "waiting_for": collaboration["target"]
                    },
                    "task_history": task_history + [new_entry],
                },
                goto=f"{collaboration['target']}_agent"
            )
        else:
            remaining_tasks = task_queue[1:]
            audio_system_result = result.get("response")
            new_entry = {
                "device": "audio system",
                "type": "task_completion",
                "action_taken": action,
                "result": audio_system_result,
            }
            return Command(
                update={
                    "audio_system_result": audio_system_result,
                    "task_queue": remaining_tasks,
                    "task_history": task_history + [new_entry],
                },
                goto="task_planner"
            )


builder = StateGraph(SmartHomeState)

# add node
builder.add_node("human", human)
builder.add_node("intent_analysis", intent_analysis)
builder.add_node("task_planner", task_planner)
builder.add_node("clock_agent", clock_agent)
builder.add_node("calendar_agent", calendar_agent)
builder.add_node("search_engine_agent", search_engine_agent)
builder.add_node("tv_display_agent", tv_display_agent)
builder.add_node("fridge_agent", fridge_agent)
builder.add_node("lighting_agent", lighting_agent)
builder.add_node("thermostat_agent", thermostat_agent)
builder.add_node("audio_system_agent", audio_system_agent)

# set entry and END
builder.add_edge(START, "human")
builder.add_edge("task_planner", END)

graph = builder.compile(checkpointer=MemorySaver())

