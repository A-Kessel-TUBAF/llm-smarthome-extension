"""
Evaluation script with category selection and detailed log saving
"""
import json
import uuid
import time
import sys
import os
from smart_home_langgraph import graph
from langgraph.types import Command

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Load benchmark
with open('../benchmark/benchmark_data.json', 'r', encoding='utf-8') as f:
    benchmark = json.load(f)

# Check command line argument for category filter
category_filter = sys.argv[1] if len(sys.argv) > 1 else 'all'

# Extract test cases based on category
if category_filter == 'all':
    test_cases = benchmark['test_cases']
    print(f"\nRunning ALL test cases ({len(test_cases)} total)")
else:
    test_cases = [tc for tc in benchmark['test_cases'] if tc['category'] == category_filter]
    print(f"\nRunning {category_filter.upper()} test cases ({len(test_cases)} total)")

print("=" * 70)

for i, test_case in enumerate(test_cases, 1):
    user_input = test_case['user_input']
    test_id = test_case['id']

    print("\n" + "=" * 70)
    print(f"Case {i}/{len(test_cases)}: [{test_id}] {user_input}")
    print("=" * 70)

    # Prepare log content
    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"Test Case ID: {test_id}")
    log_lines.append(f"Category: {test_case['category']}")
    log_lines.append(f"User Input: {user_input}")
    log_lines.append("=" * 70)
    log_lines.append("")

    start_time = time.time()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "messages": [],
        "task_queue": [],
        "collaboration_request": {},
        "task_history": [],
    }

    # Run until interrupt
    for event in graph.stream(initial_state, config):
        pass

    # Get user input and process
    final_state = None
    for event in graph.stream(Command(resume=user_input), config):
        node_name = list(event.keys())[0]
        state = event[node_name]
        final_state = state

        if node_name == "human":
            continue

        # Print to terminal
        print(f"Processing Node: {node_name}")

        # Add to log
        log_lines.append(f"Node: {node_name}")
        log_lines.append("-" * 70)

        # Collaboration request
        if state.get('collaboration_request') and state['collaboration_request'].get('target'):
            collab = state['collaboration_request']
            log_lines.append(f"COLLABORATION REQUEST:")
            log_lines.append(f"   From: {collab.get('requester')}")
            log_lines.append(f"   To: {collab.get('target')}")
            log_lines.append(f"   Request: {collab.get('request')}")
            log_lines.append("")

        # Pending task
        if state.get('pending_task'):
            pending = state['pending_task']
            log_lines.append(f"PENDING TASK:")
            log_lines.append(f"   Device: {pending.get('device')}")
            log_lines.append(f"   Action: {pending.get('action')}")
            log_lines.append(f"   Waiting for: {pending.get('waiting_for')}")
            log_lines.append("")

        # Log task queue
        if node_name == "task_planner":
            if state.get('task_queue'):
                log_lines.append(f"Task Queue: {json.dumps(state['task_queue'], indent=2)}")
                log_lines.append("")

        # Response & Result
        response_keys = {
            'clock_response': 'Clock',
            'calendar_response': 'Calendar',
            'search_engine_response': 'Search Engine',
            'tv_display_response': 'TV Display',
            'fridge_response': 'Fridge',
            'lighting_response': 'Lighting',
            'thermostat_response': 'Thermostat',
            'audio_system_response': 'Audio System'
        }

        for key, name in response_keys.items():
            if state.get(key):
                log_lines.append(f"COLLABORATION RESPONSE from {name}:")
                log_lines.append(f"   {state[key]}")
                log_lines.append("")

        # Log agent final results
        result_keys = {
            'clock_result': 'Clock',
            'calendar_result': 'Calendar',
            'search_engine_result': 'Search Engine',
            'tv_display_result': 'TV Display',
            'fridge_result': 'Fridge',
            'lighting_result': 'Lighting',
            'thermostat_result': 'Thermostat',
            'audio_system_result': 'Audio System'
        }

        for key, name in result_keys.items():
            if state.get(key):
                log_lines.append(f"{name} RESULT: {state[key]}")
                log_lines.append("")

        log_lines.append("")

    end_time = time.time()
    execution_time = end_time - start_time

    # Add execution time
    log_lines.append("=" * 70)
    log_lines.append(f"Execution Time: {execution_time:.2f}s")
    log_lines.append("=" * 70)
    log_lines.append("")


    # Print time to terminal
    print(f"\nTime: {execution_time:.2f}s")

    # Save log to file
    log_filename = f"logs/{test_id}.txt"
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write('\n'.join(log_lines))

    print(f"Log saved to: {log_filename}")

print("\n" + "=" * 70)
print(f"All test cases completed!")
print(f"Logs saved in: logs/ directory")
print("=" * 70)