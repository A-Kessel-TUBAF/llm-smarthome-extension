"""Build the four classification tasks as JSONL: id, text, context, labels, split, group.

    python -m classification_eval.tasks --out results/dgx/tasks            # T1-T3
    .venv-homeassistant/bin/python -m classification_eval.tasks --tasks t4 --out results/dgx/tasks

Gold labels come only from the benchmarks' own annotations; nothing is predicted here.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # system_implementation/
REPO = ROOT.parent

TASKS = {
    "t1_dexa_device_types": dict(
        multi_label=True, question="Which smart-home device types does this request need?",
        labels={"search": "web search or information lookup", "fridge": "fridge or food inventory",
                "display": "TV or screen that shows content", "audio": "speaker or music playback",
                "clock": "clock, timer or alarm", "thermostat": "thermostat or room temperature",
                "light": "lights or lighting", "calendar": "calendar, schedule or appointments"}),
    "t2_homebench_validity": dict(
        multi_label=False, question="Can this instruction be carried out with the devices of this home?",
        labels={"normal": "the device exists and supports the requested operation",
                "unexist_device": "the requested device does not exist in this home or room",
                "unexist_attribute": "the device exists but does not support the requested attribute or operation"}),
    "t3_simuhome_feasibility": dict(
        multi_label=False, question="Can this request be fulfilled in this home?",
        labels={"feasible": "the request can be fulfilled with the rooms and devices of this home",
                "infeasible": "the request cannot be fulfilled in this home (missing device, room or capability)"}),
    "t4_homeassistant_domain": dict(
        multi_label=False, question="Which kind of device does this request address?",
        labels={"cover": "blinds, shades, curtains or garage doors", "fan": "fans", "light": "lights",
                "lock": "door locks", "media-player": "TV, speaker or other media player",
                "todo": "to-do or shopping lists", "vacuum": "robot vacuum cleaners", "valve": "water or gas valves"}),
}


def write_jsonl(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def item(id_, text, labels, split="test", context="", group=""):
    return {"id": id_, "text": text, "context": context, "labels": sorted(labels), "split": split, "group": group}


def t1_dexa_device_types():
    """DEXA: device types of all required and acceptable intents (maximal interpretation).

    An intent annotated with alternatives ("clock or calendar") contributes every alternative.
    """
    items = []
    for name in ("benchmark_data.json", "extension_tasks.json"):
        for case in json.loads((REPO / "benchmark" / name).read_text(encoding="utf-8"))["test_cases"]:
            intents = case.get("required_intents", []) + case.get("acceptable_intents", [])
            labels = {t.strip() for i in intents for t in i["device_type"].split(" or ")}
            items.append(item(case["id"], case["user_input"], labels, group=case.get("dimension") or case["category"]))
    return items


def homebench_context(home):
    """The home's devices and operations, one line per device: room.device: op(param:type), ..."""
    ops = defaultdict(list)
    for m in home["method"]:
        device = m["device_name"] if m["room_name"] == "None" else f"{m['room_name']}.{m['device_name']}"
        params = ", ".join(f"{p['name']}:{p['type']}" for p in m["parameters"])
        ops[device].append(f"{m['operation']}({params})" if params else m["operation"])
    return "\n".join(f"{device}: {', '.join(names)}" for device, names in sorted(ops.items()))


def t2_homebench_validity(path=None):
    """HomeBench single-instruction cases: normal, unexist_device or unexist_attribute."""
    root = Path(path or os.getenv("HOMEBENCH_PATH") or Path.home() / "HomeBench" / "dataset")
    root = root / "dataset" if (root / "dataset").is_dir() else root
    homes = {str(h["home_id"]): homebench_context(h) for h in read_jsonl(root / "home_status_method.jsonl")}
    labels = set(TASKS["t2_homebench_validity"]["labels"])
    items = []
    for split, files in (("train", ("train_data_part1.jsonl", "train_data_part2.jsonl")),
                         ("valid", ("valid_data.jsonl",)), ("test", ("test_data.jsonl",))):
        for name in files:
            for case in read_jsonl(root / name):
                if case["type"] in labels:
                    items.append(item(case["id"], case["input"], {case["type"]}, split,
                                      homes[str(case["home_id"])], group=f"home{case['home_id']}"))
    return items


def simuhome_context(episode):
    """User location, then per room its devices with their Matter clusters."""
    lines = [f"user location: {episode['user_location']}"]
    for room, content in episode["initial_home_config"]["rooms"].items():
        devices = []
        for device in content.get("devices", []):
            clusters = sorted({key.split(".")[1] for key in device.get("attributes", {})
                               if key.count(".") >= 2 and key.split(".")[1] != "BasicInformation"})
            devices.append(f"{device['device_type']} ({', '.join(clusters)})" if clusters else device["device_type"])
        lines.append(f"{room}: {'; '.join(devices) or 'no devices'}")
    return "\n".join(lines)


def t3_simuhome_feasibility(path=None):
    """SimuHome episodes: feasible or infeasible; group = query type for stratification."""
    root = Path(path or os.getenv("SIMUHOME_PATH") or Path.home() / "SimuHome")
    directory = root / "data" / "benchmark" if (root / "data" / "benchmark").is_dir() else root
    items = []
    for file in sorted(directory.glob("*.json")):
        episode = json.loads(file.read_text(encoding="utf-8"))
        items.append(item(file.stem, episode["query"], {episode["meta"]["case"]},
                          context=simuhome_context(episode), group=episode["meta"]["query_type"]))
    return items


def t4_homeassistant_domain(path=None):
    """Home Assistant Assist cases with their (single) device category; needs the HA environment."""
    from external_benchmarks.adapters.homeassistant import load
    _, _, cases = load(path)
    return [item(case["id"], case["task"].input_text, set(case["task"].categories),
                 group=case["task"].categories[0]) for case in cases]


BUILDERS = {"t1": ("t1_dexa_device_types", t1_dexa_device_types),
            "t2": ("t2_homebench_validity", t2_homebench_validity),
            "t3": ("t3_simuhome_feasibility", t3_simuhome_feasibility),
            "t4": ("t4_homeassistant_domain", t4_homeassistant_domain)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=BUILDERS, default=["t1", "t2", "t3"])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dgx" / "tasks")
    args = parser.parse_args(argv)
    for key in args.tasks:
        name, build = BUILDERS[key]
        items = build()
        write_jsonl(args.out / f"{name}.jsonl", items)
        splits = defaultdict(int)
        for i in items:
            splits[i["split"]] += 1
        print(f"{name}: {len(items)} items {dict(splits)}")
        if key == "t2":  # Gemma 4 answers a stratified sample: 500 test items per class, seed 42
            by = defaultdict(list)
            for i in items:
                if i["split"] == "test":
                    by[i["labels"][0]].append(i["id"])
            rng = random.Random(42)
            sample = [i for label in sorted(by) for i in rng.sample(sorted(by[label]), 500)]
            (args.out.parent / "t2_sample_ids.txt").write_text("\n".join(sample) + "\n", encoding="utf-8")
            print(f"t2 sample: {len(sample)} ids -> {args.out.parent / 't2_sample_ids.txt'}")


if __name__ == "__main__":
    main()
