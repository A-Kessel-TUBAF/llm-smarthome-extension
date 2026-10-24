"""Reviewed complexity of the extended edition; independent of participant plans."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

FEATURES = ['intents', 'devices', 'graph_depth', 'graph_width', 'ambiguity',
            'temporal_constraints', 'spatial_constraints', 'conflicts']


def levels(nodes, edges):
    predecessors = defaultdict(set)
    for a, b in edges:
        predecessors[b].add(a)
    result = {}
    def visit(node, path):
        if node in path:
            raise ValueError('Cyclic reference dependency graph.')
        if node not in result:
            result[node] = 1 + max((visit(p, path | {node}) for p in predecessors[node]), default=0)
        return result[node]
    for node in nodes:
        visit(node, set())
    return result


def complexity(case, ann=None):
    ann = ann or case['annotation']
    ambiguity = ann['ambiguity']
    if ambiguity not in (0, 1, 2):
        raise ValueError(f"{case['id']}: ambiguity must be 0, 1 or 2.")
    goals = case['reference']['goals']
    active = set(ann['active_intents'])
    edges = [tuple(e) for e in ann['deps'] + ann.get('order', [])
             if e[1] in active and (isinstance(e[0], str) or e[0] in active)]
    helpers = {a for a, _ in edges if isinstance(a, str)}
    graph = levels(active | helpers, edges)
    devices = {target for g in goals if g['id'] in active for target in g['targets']}
    if case['reference'].get('schedules'):
        devices.add('clock_main')
    devices.update({'search': 'search_main', 'clock': 'clock_main', 'calendar': 'calendar_personal'}[h] for h in helpers)
    intents = sum(g['id'] in active and not g.get('implementation_step') for g in goals)
    return dict(zip(FEATURES, [intents, len(devices),
        max(graph.values(), default=1), max(Counter(graph.values()).values(), default=1), ambiguity,
        len(ann['temporal']), len(ann['spatial']), len(ann['conflicts'])]))


def load():
    data = json.loads(Path(__file__).with_name('extension_tasks.json').read_text())
    annotations = json.loads(Path(__file__).with_name('complexity_annotations.json').read_text())['cases']
    for case in data['test_cases']: case['annotation']=annotations[case['id']]
    return [(c, c.get('dimension', c['category']), complexity(c)) for c in data['test_cases']]


if __name__ == '__main__':
    rows = load()
    if '--json' in sys.argv:
        print(json.dumps({c['id']: v for c, _, v in rows}, indent=2))
    else:
        print('case,' + ','.join(FEATURES))
        for case, _, vector in rows:
            print(case['id'] + ',' + ','.join(str(vector[f]) for f in FEATURES))
