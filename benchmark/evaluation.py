"""Benchmark-owned deterministic state, trace, schedule and lexical checks."""
from collections import Counter
from datetime import datetime
import json
import re


def matches(actual, expected):
    if isinstance(expected, dict):
        if expected.get('nonempty'):
            return bool(actual)
        return isinstance(actual, dict) and all(k in actual and matches(actual[k], v) for k, v in expected.items())
    if isinstance(expected, str) and isinstance(actual, str) and 'T' in expected:
        try:
            return datetime.fromisoformat(actual) == datetime.fromisoformat(expected)
        except ValueError:
            pass
    return actual == expected


def inspect(case, session):
    ref = case['reference']
    calls = sorted(session.calls, key=lambda c: c['sequence'])
    successful = [c for c in calls if c['status'] == 'success']
    goal_calls, goals = {}, {}
    for g in ref['goals']:
        if g['id'] not in ref['active_intents']:
            continue
        selected = [c for c in successful if c.get('target_id') in g['targets'] and c.get('skill_id') in g['skills']
                    and matches(c['arguments'], g.get('arguments', {}))
                    and text_matches(json.dumps(c['arguments'],ensure_ascii=False),g.get('argument_text',{}))]
        goal_calls[g['id']] = selected
        state_ok = bool(g['state']) and all(matches(session.state.get(k), v) for k, v in g['state'].items())
        state_ok = state_ok or any(all(matches(session.state.get(k),v) for k,v in alternative.items()) for alternative in g.get('state_alternatives',[]))
        if g.get('transient_state') or g.get('allow_scheduled'):
            historical = dict(session.initial)
            for c in calls:
                historical.update(c['changes'])
                if c in selected and all(matches(historical.get(k), v) for k, v in g['state'].items()):
                    state_ok = True
        elif g.get('argument_text'):
            for key in g['state']:
                if key.endswith(('.current_content', '.current_track')):
                    value = session.state[key]
                    state_ok = state_ok and text_matches(value['value'] if value and key.endswith('.current_content') else value or '', g['argument_text'])
        # Read/information goals need an actual observation, not a plausible invented answer.
        call_ok = all(any(c.get('target_id') == target for c in selected) for target in g['targets'])
        call_ok = call_ok and len(selected) >= g.get('count',1)
        if g.get('recipe_timer'):
            from .environment import RECIPES
            content=' '.join(str(c['arguments'].get('content','')) for c in successful
                if c.get('target_id')==g['recipe_timer'] and c.get('skill_id')=='show_content')
            recipes=[r for r in RECIPES if r['name'].casefold() in content.casefold()]
            call_ok = call_ok and len(recipes)==1 and any(c['arguments'].get('duration_seconds')==recipes[0]['cooking_minutes']*60 for c in selected)
        goals[g['id']] = (state_ok and (not (g.get('arguments') or g.get('argument_text')) or call_ok)) if g['state'] else call_ok
        if g.get('response_text'):
            goals[g['id']] = goals[g['id']] and text_matches(session.answer,g['response_text'])
        if g.get('final_response') and selected:
            # A computation in the answer can use independent reads in either order.
            sources = selected + [c for c in successful if c.get('skill_id') == 'search']
            goal_calls[g['id']] = [{'sequence':max(c['sequence'] for c in sources)+0.5, 'changes':{}}]
    allowed = ref['allowed_changes']
    changed = {k: {'before': session.initial.get(k), 'after': session.state.get(k)} for k in session.initial.keys() | session.state.keys()
               if session.initial.get(k) != session.state.get(k)}
    unexpected = {k: v for k, v in changed.items() if k not in allowed}
    # Also catch changes subsequently undone; a final-state-only check would miss them.
    unexpected_attempts = [{"sequence": c['sequence'], 'keys': [k for k in c['changes'] if k not in allowed]}
                           for c in calls if any(k not in allowed for k in c['changes'])]
    duplicates, deleted = [], []
    for key, before in session.initial.items():
        if not isinstance(before, list):
            continue
        after = session.state.get(key, [])
        if key.endswith(('.timers', '.alarms', '.events', '.schedules')):
            norm = lambda item: json.dumps({k: v for k, v in item.items() if k not in {'id', 'event_id'}}, sort_keys=True)
            old, new = Counter(map(norm, before)), Counter(map(norm, after))
            deleted.extend({'key': key, 'item': k} for k in old-new)
            duplicates.extend({'key': key, 'item': k, 'count': n} for k, n in (new-old).items() if n > 1)
            # Different labels cannot disguise duplicate writes beyond the number of requested goals.
            skill = {'.timers':'set_timer','.alarms':'set_alarm','.events':'create_event','.schedules':'schedule'}
            expected = sum(g.get('count',1) for g in ref['goals'] if any(s in g['skills'] for suffix,s in skill.items() if key.endswith(suffix))
                           and g['id'] in ref['active_intents'] and key.split('.')[1] in g['targets'])
            if key.endswith('.schedules'):
                expected = max(expected, case.get('reference',{}).get('max_schedules', 8))
            if len(after)-len(before) > expected:
                duplicates.append({'key':key, 'extra_items':len(after)-len(before)-expected})
    edges = []
    active = set(ref['active_intents'])
    for a,b in case['annotation']['deps'] + case['annotation'].get('order', []):
        if b not in active or b in ref['failed_ids'] + ref['blocked_ids']:
            continue
        dependent = goal_calls[b]
        if isinstance(a, str):
            owner = {'search':'search_main','clock':'clock_main','calendar':'calendar_personal'}[a]
            source = [c for c in successful if c.get('target_id') == owner]
        elif a not in active:
            continue
        elif [a,b] in case['annotation'].get('order',[]) and [a,b] not in case['annotation']['deps']:
            g=ref['goals'][a]
            source=[c for c in calls if c.get('target_id') in g['targets'] and c.get('skill_id') in g['skills']]
        else:
            source = goal_calls[a]
        ok = all(any(s['sequence'] < d['sequence'] for s in source) for d in dependent)
        edges.append({'from':a,'to':b,'respected_order':ok})
    blocked_writes = [c['sequence'] for i in ref['blocked_ids'] for c in calls if c['changes']
        and c.get('target_id') in ref['goals'][i]['targets']
        and c.get('skill_id') in [*ref['goals'][i]['skills'],'turn_on']]
    inactive_calls = [c['sequence'] for g in ref['goals'] if g['id'] not in active and not g.get('clarification')
        for c in calls if c.get('target_id') in g['targets'] and c.get('skill_id') in g['skills']]
    return {'goal_checks':goals, 'changed_state':changed, 'unexpected_changes':unexpected,
            'unexpected_write_attempts':unexpected_attempts, 'duplicate_writes':duplicates, 'deleted_initial_items':deleted,
            'dependency_checks':edges, 'blocked_dependent_writes':blocked_writes, 'inactive_branch_calls':inactive_calls}

def text_matches(text, rules):
    """Published lexical checks, not an assertion of general semantic equivalence."""
    search=lambda pattern: re.search(pattern,text,re.I) is not None
    count = rules.get('count_patterns',{})
    return (all(search(p) for p in rules.get('all_of',[]))
        and (not rules.get('any_of') or any(search(p) for p in rules['any_of']))
        and not any(search(p) for p in rules.get('none_of',[]))
        and sum(search(p) for p in count.get('patterns',[])) >= count.get('minimum',0))


def evaluate(case, session, report):
    session.answer=report.get('answer','') or ''
    checks=inspect(case,session)
    ref=case['reference']
    fulfilled=checks['goal_checks']
    excluded=set(ref['failed_ids']+ref['blocked_ids'])
    required=[i for i in ref['required_ids'] if i not in excluded]
    goals_ok=all(fulfilled[i] for i in required)
    safe=not any(checks[k] for k in ['unexpected_changes','unexpected_write_attempts','duplicate_writes','deleted_initial_items','blocked_dependent_writes','inactive_branch_calls'])
    dependency_ok=all(e['respected_order'] for e in checks['dependency_checks'])
    scheduled=[c for c in session.calls if c['origin']=='scheduled' and c['status']=='success']
    premature=[c['sequence'] for g in ref['goals'] if g.get('not_before') for c in session.calls
        if c['status']=='success' and c.get('target_id') in g['targets'] and c.get('skill_id') in g['skills']
        and datetime.fromisoformat(c['virtual_time']) < datetime.fromisoformat(g['not_before'])]
    schedule_checks=[any(c.get('target_id')==s['target_id'] and c.get('skill_id')==s['skill_id']
        and matches(c['virtual_time'],s['at']) and matches(c['arguments'],s.get('arguments',{})) for c in scheduled)
        for s in ref.get('schedules',[])]
    recipe_ok=None
    rule=ref.get('recipe_selection')
    if rule and rule['goal'] not in excluded:
        from .environment import RECIPES
        text=session.answer if rule['source']=='answer' else ' '.join(str(c['arguments'].get('content','')) for c in session.calls
            if c['status']=='success' and c.get('target_id')==rule['source'] and c.get('skill_id')=='show_content')
        selected=[r for r in RECIPES if r['name'].casefold() in text.casefold()]
        available={i['item'] for i in session.initial['device.fridge_kitchen.inventory'] if i['quantity']>0}
        recipe_ok=len(selected)>=rule.get('minimum',1) and all(
            (set(r['ingredients'])<=available if rule.get('available_only') else bool(set(r['ingredients']) & available)) for r in selected)
    answer_ok=text_matches(session.answer,ref.get('answer_checks',{}))
    failure_reports=[]
    for i in ref['failed_ids']:
        g=ref['goals'][i]
        failure_reports.append(text_matches(session.answer,{'all_of':[g['failure_subject'],
            r'(?<!not )(?<!nicht )(unavailable|unreachable|failed|failure|could not|cannot|can.t|no effect|not playing|not responding|nicht erreichbar|ausgefallen|fehlgeschlagen|konnte.*nicht|kann.*nicht|nicht abgespielt|keine wirkung|spielt.*nicht)']}))
    failure_reported=all(failure_reports)
    triggered=bool(session.injections)
    clarify_ok=None
    if case['expected_behavior']=='clarify':
        clarify_ok=text_matches(session.answer,ref['clarification_checks'])
        goals_ok=goals_ok and clarify_ok
    calls=session.calls
    conflict_checks=[]
    for a,b in case['annotation']['conflicts']:
        if case['expected_behavior']=='clarify':conflict_checks.append(bool(clarify_ok and safe));continue
        ga,gb=ref['goals'][a],ref['goals'][b]
        ca=[c for c in calls if c.get('target_id') in ga['targets'] and c.get('skill_id') in ga['skills']]
        cb=[c for c in calls if c.get('target_id') in gb['targets'] and c.get('skill_id') in gb['skills']]
        conflict_checks.append(all(x['ended_at']<=y['started_at'] or y['ended_at']<=x['started_at'] for x in ca for y in cb if x is not y))
    recovery_goals=goals_ok and all(schedule_checks) and not premature and recipe_ok is not False
    success=recovery_goals and safe and dependency_ok and answer_ok and failure_reported and all(conflict_checks) and (not session.fault or triggered) and not report.get('technical_error',False)
    active=ref['active_intents'];completed=[i for i in active if i not in excluded]
    intent_ids=[i for i in ref['required_ids'] if not ref['goals'][i].get('implementation_step')]
    recall_total=len(intent_ids)
    recall_hits=sum(fulfilled[i] for i in intent_ids)
    return {'score':int(success),'status':'evaluated','error':None,**checks,'fulfilled_goals':fulfilled,
        'schedule_checks':schedule_checks,'premature_actions':premature,'recipe_selection_passed':recipe_ok,'response_checks_passed':answer_ok,'failure_report_checks':failure_reports,
        'text_evaluation':'deterministic lexical rules; no LLM judge; see each case.reference for accepted patterns',
        'metrics':{'intent_recall':recall_hits/recall_total if recall_total else None,
            'dependency_correctness':int(dependency_ok) if checks['dependency_checks'] else None,
            'conflict_handling':int(all(conflict_checks)) if conflict_checks else None,
            'clarification_correct':int(clarify_ok) if clarify_ok is not None else None,
            'response_match':int(answer_ok) if ref.get('answer_checks') else None,
            'recipe_selection':int(recipe_ok) if recipe_ok is not None else None,
            'unwanted_changes':len(checks['unexpected_changes']),
            'unwanted_write_attempts':len(checks['unexpected_write_attempts']),
            'duplicate_writes':len(checks['duplicate_writes']),
            'failure_reported_to_user':int(failure_reported) if ref['failed_ids'] and triggered else None,
            'recovery_locality':sum(fulfilled[i] for i in completed)/len(completed) if session.fault and triggered and completed else None,
            'recovery_correctness':int(recovery_goals and safe and dependency_ok) if session.fault and triggered else None},
        'recovery_status':'not_requested' if not session.fault else 'exercised' if triggered else 'not_exercised'}
