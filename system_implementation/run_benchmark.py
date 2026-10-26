"""Extended DEXA: neutral episodes, fresh environments and automatic evaluation.
The untouched 60-case reference can still be run with --dataset original.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from configuration import activate_config, config_path as configured_path, load_config
from external_benchmarks.common import (Episode, case_file, create_system, prepare_output,
    summarize_calls, system_report, write_json)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from benchmark.complexity import complexity
from benchmark.environment import Session
from benchmark.evaluation import evaluate

DIMENSIONS = ('dependency_structure','ambiguity','scale','concurrency','failure_recovery')
FAILURES = ('none','transient','unavailable','no_effect','state_changed')


def load_cases(dataset='extended', original_path=None, *, suite='all', failure_mode=None, dimension=None):
    if dataset not in {'extended','extension','all','original'}:
        raise ValueError('Unknown DEXA dataset.')
    path = Path(original_path or ROOT.parent/'benchmark'/('benchmark_data.json' if dataset=='original' else 'extension_tasks.json'))
    cases = json.loads(path.read_text(encoding='utf-8'))['test_cases']
    if dataset=='original':
        if suite!='all' or failure_mode or dimension:
            raise ValueError('Suite, failure and dimension filters belong to the extended edition.')
        return cases, [path]
    annotation_path=path.with_name('complexity_annotations.json')
    annotations=json.loads(annotation_path.read_text(encoding='utf-8'))['cases']
    for c in cases:
        if 'reference' not in c:
            raise ValueError('Extended DEXA needs reference predicates; use --dataset original for historical inputs.')
        c['annotation']=annotations[c['id']]
    cases = [c for c in cases if (suite=='all' or c['suite']==suite)
        and (not failure_mode or c.get('injected_failure',{}).get('mode','none')==failure_mode)
        and (not dimension or c.get('dimension')==dimension)]
    for c in cases:
        c['complexity'] = complexity(c)
    return cases, [path, annotation_path]


def usage(report):
    known = report.get('telemetry_available', 'llm_calls' in report)
    fields = ('llm_calls','llm_attempts','llm_retries','llm_call_errors','input_tokens','output_tokens','calls_with_usage')
    if not known:
        return {'available':False, **dict.fromkeys(fields)}
    calls = report.get('llm_calls') or []
    result = summarize_calls(calls)
    for token_field in ('input_tokens','output_tokens'):
        if any(c.get(token_field) is None for c in calls): result[token_field]=None
    return {'available':True, **{k:result[k] for k in fields},
            'tokens_complete':result['calls_with_usage']==len(calls)}


def summarize(rows):
    evaluated = [r for r in rows if r['evaluation']['score'] in (0,1)]
    successes = sum(r['evaluation']['score']==1 for r in evaluated)
    result = {'cases':len(rows),'evaluated_cases':len(evaluated),'evaluation_errors':len(rows)-len(evaluated),
        'successes':successes,'success_rate':successes/len(rows) if rows and len(evaluated)==len(rows) else None,
        'evaluation_coverage':len(evaluated)/len(rows) if rows else 0,
        'errors':sum(bool(r['report'].get('error')) for r in rows),
        'runtime_errors':sum(bool(r['report'].get('technical_error')) for r in rows),
        'mean_latency_s':sum(r['latency_s'] for r in rows)/len(rows) if rows else 0,
        'tool_calls':sum(r['metrics']['tool_calls'] for r in rows),
        'tool_errors':sum(r['metrics']['tool_errors'] for r in rows),
        'injections_triggered':sum(r['injection_status']=='triggered' for r in rows),
        'faults_not_exercised':sum(r['injection_status']=='not_triggered' for r in rows),
        'telemetry_cases':sum(r['model_usage']['available'] for r in rows)}
    for field in ('llm_calls','llm_attempts','llm_retries','llm_call_errors','input_tokens','output_tokens'):
        values = [r['model_usage'][field] for r in rows if r['model_usage'].get(field) is not None]
        result[field] = sum(values) if len(values)==len(rows) else None
    metric_names = {k for r in evaluated for k in r['evaluation']['metrics']}
    for metric in sorted(metric_names):
        values = [r['evaluation']['metrics'][metric] for r in evaluated if r['evaluation']['metrics'][metric] is not None]
        result['mean_'+metric] = sum(values)/len(values) if values else None
    return result


def run_original(cases, output, system, settings):
    """Keep the historical direct run_request path; no new episode interface or scoring."""
    if system is None or system=='mas':
        import smart_home_langgraph as system
    elif isinstance(system,(str,Path)):
        path = Path(system)
        spec = importlib.util.spec_from_file_location('_legacy_dexa_system',path)
        system = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system)
    if not hasattr(system,'run_request'):
        raise ValueError('The original benchmark uses run_request; choose a MAS module or the original wrapper.')
    rows = []
    for c in cases:
        events,error = [],None
        started=time.perf_counter()
        try:
            events=list(system.run_request(c['user_input'],**settings))
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        row={'id':c['id'],'input':c['user_input'],'events':events,'error':error,'latency_s':time.perf_counter()-started}
        write_json(case_file(output,c['id']),row)
        rows.append(row)
    summary={'overall':{'cases':len(rows),'errors':sum(bool(r['error']) for r in rows)},'complete':True,
             'note':'Unmodified historical reference: direct system logs, no extended scoring.'}
    write_json(output/'summary.json',summary)
    return summary


def run_benchmark(category='all', benchmark_path=None, *, system=None, config_path=None,
                  limit=0, case_ids=(), output_dir=None, dataset='extended', dimension=None,
                  suite='all', failure_mode=None, resume=False, **settings):
    if limit<0 or suite not in {'all','core','robustness'} or failure_mode not in (None,*FAILURES):
        raise ValueError('Invalid limit/suite/failure selection.')
    activate_config(config_path)
    if benchmark_path is None and dataset!='original':
        benchmark_path=configured_path(load_config(config_path)['benchmarks']['dexa']['path'])
    cases,paths=load_cases(dataset,benchmark_path,suite=suite,failure_mode=failure_mode,dimension=dimension)
    unknown=set(case_ids)-{c['id'] for c in cases}
    if unknown:
        raise ValueError('Unknown/filtered case IDs: '+', '.join(sorted(unknown)))
    cases=[c for c in cases if (category=='all' or c['category']==category) and (not case_ids or c['id'] in case_ids)]
    if limit: cases=cases[:limit]
    if not cases: raise ValueError('No DEXA cases match the selection.')
    output=Path(output_dir or ROOT/'logs/dexa')
    if dataset=='original':
        output.mkdir(parents=True,exist_ok=True)
        return run_original(cases,output,system,settings)
    participant=create_system(system,settings,config_path)
    sources=paths+[ROOT.parent/'benchmark'/f for f in ('__init__.py','environment.py','evaluation.py','complexity.py')]
    metadata={'benchmark':'dexa_extended','case_ids':[c['id'] for c in cases],'suite':suite,'failure_mode':failure_mode,
        'settings':settings, 'configuration':load_config(config_path),
        'participant':participant.metadata() if hasattr(participant,'metadata') else {'name':getattr(participant,'name',type(participant).__name__)},
        'sources':{str(p.relative_to(ROOT.parent)) if p.is_relative_to(ROOT.parent) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    output=prepare_output(output,metadata,resume)
    rows=[]
    for case in cases:
        path=case_file(output,case['id'])
        previous=json.loads(path.read_text()) if resume and path.exists() else None
        if previous and previous['evaluation']['score'] in (0,1):
            rows.append(previous)
            continue
        session=Session(case)
        if previous:
            # Retry only the failed evaluator, never repeat the participant's actions.
            session.state=previous['final_state'];session.initial=previous['initial_state']
            session.calls=previous['tool_calls'];session.injections=previous['injections']
            report,elapsed=previous['report'],previous['latency_s']
        else:
            episode=Episode('dexa',case['user_input'],session.public_context(),deepcopy(session.tools),session.call,session)
            started=time.perf_counter()
            report=system_report(participant,episode)
            elapsed=time.perf_counter()-started
            session.finish_schedules()
        try:
            evaluation=evaluate(case,session,report)
        except Exception as exc:
            evaluation={'score':None,'status':'evaluation_error','error':f'{type(exc).__name__}: {exc}'}
        calls=[c for c in session.calls if c['origin']=='system']
        injected=session.injections[0]['sequence'] if session.injections else None
        row={'id':case['id'],'input':case['user_input'],'category':case['category'],'dimension':case.get('dimension','original'),
            'suite':case['suite'],'pair_id':case.get('pair_id'),'failure_mode':case.get('injected_failure',{}).get('mode','none'),
            'complexity':case['complexity'],'expected_behavior':case['expected_behavior'],'completed':evaluation['score'] in (0,1),
            'latency_s':elapsed,'report':report,'model_usage':usage(report),'initial_state':session.initial,'final_state':session.state,
            'tool_calls':session.calls,'injections':session.injections,
            'injection_status':'triggered' if injected else 'not_triggered' if session.fault else 'not_requested',
            'metrics':{'tool_calls':len(calls),'tool_errors':sum(c['status']=='error' for c in calls),
                'scheduled_calls':sum(c['origin']=='scheduled' for c in session.calls),
                'tool_calls_after_injection':sum(c['sequence']>injected for c in calls) if injected else None},
            'evaluation':evaluation}
        write_json(path,row);rows.append(row)
        print(f"[{len(rows)}/{len(cases)}] {case['id']}: score={evaluation['score']} ({elapsed:.2f}s)",flush=True)
    summary={'overall':summarize(rows),'complete':all(r['evaluation']['score'] in (0,1) for r in rows)}
    for field in ('suite','dimension','category','failure_mode','pair_id'):
        summary['by_'+field]={str(value):summarize([r for r in rows if r[field]==value]) for value in dict.fromkeys(r[field] for r in rows) if value is not None}
    write_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('category',nargs='?',default='all')
    parser.add_argument('benchmark_path',nargs='?')
    parser.add_argument('--dataset',choices=('original','extended','extension','all'),default='extended')
    parser.add_argument('--system',default='mas')
    parser.add_argument('--suite',choices=('all','core','robustness'),default='all')
    parser.add_argument('--failure-mode',choices=FAILURES)
    parser.add_argument('--dimension',choices=DIMENSIONS)
    parser.add_argument('--case-id',dest='case_ids',action='append',default=[])
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--config',dest='config_path',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--settings-json',default='{}')
    args=vars(parser.parse_args())
    args.update(json.loads(args.pop('settings_json')))
    result=run_benchmark(**args)
    if not result['complete']: parser.exit(3,'Some automatic evaluations are incomplete; see the per-case error.\n')
