"""Bridge saved V2 tool responses to the existing source-material organizer."""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from evidence_preparation import (
    EvidencePreparation, EvidenceGroup, PreparedSourceRecord,
    _inventory, _scalar_counts, resolve_json_pointer, selection_input,
)
from evidence_verification import verify_evidence_result
from run_nim_evidence import run_prepared_group
from automatic_followup import followup_catalog, execute_followups, matching_records, validate_organization_run
from execution_coverage import coverage_index, reconcile_gap
from human_entity_context import HumanConfirmedEntityContext, HumanResearchClarification


def _native_rows(call):
    """Return native record pointers; never rewrite or summarize source values."""
    raw = call['raw_response']
    source = call['source']
    if isinstance(raw.get('raw_record'), dict):
        return '/raw_record', None, 1
    if raw.get('raw_xml'):
        return '', None, 1
    if source == 'open_targets':
        disease, target = raw.get('disease') or {}, raw.get('target') or {}
        if 'associatedTargets' in disease:
            collection = disease['associatedTargets']
            return '/disease/associatedTargets/rows', collection['rows'], collection['count']
        if 'drugAndClinicalCandidates' in target:
            collection = target['drugAndClinicalCandidates']
            return '/target/drugAndClinicalCandidates/rows', collection['rows'], collection['count']
        if 'tractability' in target:
            return '/target', None, 1
        if 'baselineExpression' in target:
            data=target['baselineExpression']
            return '/target/baselineExpression/rows',data['rows'],data['count']
        return None  # Resolution and protein mappings remain in the execution audit.
    if source in {'clinicaltrials_gov', 'europe_pmc'}:
        return '/records', raw['records'], raw['total_count']
    if source == 'chembl' and 'mechanisms' in raw:
        return '/mechanisms', raw['mechanisms'], raw['page_meta']['total_count']
    return None  # ChEMBL target mapping is an identity aid, not another drug fact.


def prepare_execution(execution: dict) -> EvidencePreparation:
    if execution.get('schema_version') != '1.0' or not execution.get('manifest'):
        raise ValueError('Expected a saved PlanExecutor report')
    run_id = execution['run_id']
    clarification=execution.get('research_clarification')
    if clarification is not None:
        HumanResearchClarification.model_validate(clarification).for_run(execution['manifest']['original_question'],run_id)
    confirmed_context = execution.get('human_confirmed_context')
    if confirmed_context is not None:
        HumanConfirmedEntityContext.model_validate(confirmed_context).for_run(
            execution['manifest']['original_question'], run_id)
    calls = execution['calls']
    if len({c['call_id'] for c in calls}) != len(calls):
        raise ValueError('Duplicate execution call IDs')
    records, groups, collections, flags = [], [], [], []
    target_scopes = {}
    expression_scopes = {}
    steps = {s['step_id']: s for s in execution['steps']}
    for call in calls:
        if call['status'] != 'success' or 'raw_response' not in call:
            continue
        native = _native_rows(call)
        if native is None:
            continue
        pointer, rows, total = native
        pointers = [pointer] if rows is None else [f'{pointer}/{i}' for i in range(len(rows))]
        truncated = (total is not None and total > len(pointers)) or bool(call['raw_response'].get('truncated'))
        warnings = ['Collection is truncated; unreturned records remain unknown.'] if truncated else []
        if steps[call['step_id']]['status'] != 'success':
            warnings.append('Other calls in this operation failed; this material is partial.')
        collection = dict(datasource_id=call['call_id'], source=call['source'],
                          total_evidence_count=total, fetched_evidence_count=len(pointers),
                          truncated=truncated, collection_warnings=warnings)
        collections.append(collection)
        selected = []
        for p in pointers:
            raw = resolve_json_pointer(call['raw_response'], p)
            if not isinstance(raw, dict):
                raise ValueError(f'Native record must be an object: {call["call_id"]}{p}')
            native_id = (raw.get('id') or raw.get('molecule_chembl_id') or
                         raw.get('protocolSection', {}).get('identificationModule', {}).get('nctId'))
            sid = str(uuid5(NAMESPACE_URL, f'{run_id}/{call["call_id"]}{p}'))
            row = PreparedSourceRecord(
                snapshot_id=sid, source=call['source'], record_id=str(native_id) if native_id else None,
                source_run_id=run_id, datasource_id=call['call_id'], source_url=call['source_url'],
                retrieved_at=call['finished_at'], raw_record=deepcopy(raw),
                execution_origin={'run_id': run_id, 'call_id': call['call_id'], 'json_pointer': p},
            )
            selected.append(row)
            records.append(row)
            if call['source'] == 'open_targets':
                # Only native returned target IDs establish grouping, never a
                # gene symbol, LLM assertion or a ChEMBL complex component.
                target_id = ((raw.get('target') or {}).get('id') if pointer.startswith('/disease/')
                             else (call['raw_response'].get('target') or {}).get('id'))
                if target_id:
                    target_scopes[sid] = target_id
                    if 'baselineExpression' in (call['raw_response'].get('target') or {}):
                        expression_scopes[sid]=(target_id,raw.get('datasourceId'),raw.get('datatypeId'),raw.get('unit'))
        if not selected:
            continue
        variables = call['arguments'].get('variables', {})
        group_id = str(uuid5(NAMESPACE_URL, f'{run_id}/{call["call_id"]}/organization'))
        groups.append(EvidenceGroup(
            group_id=group_id, source=call['source'], datasource_id=call['call_id'],
            disease_id=variables.get('efoId', ''), target_id=variables.get('ensemblId', ''),
            drug_name_key=None, drug_name_variants=[], snapshot_ids=[r.snapshot_id for r in selected],
            context_snapshot_ids=[], record_count=len(selected),
            field_inventory=_inventory(selected, set()), scalar_value_counts=_scalar_counts(selected),
        ))
        for warning in warnings:
            flags.append({'snapshot_id': selected[0].snapshot_id, 'description': warning})
    # A source call is not necessarily a coherent organization packet: association,
    # drug and tractability calls for the SAME target must be visible together.
    original_groups = groups
    groups = []
    grouped = defaultdict(list)
    record_map = {r.snapshot_id: r for r in records}
    for group in original_groups:
        for sid in group.snapshot_ids:
            key = (('expression',*expression_scopes[sid]) if sid in expression_scopes else
                   ('target', target_scopes[sid]) if sid in target_scopes else ('call', group.datasource_id))
            grouped[key].append((record_map[sid], group))
    for key, entries in grouped.items():
        rows = [row for row, _ in entries]
        diseases = {g.disease_id for _, g in entries if g.disease_id}
        groups.append(EvidenceGroup(
            group_id=str(uuid5(NAMESPACE_URL, f'{run_id}/organization/{key}')),
            source=rows[0].source, datasource_id=key[1] if key[0] == 'call' else None,
            disease_id=next(iter(diseases)) if len(diseases) == 1 else '',
            target_id=key[1] if key[0] in {'target','expression'} else '',
            drug_name_key=None, drug_name_variants=[], snapshot_ids=[r.snapshot_id for r in rows],
            context_snapshot_ids=[], record_count=len(rows), field_inventory=_inventory(rows, set()),
            scalar_value_counts=_scalar_counts(rows),
        ))
    # The complete collection ledger is available to every group. In particular,
    # successful empty searches and failed calls are not silently converted to facts.
    ledger = [{k: deepcopy(c[k]) for k in ('call_id', 'step_id', 'source', 'tool', 'arguments',
               'source_url', 'status', 'error') if k in c} for c in calls]
    preparation = EvidencePreparation(
        model_input_contract='references_v1',
        source_run_id=run_id, research_question=execution['manifest']['original_question'],
        query_context={'manifest': execution['manifest'], 'execution_status': execution['status'],
                       'followup_candidates': followup_catalog(execution),
                       'execution_calls': ledger, 'execution_steps': [
                           {k: s[k] for k in ('step_id', 'status', 'error') if k in s}
                           for s in execution['steps']]},
        source_records=records, source_collections=collections, groups=groups, model_inputs=[],
        source_field_claims=[], collection_review_flags=flags, supplemental_requests=[],
        limitations=[
            'Open Targets records sharing a native target ID are grouped together; other sources retain their call scope. Gaps remain group-local candidates, not global retrieval instructions.',
            'The original execution retains identity/mapping calls and every raw response. Native records here retain call ID and JSON Pointer.',
            'Source material and collection status do not establish scientific sufficiency, safety, efficacy or negative findings.',
            'No evidence pruning. Only source-grounded selected requests can run in one supplemental round; other gaps remain unresolved.',
        ],
    )
    if confirmed_context is not None:
        preparation.query_context['human_confirmed_context'] = deepcopy(confirmed_context)
        preparation.query_context['human_confirmed_entity_resolutions'] = [deepcopy(e)
            for step in execution['steps'] if step['status'] == 'success'
            for e in step.get('result', {}).get('entities', [])
            if 'human_scope' in e.get('resolution_evidence', {})]
    preparation.query_context['retrieval_subject_scopes']=[dict(step_id=s['step_id'],source_ref=p['source_ref'],
        subject_scope=p['subject_scope'], **({'search_scope': p['search_scope']} if 'search_scope' in p else {}))
        for s in execution['steps'] if s['status']=='success'
        for p in s.get('result',{}).get('source_material',[]) if 'subject_scope' in p]
    preparation.query_context['executed_filters']=[dict(step_id=s['step_id'],result=s['result'])
        for s in execution['steps'] if s['status']=='success' and s.get('operation')=='filter_targets']
    preparation.query_context['native_target_scopes']=deepcopy(target_scopes)
    from filter_assessment import assessed_filters
    preparation.query_context['assessed_filters']=assessed_filters(execution)
    preparation.query_context['baseline_expression_collections']=[c for s in execution['steps']
        if s['status']=='success' for c in s.get('result',{}).get('expression_collections',[])]
    preparation.query_context['relative_expression_rankings']=[dict(step_id=s['step_id'],result=s['result']['expression_ranking'])
        for s in execution['steps'] if s['status']=='success' and s.get('result',{}).get('expression_ranking') is not None]
    if clarification is not None:
        preparation.query_context['research_clarification']=deepcopy(clarification)
    preparation.query_context['execution_coverage'] = coverage_index(execution, records, groups)
    preparation.model_inputs = [selection_input(preparation, g, preparation.research_question)[0] for g in groups]
    verify_execution_preparation(preparation, execution)
    return preparation


def verify_execution_preparation(preparation, execution):
    """Verify copied values and provenance against the saved execution, not the LLM."""
    if preparation.source_run_id != execution['run_id']:
        raise ValueError('Execution run identity mismatch')
    if 'native_target_scopes' in preparation.query_context:
        calls={c['call_id']:c for c in execution['calls']};expected={}
        for record in preparation.source_records:
            call=calls.get(record.execution_origin['call_id'])
            if call is None:raise ValueError('Unknown source call in native target map')
            if call['source']!='open_targets':continue
            target=((record.raw_record.get('target') or {}).get('id') if record.execution_origin['json_pointer'].startswith('/disease/')
                    else (call['raw_response'].get('target') or {}).get('id'))
            if target:expected[record.snapshot_id]=target
        if preparation.query_context['native_target_scopes']!=expected:
            raise ValueError('Native target scope map differs from source execution')
    if 'executed_filters' in preparation.query_context:
        expected=[dict(step_id=s['step_id'],result=s['result']) for s in execution['steps']
                  if s['status']=='success' and s.get('operation')=='filter_targets']
        if preparation.query_context['executed_filters']!=expected:raise ValueError('Selection ledger differs from execution')
    if 'assessed_filters' in preparation.query_context:
        from filter_assessment import assessed_filters
        if preparation.query_context['assessed_filters']!=assessed_filters(execution):raise ValueError('Filter assessment differs from source evaluation')
    if 'execution_coverage' in preparation.query_context and preparation.query_context['execution_coverage'] != coverage_index(
            execution, preparation.source_records, preparation.groups):
        raise ValueError('Coverage index differs from original native execution scope')
    calls = {c['call_id']: c for c in execution['calls']}
    for record in preparation.source_records:
        origin = record.execution_origin or {}
        call = calls.get(origin.get('call_id'))
        if not call or call['status'] != 'success' or origin.get('run_id') != execution['run_id']:
            raise ValueError('Record has no successful original source call')
        native = resolve_json_pointer(call['raw_response'], origin['json_pointer'])
        if (native != record.raw_record or call['source'] != record.source or call['source_url'] != record.source_url
                or record.source_run_id != execution['run_id'] or record.datasource_id != call['call_id']
                or record.retrieved_at != datetime.fromisoformat(call['finished_at'])):
            raise ValueError('Prepared material differs from original source response or provenance')
    return {'status': 'passed', 'checked_records': len(preparation.source_records)}


def organize_execution(execution, client, output_path, *, execution_path=None):
    """Prepare, organize all material groups once each, verify and save progress."""
    output_path = Path(output_path)
    prepared_path = output_path.with_suffix('.preparation.json')
    output_files = {output_path.resolve(), prepared_path.resolve(),
                    output_path.with_suffix('.followup.execution.json').resolve(),
                    output_path.with_suffix('.followup.preparation.json').resolve()}
    if execution_path and Path(execution_path).resolve() in output_files:
        raise ValueError('Organization outputs must not overwrite the source execution')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preparation = prepare_execution(execution)
    prepared_path.write_text(preparation.model_dump_json(indent=2), encoding='utf-8')
    report = {'source_run_id': execution['run_id'], 'status': 'running',
              'source_execution_file': str(Path(execution_path).resolve()) if execution_path else None,
              'research_question': preparation.research_question,
              'prepared_input_file': str(prepared_path),
              'source_integrity': verify_execution_preparation(preparation, execution),
              'execution_status': execution['status'], 'groups': [],
              'source_collections': preparation.source_collections,
              'execution_calls': preparation.query_context['execution_calls'],
              'execution_coverage': preparation.query_context['execution_coverage'],
              'limitations': preparation.limitations}
    report['target_selection_results']=deepcopy(preparation.query_context.get('executed_filters',[]))
    report['assessed_filters']=deepcopy(preparation.query_context.get('assessed_filters',[]))
    def save():
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    for group in preparation.groups:
        result = run_prepared_group(preparation, group.group_id, client)
        verification = verify_evidence_result(preparation, result)
        report['groups'].append({'group_id': group.group_id, 'result': result, 'verification': verification})
        save()
    all_valid = all(g['result']['status'] == 'references_validated' and g['verification']['status'] != 'failed'
                    for g in report['groups'])
    report['status'] = ('no_material' if not report['groups'] else
                        'references_validated' if all_valid and execution['status'] == 'completed' else 'incomplete')
    if report['groups']:
        report = finish_followup(execution, report, client, output_path)
    save()
    return report


def finish_followup(execution, report, client, output_path):
    """One bounded pass; re-organize only groups receiving new or reused material."""
    validate_organization_run(execution, report)
    original_preparation = prepare_execution(execution)
    coverage = original_preparation.query_context['execution_coverage']
    original_groups = {g.group_id: g for g in original_preparation.groups}
    report['initial_coverage_decisions'] = []
    reuse_requests = []
    non_actionable={}
    for g in report['groups']:
        for i, flag in enumerate((g['result'].get('structured_output') or {}).get('review_flags', [])):
            decision = reconcile_gap(flag, coverage, original_groups[g['group_id']], original_preparation.source_records)
            from filter_assessment import group_assessed_result,already_assessed_filter_input
            if already_assessed_filter_input(group_assessed_result(original_preparation,original_groups[g['group_id']]),decision):
                decision['disposition']='filter_input_already_evaluated'
                non_actionable[g['group_id'],i]='Typed source field is present in the evaluated retrieved collection; zero matches or its truncation alone is not an executable missing-material request.'
            report['initial_coverage_decisions'].append(dict(group_id=g['group_id'], flag_index=i, **decision))
            if (g['group_id'],i) in non_actionable:continue
            ids = sorted({sid for ref in decision['references'] for sid in ref['reuse_snapshot_ids']})
            if ids:
                reuse_requests.append(dict(tool='reuse_coverage_material', status='already_collected',
                    source_run_id=execution['run_id'],
                    result_snapshot_ids=ids, coverage_decision=decision,
                    origin_flags=[dict(group_id=g['group_id'], flag_index=i)]))
    augmented, requests = execute_followups(execution, report,non_actionable_flags=non_actionable)
    # A typed cache reuse replaces the unmapped dispatch for the SAME original flag.
    reused = {(o['group_id'], o['flag_index']) for r in reuse_requests for o in r['origin_flags']}
    requests = [r for r in requests if not (r['status'] == 'unmapped' and
                any((o['group_id'], o['flag_index']) in reused for o in r['origin_flags']))] + reuse_requests
    report['followup'] = {'round_limit': 1, 'requests': requests,
                          'network_calls': len(augmented['calls']) - len(execution['calls'])}
    if not requests or all(r['status']=='not_required' for r in requests):
        report['followup']['status'] = 'not_needed' if report['status'] == 'references_validated' else 'not_determined'
        return report
    report['initial_groups'] = deepcopy(report['groups'])
    report['initial_prepared_input_file'] = report['prepared_input_file']
    report['initial_source_execution_file'] = report['source_execution_file']
    augmented_path = Path(output_path).with_suffix('.followup.execution.json')
    augmented_path.write_text(json.dumps(augmented, ensure_ascii=False, indent=2), encoding='utf-8')
    report['followup']['execution_file'] = str(augmented_path.resolve())
    preparation = prepare_execution(augmented)
    by_group = {g.group_id: g for g in preparation.groups}
    changed = set()
    for request in requests:
        if request['status'] not in {'success', 'already_collected'}:
            continue
        ids = (request['result_snapshot_ids'] if request['tool'] == 'reuse_coverage_material' else
               matching_records(request, preparation.source_records))
        request['result_snapshot_ids'] = ids
        for origin in request['origin_flags']:
            group = by_group.get(origin['group_id'])
            if group is None:
                continue
            additional = set(ids) - set(group.snapshot_ids + group.context_snapshot_ids)
            if additional:
                group.context_snapshot_ids.extend(sorted(additional))
                changed.add(group.group_id)
    preparation.model_inputs = [selection_input(preparation, g, preparation.research_question)[0] for g in preparation.groups]
    final_path = Path(output_path).with_suffix('.followup.preparation.json')
    final_path.write_text(preparation.model_dump_json(indent=2), encoding='utf-8')
    report['followup']['prepared_input_file'] = str(final_path.resolve())
    report['prepared_input_file'] = str(final_path.resolve())
    report['source_execution_file'] = str(augmented_path.resolve())
    report['execution_status'] = augmented['status']
    report['source_collections'] = preparation.source_collections
    report['source_integrity'] = verify_execution_preparation(preparation, augmented)
    report['execution_calls'] = preparation.query_context['execution_calls']
    report['execution_coverage'] = preparation.query_context['execution_coverage']
    for index, old in enumerate(report['groups']):
        if old['group_id'] in changed:
            result = run_prepared_group(preparation, old['group_id'], client)
            report['groups'][index] = {'group_id': old['group_id'], 'result': result,
                                      'verification': verify_evidence_result(preparation, result)}
            Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    report['followup']['reorganized_groups'] = sorted(changed)
    report['followup']['status'] = 'one_round_finished'
    report['followup']['remaining_flags'] = [
        {'group_id': g['group_id'], 'flag': flag} for g in report['groups']
        for flag in (g['result'].get('structured_output') or {}).get('review_flags', [])]
    candidate_status = {c['request_id']: c['status'] for c in preparation.query_context['followup_candidates']}
    for item in report['followup']['remaining_flags']:
        state = candidate_status.get(item['flag'].get('followup_request_id'))
        item['disposition'] = {'ready': 'round_limit_reached', 'attempted': 'already_attempted',
                               'available': 'available_material_requires_review'}.get(state, 'unmapped')
        item['coverage'] = reconcile_gap(item['flag'], preparation.query_context['execution_coverage'],
                                        by_group[item['group_id']], preparation.source_records)
        if item['coverage']['references']:
            item['disposition'] = 'typed_coverage_requires_review'
    if any(r['status'] == 'error' for r in requests) or any(
            g['result']['status'] != 'references_validated' or g['verification']['status'] == 'failed'
            for g in report['groups']):
        report['status'] = 'incomplete'
    # Fetch success is never called scientific gap resolution. Preserve the new flags.
    return report
