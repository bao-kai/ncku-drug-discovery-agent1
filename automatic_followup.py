"""One-round follow-up bindings derived only from native source locators."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from uuid import NAMESPACE_URL, uuid5

from query_clinicaltrials import get_clinical_trial, search_clinical_trials
from query_europepmc import get_publication_by_pmid, get_open_access_full_text, search_publications


TOOLS = {
    'get_clinical_trial': get_clinical_trial,
    'get_publication_by_pmid': get_publication_by_pmid,
    'get_open_access_full_text': get_open_access_full_text,
    'search_clinical_trials': search_clinical_trials,
    'search_publications': search_publications,
}


def signature(tool, args):
    return json.dumps([tool, args], sort_keys=True, ensure_ascii=False)


def validate_organization_run(execution, organization):
    if organization.get('source_run_id') != execution['run_id']:
        raise ValueError('Organization source_run_id must match the execution run')
    if any(g['result'].get('source_run_id') != execution['run_id'] for g in organization['groups']):
        raise ValueError('Organization group result belongs to a different or unscoped run')


def followup_catalog(execution):
    candidates, available = {}, {}
    attempted = {signature(c['tool'], c['arguments']): c['call_id'] for c in execution['calls']}

    def add(tool, args, source, call, pointer, label):
        key = signature(tool, args)
        item = candidates.setdefault(key, dict(
            request_id=str(uuid5(NAMESPACE_URL, key)), tool=tool, arguments=args,
            source=source, material=label, origins=[], source_run_id=execution['run_id']))
        origin = ({'call_id': call['call_id'], 'json_pointer': pointer} if call is not None else
                  {'reference_scope': 'manifest', 'json_pointer': '/original_question'})
        if origin not in item['origins']:
            item['origins'].append(origin)

    def walk(node, pointer, call):
        if isinstance(node, dict):
            for key, value in node.items():
                path = pointer + '/' + key.replace('~', '~0').replace('/', '~1')
                if key in {'nctId', 'clinicalReportId'} and isinstance(value, str) and re.fullmatch(r'NCT\d{8}', value.upper()):
                    add('get_clinical_trial', {'nct_id': value.upper()}, 'clinicaltrials_gov', call, path, 'Full native study record')
                if key in {'pmid', 'literature'}:
                    for i, v in enumerate(value if isinstance(value, list) else [value]):
                        if re.fullmatch(r'[1-9]\d*', str(v)):
                            add('get_publication_by_pmid', {'pmid': str(v)}, 'europe_pmc', call,
                                path + '/' + str(i) if isinstance(value, list) else path, 'Publication metadata and abstract')
                if key in {'pmcid', 'pubMedCentralIds'}:
                    for i, v in enumerate(value if isinstance(value, list) else [value]):
                        if isinstance(v, str) and re.fullmatch(r'PMC[1-9]\d*', v.upper()):
                            add('get_open_access_full_text', {'pmcid': v.upper()}, 'europe_pmc', call,
                                path + '/' + str(i) if isinstance(value, list) else path, 'Available full-text XML')
                walk(value, path, call)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, pointer + '/' + str(i), call)

    for call in execution['calls']:
        if call['status'] != 'success':
            continue
        raw = call.get('raw_response') or {}
        walk(raw, '', call)
        rows = raw.get('records', [])
        if isinstance(raw.get('raw_record'), dict):
            rows = [raw['raw_record']]
        for row_index, row in enumerate(rows):
            if call['source'] == 'clinicaltrials_gov':
                nct = row.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
                if nct:
                    available[signature('get_clinical_trial', {'nct_id': nct})] = call['call_id']
            if call['source'] == 'europe_pmc' and row.get('source') == 'MED' and row.get('id'):
                args = {'pmid': str(row['id'])}
                location = '/raw_record/id' if isinstance(raw.get('raw_record'), dict) else f'/records/{row_index}/id'
                add('get_publication_by_pmid', args, 'europe_pmc', call, location, 'Publication metadata and abstract')
                available[signature('get_publication_by_pmid', args)] = call['call_id']
        if raw.get('raw_xml') and raw.get('record_id'):
            available[signature('get_open_access_full_text', {'pmcid': raw['record_id']})] = call['call_id']
        token_key = 'next_page_token' if call['source'] == 'clinicaltrials_gov' else 'next_cursor_mark'
        arg_key = 'page_token' if call['source'] == 'clinicaltrials_gov' else 'cursor_mark'
        if call['tool'] in {'search_clinical_trials', 'search_publications'} and raw.get('truncated') and raw.get(token_key):
            args = dict(call['arguments'], **{arg_key: raw[token_key]})
            add(call['tool'], args, call['source'], call, '/' + token_key,
                'Next page of the SAME search, at most 20 records; not a broader query')
    # Explicit user-supplied identifiers are also locators; never parse gap prose.
    question = execution['manifest']['original_question']
    for nct in re.findall(r'(?<![A-Za-z0-9_])NCT\d{8}(?!\d)', question, re.I):
        add('get_clinical_trial', {'nct_id': nct.upper()}, 'clinicaltrials_gov', None, '', 'Full native study record')
    for pmcid in re.findall(r'(?<![A-Za-z0-9_])PMC[1-9]\d*(?!\d)', question, re.I):
        add('get_open_access_full_text', {'pmcid': pmcid.upper()}, 'europe_pmc', None, '', 'Available full-text XML')
    for pmid in re.findall(r'(?<![A-Za-z0-9_])PMID\s*:?\s*([1-9]\d*)(?!\d)', question, re.I):
        add('get_publication_by_pmid', {'pmid': pmid}, 'europe_pmc', None, '', 'Publication metadata and abstract')
    for key, item in candidates.items():
        item['status'] = 'available' if key in available else 'attempted' if key in attempted else 'ready'
        item['existing_call_id'] = available.get(key) or attempted.get(key)
    return list(candidates.values())


def validate_followup_selection(selection, candidates):
    by_id = {c['request_id']: c for c in candidates}
    for gap in selection.review_flags:
        rid = getattr(gap, 'followup_request_id', None)
        if rid is None:
            continue
        if rid not in by_id:
            raise ValueError('Unknown source-grounded follow-up request ID')
        request = by_id[rid]
        if gap.subject_ref and request['tool'] == 'get_clinical_trial' and gap.subject_ref != request['arguments']['nct_id']:
            raise ValueError('Follow-up study conflicts with the gap subject')


def matching_records(request, records):
    run_id = request.get('source_run_id')
    if not run_id:
        raise ValueError('Cached material request requires a source run')
    matches = []
    for record in records:
        if record.source_run_id != run_id or (record.execution_origin or {}).get('run_id') != run_id:
            raise ValueError('Cached source record belongs to a different run')
        if record.execution_origin['call_id'] != request.get('result_call_id'):
            continue
        raw, args = record.raw_record, request['arguments']
        if request['tool'] == 'get_clinical_trial' and raw.get('protocolSection', {}).get('identificationModule', {}).get('nctId') != args['nct_id']:
            continue
        if request['tool'] == 'get_publication_by_pmid' and (raw.get('source') != 'MED' or str(raw.get('id')) != args['pmid']):
            continue
        if request['tool'] == 'get_open_access_full_text' and raw.get('record_id') != args['pmcid']:
            continue
        matches.append(record.snapshot_id)
    return matches


def execute_followups(execution, organization, *, non_actionable_flags=None):
    """Return an augmented copy plus request decisions; never infer from gap prose."""
    validate_organization_run(execution, organization)
    augmented = deepcopy(execution)
    by_id = {c['request_id']: c for c in followup_catalog(execution)}
    decisions, selected = [], {}
    for group in organization['groups']:
        if group['result']['status'] != 'references_validated' or group['verification']['status'] == 'failed':
            continue
        for index, gap in enumerate(group['result']['structured_output']['review_flags']):
            origin = {'group_id': group['group_id'], 'flag_index': index}
            if (group['group_id'],index) in (non_actionable_flags or {}):
                decisions.append(dict(status='not_required',reason=non_actionable_flags[group['group_id'],index],origin_flags=[origin]))
                continue
            rid = gap.get('followup_request_id')
            if not rid or rid not in by_id:
                decisions.append(dict(status='unmapped', origin_flags=[origin], gap=gap))
                continue
            decision = selected.setdefault(rid, {**deepcopy(by_id[rid]), 'origin_flags': []})
            decision['origin_flags'].append(origin)
    for request in selected.values():
        if request['status'] != 'ready':
            request['status'] = 'already_collected' if request['status'] == 'available' else 'already_attempted'
            request['result_call_id'] = request['existing_call_id']
            decisions.append(request)
            continue
        if any(c.get('followup_round') for c in execution['calls']):
            request['status'] = 'round_limit_reached'
            decisions.append(request)
            continue
        sid = 'followup_' + request['request_id']
        call = {'call_id': sid + ':call', 'step_id': sid, 'source': request['source'],
                'tool': request['tool'], 'arguments': request['arguments'],
                'followup_round': 1, 'origin_flags': request['origin_flags'],
                'origin_locators': request['origins'], 'started_at': datetime.now(timezone.utc).isoformat()}
        try:
            raw = TOOLS[request['tool']](**request['arguments'])
            if hasattr(raw, 'model_dump'):
                raw = raw.model_dump(mode='json')
            call.update(raw_response=raw, source_url=raw['source_url'], status=raw['status'])
            if raw.get('error'):
                call['error'] = raw['error']
        except Exception as exc:
            call.update(status='error', error=f'{type(exc).__name__}: {exc}')
        call['finished_at'] = datetime.now(timezone.utc).isoformat()
        augmented['calls'].append(call)
        augmented['steps'].append({'step_id': sid, 'operation': request['tool'],
                                   'status': 'success' if call['status'] in {'success', 'not_found', 'unavailable'} else 'error'})
        if call['status'] == 'error':
            augmented['status'] = 'incomplete'
        request.update(status=call['status'], result_call_id=call['call_id'])
        decisions.append(request)
    return augmented, decisions
