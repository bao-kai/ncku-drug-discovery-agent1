from copy import deepcopy
import json

import pytest

import automatic_followup as followup
import execution_organization as bridge
import run_nim_evidence
from evidence_selection import ScopedMaterialHandoff
from test_execution_organization import execution


def trial_execution():
    e = execution()
    e['calls'] = [e['calls'][3]]
    e['steps'] = [e['steps'][3]]
    e['calls'][0]['raw_response']['records'][0]['protocolSection']['referencesModule'] = {
        'references': [{'pmid': '456', 'citation': 'A citation without an abstract'}]}
    return e


def response(payload):
    return dict(status='response_received', finish_reason='stop', response_text=json.dumps(payload))


def organizer(client, messages):
    from evidence_packet import unpack_input
    data = unpack_input(json.loads(messages[1]['content']))
    claims = [{'question_id': 'Q1', 'evidence_ids': [next(iter(data['catalog']))]}]
    if any(r['source'] == 'europe_pmc' and r['raw_record'].get('id') == '456' for r in data['records']):
        return response({'claims': claims, 'review_flags': []})
    request = next(c for c in data['followup_candidates'] if c['arguments'] == {'pmid': '456'})
    return response({'claims': claims, 'review_flags': [{'question_id': 'Q1', 'evidence_ids': [],
        'missing_material': 'Abstract for the cited PMID 456', 'followup_request_id': request['request_id']}]})


def publication(pmid):
    assert pmid == '456'
    return dict(source='europe_pmc', source_url='https://example.org/456', record_id=pmid,
                status='success', retrieved_at='2026-11-08T00:00:00+00:00',
                raw_record={'id': pmid, 'source': 'MED', 'title': 'Title', 'abstractText': 'Native abstract'})


def test_real_gap_to_exact_tool_to_reorganization(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run_nim_evidence, 'stream_nim_completion', organizer)
    monkeypatch.setitem(followup.TOOLS, 'get_publication_by_pmid', lambda **kw: calls.append(kw) or publication(**kw))
    e = trial_execution()
    before = deepcopy(e)
    report = bridge.organize_execution(e, object(), tmp_path / 'out.json')
    assert e == before
    assert calls == [{'pmid': '456'}]
    request = report['followup']['requests'][0]
    assert request['status'] == 'success'
    assert request['origins'][0]['json_pointer'].endswith('/pmid')
    assert request['result_snapshot_ids']
    assert len(report['initial_groups'][0]['result']['structured_output']['review_flags']) == 1
    assert report['groups'][0]['result']['structured_output']['review_flags'] == []
    assert report['followup']['remaining_flags'] == []
    augmented = json.loads((tmp_path / 'out.followup.execution.json').read_text(encoding='utf-8'))
    assert augmented['calls'][-1]['origin_flags']
    assert augmented['calls'][-1]['followup_round'] == 1
    assert report['source_integrity']['checked_records'] == 2


def test_cached_material_reused_without_network(monkeypatch, tmp_path):
    e = trial_execution()
    cached = deepcopy(execution()['calls'][4])
    cached['raw_response']['records'] = [publication('456')['raw_record'], {'id': '999', 'source': 'MED'}]
    e['calls'].append(cached)
    e['steps'].append({'step_id': cached['step_id'], 'status': 'success'})
    monkeypatch.setattr(run_nim_evidence, 'stream_nim_completion', organizer)
    monkeypatch.setitem(followup.TOOLS, 'get_publication_by_pmid', lambda **kw: pytest.fail('Should reuse cached record'))
    report = bridge.organize_execution(e, object(), tmp_path / 'out.json')
    assert report['followup']['requests'][0]['status'] == 'already_collected'
    assert len(report['followup']['requests'][0]['result_snapshot_ids']) == 1
    assert report['followup']['remaining_flags'] == []


@pytest.mark.parametrize('status', ['error', 'not_found', 'unavailable'])
def test_unsuccessful_lookup_remains_unresolved(monkeypatch, tmp_path, status):
    monkeypatch.setattr(run_nim_evidence, 'stream_nim_completion', organizer)
    monkeypatch.setitem(followup.TOOLS, 'get_publication_by_pmid', lambda **kw: {
        'source_url': 'https://example.org/456', 'status': status, 'raw_record': None, 'error': status})
    report = bridge.organize_execution(trial_execution(), object(), tmp_path / 'out.json')
    assert report['followup']['requests'][0]['status'] == status
    assert report['followup']['remaining_flags']
    assert report['followup']['reorganized_groups'] == []
    if status == 'error':
        assert report['status'] == 'incomplete'


def report_request(e, request_id, groups=1):
    return {'source_run_id':e['run_id'], 'groups': [dict(group_id=f'g{i}', result={'source_run_id':e['run_id'], 'status': 'references_validated', 'structured_output': {
        'review_flags': [{'followup_request_id': request_id}]}}, verification={'status': 'needs_review'}) for i in range(groups)]}


def test_repeated_flags_execute_only_one_lookup(monkeypatch):
    e = trial_execution()
    candidate = next(c for c in followup.followup_catalog(e) if c['tool'] == 'get_publication_by_pmid')
    calls = []
    monkeypatch.setitem(followup.TOOLS, 'get_publication_by_pmid', lambda **kw: calls.append(kw) or publication(**kw))
    augmented, requests = followup.execute_followups(e, report_request(e, candidate['request_id'], 3))
    assert len(calls) == 1
    assert len(requests[0]['origin_flags']) == 3


def test_unmapped_gap_does_not_guess_parameters():
    e = trial_execution()
    augmented, requests = followup.execute_followups(e, report_request(e, None))
    assert augmented['calls'] == e['calls']
    assert requests[0]['status'] == 'unmapped'


def test_model_cannot_invent_a_request_or_conflict_with_study_subject():
    gap = {'question_id': 'Q1', 'evidence_ids': [], 'missing_material': 'Status', 'followup_request_id': 'fake'}
    selected = ScopedMaterialHandoff.model_validate({'claims': [], 'review_flags': [gap]})
    with pytest.raises(ValueError, match='Unknown'):
        followup.validate_followup_selection(selected, [])
    candidate = next(c for c in followup.followup_catalog(trial_execution()) if c['tool'] == 'get_clinical_trial')
    gap.update(followup_request_id=candidate['request_id'], subject_ref='NCT99999999')
    with pytest.raises(ValueError, match='conflicts'):
        followup.validate_followup_selection(ScopedMaterialHandoff.model_validate({'claims': [], 'review_flags': [gap]}), [candidate])


def test_next_page_uses_exact_original_search_and_stops_after_one_round(monkeypatch):
    e = execution()
    e['calls'] = [e['calls'][4]]
    c = e['calls'][0]
    c.update(tool='search_publications', arguments={'query': '"PDAC" AND KRAS'})
    c['raw_response'].update(next_cursor_mark='native-next', truncated=True)
    e['steps'] = [e['steps'][4]]
    candidate = next(c for c in followup.followup_catalog(e) if c['tool'] == 'search_publications')
    assert candidate['arguments'] == {'query': '"PDAC" AND KRAS', 'cursor_mark': 'native-next'}
    def page(**kw):
        assert kw == candidate['arguments']
        return {'source_url': 'https://example.org/page2', 'status': 'success', 'records': [],
                'total_count': 50, 'truncated': True, 'next_cursor_mark': 'third'}
    monkeypatch.setitem(followup.TOOLS, 'search_publications', page)
    augmented, _ = followup.execute_followups(e, report_request(e, candidate['request_id']))
    third = next(c for c in followup.followup_catalog(augmented) if c['arguments'].get('cursor_mark') == 'third')
    _, decisions = followup.execute_followups(augmented, report_request(augmented, third['request_id']))
    assert decisions[0]['status'] == 'round_limit_reached'


def test_native_pmcid_produces_fulltext_binding_without_guessing_url():
    e = execution()
    e['calls'][4]['raw_response']['records'][0]['pmcid'] = 'PMC123456'
    candidate = next(c for c in followup.followup_catalog(e) if c['tool'] == 'get_open_access_full_text')
    assert candidate['arguments'] == {'pmcid': 'PMC123456'}
    assert candidate['origins'][0]['json_pointer'] == '/records/0/pmcid'


def test_fulltext_native_xml_is_preserved_in_preparation():
    e = execution()
    c = deepcopy(e['calls'][4])
    c.update(call_id='xml', step_id='xml', tool='get_open_access_full_text', arguments={'pmcid': 'PMC123456'},
             raw_response={'status': 'success', 'record_id': 'PMC123456', 'raw_xml': '<article>native</article>'})
    e['calls'].append(c)
    e['steps'].append({'step_id': 'xml', 'status': 'success'})
    p = bridge.prepare_execution(e)
    record = next(r for r in p.source_records if r.execution_origin['call_id'] == 'xml')
    assert record.raw_record['raw_xml'] == '<article>native</article>'


@pytest.mark.parametrize('identifier,tool,args', [
    ('NCT12345678', 'get_clinical_trial', {'nct_id': 'NCT12345678'}),
    ('PMC123456', 'get_open_access_full_text', {'pmcid': 'PMC123456'}),
])
def test_explicit_question_locator_dispatches_correct_tool(monkeypatch, identifier, tool, args):
    e = trial_execution()
    e['manifest']['original_question'] = f'Please retrieve {identifier}'
    candidate = next(c for c in followup.followup_catalog(e) if c['tool'] == tool and c['arguments'] == args)
    assert candidate['origins'] == [{'reference_scope': 'manifest', 'json_pointer': '/original_question'}]
    called = []
    monkeypatch.setitem(followup.TOOLS, tool, lambda **kw: called.append(kw) or {
        'source_url': 'https://example.org', 'status': 'not_found'})
    _, decisions = followup.execute_followups(e, report_request(e, candidate['request_id']))
    assert called == [args]
    assert decisions[0]['status'] == 'not_found'
