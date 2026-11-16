from copy import deepcopy
import json

import pytest

import execution_organization as bridge
from evidence_preparation import resolve_json_pointer, selection_input
from evidence_selection import build_catalog
from evidence_verification import verify_evidence_result
import run_nim_evidence


def execution():
    def call(cid, source, raw):
        return dict(call_id=cid, step_id=cid, source=source, tool='test',
                    arguments={'variables': {'efoId': 'D', 'ensemblId': 'T'}},
                    source_url='https://example.org/' + cid, status='success',
                    finished_at='2026-11-08T00:00:00+00:00', raw_response=raw)
    calls = [
        call('association', 'open_targets', {'disease': {'id': 'D', 'associatedTargets': {
            'count': 10, 'rows': [{'target': {'id': 'T', 'approvedSymbol': 'KRAS'}, 'score': .8}]}}}),
        call('tractability', 'open_targets', {'target': {'id': 'T', 'tractability': [
            {'modality': 'SM', 'label': 'Ligand', 'value': True}]}}),
        call('mechanisms', 'chembl', {'mechanisms': [{'molecule_chembl_id': 'CHEMBL1',
             'target_chembl_id': 'CHEMBL2', 'mechanism_of_action': 'inhibitor'}],
             'page_meta': {'total_count': 1, 'next': None}}),
        call('trials', 'clinicaltrials_gov', {'records': [{'protocolSection': {
            'identificationModule': {'nctId': 'NCT00000001'},
            'statusModule': {'overallStatus': 'COMPLETED'}}}], 'total_count': 1, 'truncated': False}),
        call('papers', 'europe_pmc', {'records': [{'id': '123', 'source': 'MED',
             'title': 'Target evidence', 'abstractText': 'Source text', 'doi': '10.1/example'}],
             'total_count': 2, 'truncated': True}),
    ]
    return dict(schema_version='1.0', run_id='run-test', status='completed',
                manifest={'original_question': 'What source material was collected for KRAS?'},
                calls=calls, steps=[{'step_id': c['step_id'], 'status': 'success'} for c in calls])


def test_four_sources_have_native_pointer_identity_and_selectable_units():
    e = execution()
    p = bridge.prepare_execution(e)
    assert len(p.groups) == 4
    target_group = next(g for g in p.groups if g.target_id == 'T')
    assert target_group.record_count == 2
    assert {r.source for r in p.source_records} == {'open_targets', 'chembl', 'clinicaltrials_gov', 'europe_pmc'}
    assert bridge.verify_execution_preparation(p, e)['checked_records'] == 5
    record_map = {r.snapshot_id: r for r in p.source_records}
    calls = {c['call_id']: c for c in e['calls']}
    for group in p.groups:
        _, units = selection_input(p, group, p.research_question)
        assert units
        for unit in units.values():
            if unit['reference_scope'] != 'raw_record':
                continue
            for sid in unit['snapshot_ids']:
                record = record_map[sid]
                origin = record.execution_origin
                native = calls[origin['call_id']]['raw_response']
                assert resolve_json_pointer(native, origin['json_pointer'] + unit['json_pointer']) == unit['value']
    assert any(c['truncated'] for c in p.source_collections)


@pytest.mark.parametrize('damage', ['value', 'url', 'call', 'run', 'time'])
def test_changed_raw_value_or_provenance_is_rejected(damage):
    e = execution()
    p = bridge.prepare_execution(e)
    row = p.source_records[0]
    if damage == 'value':
        row.raw_record['score'] = 99
    elif damage == 'url':
        row.source_url = 'https://wrong.invalid/'
    elif damage == 'call':
        row.execution_origin['call_id'] = 'absent'
    elif damage == 'run':
        row.execution_origin['run_id'] = 'other-run'
    else:
        row.retrieved_at = row.retrieved_at.replace(year=2000)
    with pytest.raises(ValueError):
        bridge.verify_execution_preparation(p, e)


def fake_stream(client, messages):
    from evidence_packet import unpack_input
    payload = unpack_input(json.loads(messages[1]['content']))
    return dict(status='response_received', finish_reason='stop', reasoning_text='',
                response_text=json.dumps({'claims': [{'question_id': 'Q1', 'evidence_ids': [next(iter(payload['catalog']))]}],
                                          'review_flags': []}))


def test_all_groups_organized_without_retrieval_and_with_source_verification(monkeypatch, tmp_path):
    monkeypatch.setattr(run_nim_evidence, 'stream_nim_completion', fake_stream)
    e = execution()
    report = bridge.organize_execution(e, object(), tmp_path / 'organized.json')
    assert report['status'] == 'references_validated'
    assert len(report['groups']) == 4
    assert (tmp_path / 'organized.preparation.json').exists()
    for group in report['groups']:
        assert group['verification']['status'] != 'failed', group['verification']
        assert group['result']['structured_output']['claims']
        for source in group['result']['structured_output']['sources'].values():
            assert source['execution_origin']['run_id'] == e['run_id']


def test_failed_calls_and_empty_searches_are_not_fabricated_facts(monkeypatch, tmp_path):
    e = execution()
    e['status'] = 'incomplete'
    e['calls'][0].update(status='error', error='Timeout')
    e['steps'][0]['status'] = 'error'
    e['calls'][-1]['raw_response'].update(records=[], total_count=0, truncated=False)
    p = bridge.prepare_execution(e)
    assert len(p.groups) == 3
    assert any(c['status'] == 'error' for c in p.query_context['execution_calls'])
    assert any(c['datasource_id'] == 'papers' and c['fetched_evidence_count'] == 0 for c in p.source_collections)
    monkeypatch.setattr(run_nim_evidence, 'stream_nim_completion', fake_stream)
    assert bridge.organize_execution(e, object(), tmp_path / 'partial.json')['status'] == 'incomplete'


def test_invalid_model_reference_fails_without_retry(monkeypatch, tmp_path):
    count = []
    def invalid(client, messages):
        count.append(1)
        return dict(status='response_received', finish_reason='stop', response_text=json.dumps({
            'claims': [{'question_id': 'Q1', 'evidence_ids': ['FAKE']}], 'review_flags': []}))
    monkeypatch.setattr(run_nim_evidence, 'stream_nim_completion', invalid)
    e = execution()
    e['calls'] = e['calls'][:1]
    e['steps'] = e['steps'][:1]
    report = bridge.organize_execution(e, object(), tmp_path / 'bad.json')
    assert len(count) == 1
    assert report['status'] == 'incomplete'
    assert report['groups'][0]['result']['structured_output'] is None
    assert report['groups'][0]['verification']['status'] == 'failed'


def test_related_calls_share_one_packet_without_invented_absent_fields():
    p = bridge.prepare_execution(execution())
    group = next(g for g in p.groups if g.target_id == 'T')
    model_input, units = selection_input(p, group, p.research_question)
    from evidence_packet import unpack_input
    data = unpack_input(json.loads(model_input.messages[1]['content']))
    assert {r['execution_origin']['call_id'] for r in data['records']} == {'association', 'tractability'}
    assert {'/score', '/tractability'} <= {u['json_pointer'] for u in units.values()}
    assert not any(u['field_state'] == 'absent' for u in units.values())


@pytest.mark.parametrize('suffix', ['.preparation.json', '.followup.execution.json', '.followup.preparation.json'])
def test_prepared_file_cannot_overwrite_original_execution(tmp_path, suffix):
    original = tmp_path / ('organization' + suffix)
    original.write_text('original', encoding='utf-8')
    with pytest.raises(ValueError, match='must not overwrite'):
        bridge.organize_execution(execution(), object(), tmp_path / 'organization.json', execution_path=original)
    assert original.read_text(encoding='utf-8') == 'original'
