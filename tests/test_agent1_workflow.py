import json

import agent1_workflow
import plan_execution
import run_nim_evidence
from test_executable_inventory import runtime_payload
from test_plan_execution import fake_graphql
from test_execution_organization import fake_stream


def test_natural_language_to_final_delivery_uses_runtime_contract(monkeypatch, tmp_path):
    payload = runtime_payload()
    payload['plan_steps'][2]['sources'] = ['open_targets']
    def planner(client, messages):
        assert 'source_evidence_inventory_v1' in json.dumps(messages)
        return {'status': 'response_received', 'finish_reason': 'stop',
                'response_text': json.dumps(payload, ensure_ascii=False)}
    def source(query, variables):
        if 'queryString' in variables:
            return {'search': {'hits': [{'id': 'D', 'name': payload['entities'][0]['mention']}]}}
        return fake_graphql(query, variables)
    monkeypatch.setattr(agent1_workflow, 'stream_nim_completion', planner)
    monkeypatch.setattr(plan_execution.ot, '_graphql', source)
    monkeypatch.setattr(run_nim_evidence, 'stream_nim_completion', fake_stream)
    result = agent1_workflow.run_query(payload['original_question'], object(), tmp_path/'run')
    assert result['status'] == 'references_validated'
    assert result['stages']['planner']['execution_checked'] is True
    assert result['stages']['organization']['followup']['status'] == 'not_needed'
    assert result['delivery']['analysis_step_ids'] == ['step_5', 'step_6']
    saved = json.loads((tmp_path/'run'/'execution.json').read_text(encoding='utf-8'))
    assert saved['steps'][-1]['result']['targets'][0]['known_drug_count'] == 1
    assert (tmp_path/'run'/'planner_call_1.input.json').exists()


def test_unexecutable_natural_language_plan_stops_before_any_source_call(monkeypatch, tmp_path):
    payload = runtime_payload()
    payload['plan_steps'][5]['ranking']['secondary_metrics'] = ['tractability_score']
    monkeypatch.setattr(agent1_workflow, 'stream_nim_completion', lambda *a: {
        'status': 'response_received', 'finish_reason': 'stop', 'response_text': json.dumps(payload)})
    def forbidden(*a, **kw):
        raise AssertionError('Unexecutable Plan must not reach tools')
    monkeypatch.setattr(plan_execution.ot, '_graphql', forbidden)
    result = agent1_workflow.run_query(payload['original_question'], object(), tmp_path/'failed')
    assert result['status'] == 'blocked_at_planner'
    assert 'execution' not in result['stages']
    assert not (tmp_path/'failed'/'execution.json').exists()
