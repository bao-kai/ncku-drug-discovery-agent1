from copy import deepcopy
import json
from pathlib import Path

import pytest

import agent1_workflow
import automatic_followup as followup
import plan_execution
import run_nim_evidence
from execution_organization import prepare_execution, finish_followup
from test_automatic_followup import trial_execution, report_request
from test_execution_organization import execution, fake_stream
from test_executable_inventory import runtime_payload
from test_plan_execution import fake_graphql


@pytest.mark.parametrize('foreign_part',['report','group_result'])
@pytest.mark.parametrize('entry',['execute','finish'])
def test_foreign_followup_decision_rejected_before_source_call(monkeypatch,foreign_part,entry,tmp_path):
    e=trial_execution()
    candidate=next(c for c in followup.followup_catalog(e) if c['tool']=='get_publication_by_pmid')
    report=report_request(e,candidate['request_id'])
    report['source_run_id']=e['run_id']
    report['groups'][0]['result']['source_run_id']=e['run_id']
    if foreign_part=='report':
        report['source_run_id']='foreign-synthetic-run'
    else:
        report['groups'][0]['result']['source_run_id']='foreign-synthetic-run'
    calls=[]
    monkeypatch.setitem(followup.TOOLS,'get_publication_by_pmid',lambda **kw:calls.append(kw) or dict(
        source_url='https://example.org/synthetic',status='not_found'))
    with pytest.raises(ValueError,match='run'):
        if entry=='execute':
            followup.execute_followups(e,report)
        else:
            finish_followup(e,report,object(),tmp_path/'out.json')
    assert calls==[]


@pytest.mark.parametrize('origin',['foreign','missing'])
def test_foreign_execution_record_rejected_before_model_call(monkeypatch,origin):
    p=prepare_execution(execution())
    p.source_records[0].source_run_id='foreign-synthetic-run'
    p.source_records[0].execution_origin['run_id']='foreign-synthetic-run'
    if origin=='missing':
        p.source_records[0].execution_origin=None
    group=next(g for g in p.groups if p.source_records[0].snapshot_id in g.snapshot_ids)
    calls=[]
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',lambda *a:calls.append(a) or fake_stream(*a))
    with pytest.raises(ValueError,match='run'):
        run_nim_evidence.run_prepared_group(p,group.group_id,object())
    assert calls==[]


def test_cached_record_from_different_run_rejected_even_with_same_call_and_native_id():
    p=prepare_execution(execution())
    row=next(r for r in p.source_records if r.source=='europe_pmc')
    row.source_run_id='foreign-synthetic-run'
    row.execution_origin['run_id']='foreign-synthetic-run'
    row.raw_record={'id':'456','source':'MED'}
    request=dict(source_run_id=p.source_run_id,result_call_id=row.execution_origin['call_id'],
                 tool='get_publication_by_pmid',arguments={'pmid':'456'})
    with pytest.raises(ValueError,match='run'):
        followup.matching_records(request,[row])


def test_two_fresh_queries_share_client_but_not_messages_or_source_records(monkeypatch,tmp_path):
    # Synthetic transport test, not live generalization or biomedical evidence.
    client=object()
    active={}
    planner_messages=[]
    organizer_inputs=[]
    def planner(actual_client,messages):
        assert actual_client is client
        question=next(q for q in ('Find targets for SYNTHETIC_ALPHA','Find targets for SYNTHETIC_BETA')
                      if q in json.dumps(messages))
        marker='ALPHA' if question.endswith('ALPHA') else 'BETA'
        active['marker']=marker
        payload=runtime_payload()
        payload['original_question']=question
        payload['entities'][0]['mention']='SYNTHETIC_'+marker
        payload['plan_steps'][2]['sources']=['open_targets']
        planner_messages.append(dict(question=question,messages=deepcopy(messages)))
        return dict(status='response_received',finish_reason='stop',response_text=json.dumps(payload))
    def graphql(query,variables):
        if 'queryString' in variables:
            return {'search':{'hits':[dict(id='D_'+active['marker'],name=variables['queryString'])]}}
        raw=fake_graphql(query,variables)
        if 'disease' in raw:
            raw['disease']['name']='SYNTHETIC_'+active['marker']
            if 'associatedTargets' in raw['disease']:
                raw['disease']['associatedTargets']['rows'][0]['target']['approvedSymbol']='EVIDENCE_ONLY_'+active['marker']
        return raw
    def organizer(actual_client,messages):
        assert actual_client is client
        from evidence_packet import unpack_input
        organizer_inputs.append(unpack_input(json.loads(messages[1]['content'])))
        return fake_stream(actual_client,messages)
    monkeypatch.setattr(agent1_workflow,'stream_nim_completion',planner)
    monkeypatch.setattr(plan_execution.ot,'_graphql',graphql)
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',organizer)
    results=[agent1_workflow.run_query('Find targets for SYNTHETIC_'+m,client,tmp_path/m) for m in ('ALPHA','BETA')]
    assert all(r['status']=='references_validated' for r in results)
    for index,other in ((0,'BETA'),(1,'ALPHA')):
        assert 'SYNTHETIC_'+other not in json.dumps(planner_messages[index]['messages'])
        assert 'EVIDENCE_ONLY_'+other not in json.dumps(organizer_inputs[index])
    runs=[json.loads((tmp_path/m/'execution.json').read_text(encoding='utf-8')) for m in ('ALPHA','BETA')]
    assert runs[0]['run_id']!=runs[1]['run_id']
    record_sets=[{r['snapshot_id'] for r in data['records']} for data in organizer_inputs]
    assert not record_sets[0]&record_sets[1]
    assert [{r['source_run_id'] for r in data['records']} for data in organizer_inputs]==[{r['run_id']} for r in runs]
