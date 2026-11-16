from copy import deepcopy
import json

import pytest

import agent1_workflow
import plan_execution as execution
import run_nim_evidence
from execution_organization import prepare_execution
from human_entity_context import HumanConfirmedEntityContext
from natural_language_planner import NaturalLanguageResearchPlanner
from schemas import QueryManifestV2
from test_executable_inventory import runtime_payload
from test_natural_language_planner import FakeAgent
from test_plan_execution import fake_graphql
from test_execution_organization import fake_stream


QUESTION='研究病甲的標的與 SUB_A 的關聯；另解析病乙。'


def context(question=QUESTION, run_id='run-one'):
    return HumanConfirmedEntityContext(original_question=question,
        confirmation_question='病甲與 SUB_A 是否均指同一指定研究範圍？', user_response='是',
        approval_basis='explicit_user_scope_confirmation',run_id=run_id,
        entities=[dict(entity_type='disease',mentions=['病甲','SUB_A'],entity_id='D_A',source_name='Source disease A')])


def payload():
    p=runtime_payload()
    p['original_question']=QUESTION
    p['entities']=[dict(entity_ref=ref,entity_type='disease',mention=mention,resolution_status='unresolved')
                   for ref,mention in [('broad','病甲'),('specific','SUB_A'),('other','病乙')]]
    step=p['plan_steps'][0]
    p['plan_steps']=[{**deepcopy(step),'step_id':f'resolve_{e["entity_ref"]}',
                     'entity_refs':[e['entity_ref']],'outputs':[f'identity_{e["entity_ref"]}']}
                     for e in p['entities']]
    return p


def planned(p=None, c=None):
    p=p or payload()
    agent=FakeAgent([{'structured_response':p}])
    report=NaturalLanguageResearchPlanner(agent=agent,execution_mode=True).run_with_report(
        p['original_question'],max_attempts=1,human_confirmed_context=c)
    return report,agent


def test_same_confirmed_scope_split_by_model_binds_without_deleting_steps():
    report,agent=planned(c=context())
    assert report.status=='passed'
    assert not report.blind_first_attempt
    assert [e.entity_id for e in report.final_plan.entities]==['D_A','D_A',None]
    assert all(e.resolution_status=='unresolved' for e in report.final_plan.entities)
    assert [e.mention for e in report.final_plan.entities]==['病甲','SUB_A','病乙']
    assert len(report.final_plan.plan_steps)==3
    assert context().user_response in json.dumps(agent.calls,ensure_ascii=False)
    assert 'HUMAN research-scope' in json.dumps(agent.calls)
    assert report.human_confirmed_context.run_id=='run-one'


@pytest.mark.parametrize('change', ['question','run'])
def test_confirmation_cannot_be_attached_to_other_question_or_run(change):
    c=context()
    with pytest.raises(ValueError,match='different'):
        c.for_run('other question' if change=='question' else QUESTION,'other-run' if change=='run' else 'run-one')
    plan=QueryManifestV2.model_validate(payload())
    with pytest.raises(ValueError,match='different run'):
        execution.PlanExecutor().run(plan,run_id='other-run',human_confirmed_context=c)


def test_conflicting_confirmation_and_wrong_model_id_are_rejected():
    data=context().model_dump()
    data['entities'].append(dict(entity_type='disease',mentions=['病甲'],entity_id='D_OTHER',source_name='Other'))
    with pytest.raises(ValueError,match='Contradictory'):
        HumanConfirmedEntityContext.model_validate(data)
    p=payload()
    p['entities'][0]['entity_id']='MODEL_WRONG'
    report,_=planned(p,context())
    assert report.final_plan is None
    assert 'contradicts human confirmation' in str(report.attempts[0].errors)


@pytest.mark.parametrize('bad', ['id','name'])
def test_human_confirmation_never_exempts_native_source_identity_check(monkeypatch,bad):
    p=payload()
    p['entities']=p['entities'][:1]
    p['plan_steps']=p['plan_steps'][:1]
    report,_=planned(p,context())
    calls=[]
    def source(query,variables):
        calls.append(variables)
        return {'disease':dict(id='WRONG' if bad=='id' else 'D_A',name='Wrong disease' if bad=='name' else 'Source disease A')}
    monkeypatch.setattr(execution.ot,'_graphql',source)
    result=execution.PlanExecutor().run(report.final_plan,run_id='run-one',human_confirmed_context=context())
    assert result['status']=='incomplete' and result['steps'][0]['status']=='error'
    assert calls==[{'efoId':'D_A'}]


def test_multi_disease_unconfirmed_entity_keeps_own_resolution_and_raw_audit(monkeypatch,tmp_path):
    report,_=planned(c=context())
    def source(query,variables):
        if 'queryString' in variables:
            assert variables['queryString']=='病乙'
            return {'search':{'hits':[]}}
        return {'disease':dict(id='D_A',name='Source disease A')}
    monkeypatch.setattr(execution.ot,'_graphql',source)
    result=execution.PlanExecutor().run(report.final_plan,tmp_path/'execution.json',
                                       run_id='run-one',human_confirmed_context=context())
    assert result['status']=='incomplete'
    assert [s['status'] for s in result['steps']]==['success','success','error']
    for step in result['steps'][:2]:
        proof=step['result']['entities'][0]['resolution_evidence']
        assert proof['human_scope']['approval_basis']=='explicit_user_scope_confirmation'
        assert proof['source_identity']['name']=='Source disease A'
    assert len(result['calls'])==3 and all('raw_response' in c for c in result['calls'])
    prep=prepare_execution(result)
    assert prep.query_context['human_confirmed_context']['run_id']=='run-one'
    assert len(prep.query_context['human_confirmed_entity_resolutions'])==2
    foreign=deepcopy(result)
    foreign['human_confirmed_context']['run_id']='foreign'
    with pytest.raises(ValueError,match='different run'):
        prepare_execution(foreign)


def test_same_question_without_confirmation_is_not_implicitly_resolved(monkeypatch):
    report,_=planned(c=context())
    assert report.final_plan.entities[0].entity_id=='D_A'
    def source(query,variables):
        return {'search':{'hits':[]}} if 'queryString' in variables else {'disease':dict(id='D_A',name='Source disease A')}
    monkeypatch.setattr(execution.ot,'_graphql',source)
    executor=execution.PlanExecutor()
    confirmed_result=executor.run(report.final_plan,run_id='run-one',human_confirmed_context=context())
    assert confirmed_result['steps'][0]['status']=='success'
    unconfirmed,_=planned()
    assert unconfirmed.blind_first_attempt
    assert all(e.entity_id is None for e in unconfirmed.final_plan.entities)
    result=executor.run(unconfirmed.final_plan)
    assert result['status']=='incomplete'
    assert 'human_confirmed_context' not in result
    assert len(result['calls'])==3


def test_two_confirmed_disease_scopes_keep_distinct_ids(monkeypatch):
    c=context()
    data=c.model_dump()
    data['entities'].append(dict(entity_type='disease',mentions=['病乙'],entity_id='D_B',source_name='Source disease B'))
    c=HumanConfirmedEntityContext.model_validate(data)
    report,_=planned(c=c)
    assert [e.entity_id for e in report.final_plan.entities]==['D_A','D_A','D_B']
    def source(query,variables):
        assert 'queryString' not in variables
        id=variables['efoId']
        return {'disease':dict(id=id,name='Source disease A' if id=='D_A' else 'Source disease B')}
    monkeypatch.setattr(execution.ot,'_graphql',source)
    result=execution.PlanExecutor().run(report.final_plan,run_id='run-one',human_confirmed_context=c)
    assert result['status']=='completed'
    assert [s['result']['entities'][0]['id'] for s in result['steps']]==['D_A','D_A','D_B']


def test_run_query_preserves_original_confirmation_and_propagates_to_organizer(monkeypatch,tmp_path):
    p=runtime_payload()
    p['original_question']='病甲與 SUB_A 的治療標的'
    p['entities'][0]['mention']='病甲'
    p['plan_steps'][2]['sources']=['open_targets']
    c=context(p['original_question'],None)
    c.entities[0].source_name='PDAC'  # Native mock source label, not a real-world terminology claim.
    def planner(client,messages):
        text=json.dumps(messages,ensure_ascii=False)
        assert p['original_question'] in text and 'explicit_user_scope_confirmation' in text
        return {'status':'response_received','finish_reason':'stop','response_text':json.dumps(p)}
    monkeypatch.setattr(agent1_workflow,'stream_nim_completion',planner)
    monkeypatch.setattr(execution.ot,'_graphql',fake_graphql)
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',fake_stream)
    result=agent1_workflow.run_query(p['original_question'],object(),tmp_path/'run',human_confirmed_context=c)
    assert result['status']=='references_validated'
    saved=json.loads((tmp_path/'run/execution.json').read_text(encoding='utf-8'))
    assert saved['manifest']['original_question']==p['original_question']
    assert saved['run_id']==result['run_id']==saved['human_confirmed_context']['run_id']
    assert (tmp_path/'run/human_confirmed_context.json').exists()
    assert result['stages']['organization']['followup']['status']=='not_needed'
    assert c.run_id is None  # Caller record was not turned into reusable shared state.
