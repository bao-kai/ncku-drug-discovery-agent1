from copy import deepcopy
import json

import pytest

import plan_execution as execution
import run_nim_evidence
from schemas import QueryManifestV2
from execution_organization import prepare_execution,organize_execution,verify_execution_preparation
from evidence_preparation import selection_input
from evidence_packet import unpack_input
from evidence_selection import ScopedMaterialHandoff,assemble_question_selection,question_catalog
from evidence_verification import verify_evidence_result
from filter_assessment import assessed_filters,group_assessed_result,filter_answer,already_assessed_filter_input
from test_agent1_new_bindings import payload,entity,step


def source_execution(monkeypatch,score=.6,pocket=False,fail=False,datasources=None):
    p=payload([entity('d','condition','disease','D')],[step('resolve','resolve_disease',['d'],['open_targets']),
        step('discover','discover_targets',['d'],['open_targets'],['resolve']),
        step('tract','retrieve_tractability',[],['open_targets'],['discover']),
        step('filter','filter_targets',['d'],[],['discover','tract'])])
    raw=p.model_dump();raw['original_question']='condition targets association > 0.7 with small molecule pocket evidence'
    raw['global_safeguards']+=['association_score_not_probability','association_does_not_imply_efficacy']
    raw['plan_steps'][-1]['filters']={'all':[dict(field='overall_association_score',operator='gt',value=.7),
        dict(field='small_molecule_pocket_evidence',operator='is_true',value=True)]}
    def gql(query,variables):
        if 'DiseaseIdentity' in query:return dict(disease=dict(id='D',name='condition'))
        if 'associatedTargets' in query:return dict(disease=dict(id='D',associatedTargets=dict(count=100,rows=[
            dict(target=dict(id='T',approvedSymbol='TARGET'),score=score,datasourceScores=datasources or [])])))
        if 'tractability' in query:
            if fail:raise RuntimeError('Source unavailable')
            return dict(target=dict(id='T',tractability=[dict(label=label,modality='SM',value=pocket) for label in
                ['High-Quality Pocket','Med-Quality Pocket','Structure with Ligand']]))
        raise AssertionError(query)
    monkeypatch.setattr(execution.ot,'_graphql',gql)
    return execution.PlanExecutor().run(QueryManifestV2.model_validate(raw))


def empty_model(*args):return dict(status='response_received',finish_reason='stop',response_text='{"claims":[],"review_flags":[],"content_annotations":[]}')


def test_evaluated_empty_allows_empty_model_response_and_no_followup(monkeypatch,tmp_path):
    e=source_execution(monkeypatch);p=prepare_execution(e);g=p.groups[0]
    a=assessed_filters(e)[0]
    assert (a['state'],a['evaluated_count'],a['qualified_count'],a['unknown_count'])==('evaluated_empty',1,0,0)
    assert a['predicate']['all'][0]['value']==.7 and a['source_calls']
    model,_=selection_input(p,g,p.research_question)
    assert unpack_input(json.loads(model.messages[1]['content']))['assessed_result']['allow_empty_response']
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',empty_model)
    report=organize_execution(e,object(),tmp_path/'org.json')
    assert report['status']=='references_validated'
    assert report['followup']['status']=='not_needed' and report['followup']['network_calls']==0
    result=report['groups'][0]['result']
    assert result['structured_output']['claims']==result['structured_output']['review_flags']==[]
    assert result['structured_output']['assessed_result']['state']=='evaluated_empty'
    assert result['structured_output']['selection_scope']['material_role']=='candidate_background'
    assert verify_evidence_result(p,result)['status']!='failed'
    text=filter_answer(e,report,'/saved/execution.json')
    assert '符合 0 筆' in text and '1/100' in text and '不表示全資料庫' in text
    assert '來源 API' in text


@pytest.mark.parametrize('case',['unknown','failure','tampered','matches'])
def test_real_missing_failure_forged_zero_or_nonempty_do_not_get_empty_permission(monkeypatch,case):
    e=source_execution(monkeypatch,score=.9,pocket=None if case=='unknown' else True,fail=case=='failure')
    if case=='tampered':e['steps'][-1]['result']['targets']=[]
    p=prepare_execution(e);g=p.groups[0]
    assessed=group_assessed_result(p,g)
    assert not assessed or not assessed['allow_empty_response']
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',empty_model)
    r=run_nim_evidence.run_prepared_group(p,g.group_id,object())
    assert r['status']=='invalid_response'
    if case=='unknown':assert p.query_context['assessed_filters'][0]['unknown_count']==1
    if case=='failure':assert any(c['state']=='source_error' for c in p.query_context['execution_coverage'])
    if case=='tampered':assert p.query_context['assessed_filters'][0]['state']=='not_evaluated'


def test_positive_qualified_results_preserve_flags_and_never_turn_background_into_result(monkeypatch):
    e=source_execution(monkeypatch,score=.9,pocket=True);a=assessed_filters(e)[0]
    assert a['state']=='evaluated_matches' and a['qualified_count']==1
    row=a['qualified_targets'][0]
    assert row['predicted_pocket_flags']['High-Quality Pocket'] is True
    assert row['experimental_structure_with_ligand'] is True
    p=prepare_execution(e);g=p.groups[0]
    assert not group_assessed_result(p,g)['allow_empty_response']


def test_legacy_empty_and_wrong_question_key_stay_invalid(monkeypatch):
    e=source_execution(monkeypatch);p=prepare_execution(e);g=p.groups[0]
    empty=ScopedMaterialHandoff(claims=[],review_flags=[])
    with pytest.raises(ValueError):assemble_question_selection(empty,g.group_id,p.research_question,{}, {})
    assert question_catalog(p.research_question)=={'Q1':p.research_question}
    model,units=selection_input(p,g,p.research_question)
    wrong=ScopedMaterialHandoff(claims=[dict(question_id='Q4',evidence_ids=[next(iter(units))])],review_flags=[])
    with pytest.raises(ValueError,match='question_id'):assemble_question_selection(wrong,g.group_id,p.research_question,units,{r.snapshot_id:r for r in p.source_records})


def test_forged_assessment_is_rejected_by_execution_verifier(monkeypatch):
    e=source_execution(monkeypatch);p=prepare_execution(e)
    p.query_context['assessed_filters'][0]['evaluated_count']=999
    with pytest.raises(ValueError,match='assessment'):verify_execution_preparation(p,e)


def test_present_typed_predicate_input_not_reused_just_to_find_matches(monkeypatch,tmp_path):
    e=source_execution(monkeypatch);p=prepare_execution(e);g=p.groups[0]
    c=next(c for c in p.query_context['execution_coverage'] if c['material_kind']=='association_collection')
    field=next(f for f in c['field_options'] if f['json_pointer']=='/score')
    def model(*args):return dict(status='response_received',finish_reason='stop',response_text=json.dumps(dict(claims=[],review_flags=[dict(
        question_id='Q1',evidence_ids=['E1'],missing_material='No qualifying target in retrieved scope',
        coverage_refs=[dict(coverage_id=c['coverage_id'],subject_ref=c['subject_ref'],source=c['source'],material_kind=c['material_kind'],field_id=field['field_id'])])],content_annotations=[])))
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',model)
    report=organize_execution(e,object(),tmp_path/'org.json')
    assert report['followup']['status']=='not_needed'
    assert report['followup']['requests'][0]['status']=='not_required'
    assert report['groups'][0]['result']['structured_output']['review_flags']  # Original flag retained, never silently erased.
    decision=report['initial_coverage_decisions'][0]
    assert decision['disposition']=='filter_input_already_evaluated'
    altered=deepcopy(decision);altered['references'][0]['detail_state']='source_null'
    assert not already_assessed_filter_input(group_assessed_result(p,g),altered)
    requested=deepcopy(decision);requested['original_flag']['followup_request_id']='explicit-material-request'
    assert not already_assessed_filter_input(group_assessed_result(p,g),requested)


def test_present_truncated_collection_scope_flag_kept_without_cache_reorganization(monkeypatch,tmp_path):
    e=source_execution(monkeypatch,datasources=[dict(id='source',score=.6)]);p=prepare_execution(e)
    c=next(c for c in p.query_context['execution_coverage'] if c['material_kind']=='association_collection')
    field=next(f for f in c['field_options'] if f['json_pointer']=='/datasourceScores')
    response=dict(claims=[],review_flags=[dict(question_id='Q1',evidence_ids=['E1'],
        missing_material='Full collection beyond retrieved page',coverage_refs=[dict(
        coverage_id=c['coverage_id'],subject_ref=c['subject_ref'],source=c['source'],
        material_kind=c['material_kind'],field_id=field['field_id'])])],content_annotations=[])
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',lambda *args:dict(
        status='response_received',finish_reason='stop',response_text=json.dumps(response)))
    report=organize_execution(e,object(),tmp_path/'org.json')
    assert report['followup']['status']=='not_needed'
    assert report['followup']['network_calls']==0
    assert report['groups'][0]['result']['structured_output']['review_flags']
    decision=report['initial_coverage_decisions'][0]
    assert decision['disposition']=='filter_input_already_evaluated'
    assessed=group_assessed_result(p,p.groups[0])
    for state in ['source_null','absent','source_error','unknown']:
        changed=deepcopy(decision);changed['references'][0]['detail_state']=state
        assert not already_assessed_filter_input(assessed,changed)
    changed=deepcopy(decision);changed['references'][0]['call_id']='foreign'
    assert not already_assessed_filter_input(assessed,changed)
    changed=deepcopy(decision);changed['original_flag']['followup_request_id']='explicit-material-request'
    assert not already_assessed_filter_input(assessed,changed)
