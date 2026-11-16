from copy import deepcopy
import json

import pytest

from evidence_packet import pack_input,unpack_input
from evidence_preparation import EvidencePreparation,selection_input
from execution_organization import prepare_execution,verify_execution_preparation,organize_execution
from evidence_selection import (ScopedMaterialHandoff,group_source_subjects,group_selection_scope,
    validate_content_annotation_scopes,assemble_question_selection)
from evidence_verification import verify_evidence_result
from run_nim_evidence import run_prepared_group
from test_execution_organization import execution


def prepared():
    e=execution();p=prepare_execution(e);g=p.groups[0]
    model,units=selection_input(p,g,p.research_question)
    return e,p,g,model,units


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False)


def test_packet_roundtrip_preserves_native_data_provenance_all_units_and_counts():
    e,p,g,model,units=prepared()
    wire=json.loads(model.messages[1]['content']);data=unpack_input(wire)
    assert wire['input_contract']=='references_v1'
    assert wire['data']['query_context']['execution_coverage']=={'$packet_ref':'/execution_coverage'}
    assert canonical(unpack_input(pack_input(data)))==canonical(data)
    assert data['execution_coverage']==p.query_context['execution_coverage']
    assert data['query_context']['execution_coverage']==data['execution_coverage']
    source={r.snapshot_id:r.model_dump(mode='json') for r in p.source_records}
    assert data['records']==[source[s] for s in g.snapshot_ids+g.context_snapshot_ids]
    assert set(data['catalog'])==set(units)
    for key,unit in data['catalog'].items():
        original=units[key]
        assert unit['json_pointer']==original['json_pointer']
        if 'value' in unit:assert canonical(unit['value'])==canonical(original['value'])
        assert [data['records'][i]['snapshot_id'] for i in unit['record_indices']]==original['snapshot_ids']
    assert data['group']['scalar_value_counts']==g.scalar_value_counts
    assert verify_execution_preparation(p,e)['status']=='passed'


def test_native_layouts_distinguish_absent_null_false_zero_and_precision():
    _,_,_,model,_=prepared();data=unpack_input(json.loads(model.messages[1]['content']))
    first=data['records'][0]
    rows=[dict(v=None,z=0,b=False,x=0.12345678901234568),dict(v=False,z=0.0,b=True),dict(z=-0.0,v=0),dict(z=0)]
    data['records']=[{**deepcopy(first),'snapshot_id':f's{i}','raw_record':r} for i,r in enumerate(rows)]
    data['catalog']={};data['group']['scalar_value_counts']={};data['query_context']['relative_expression_rankings']=[]
    decoded=unpack_input(pack_input(data))
    assert canonical(decoded)==canonical(data)
    assert 'v' not in decoded['records'][3]['raw_record']
    assert decoded['records'][0]['raw_record']['v'] is None
    assert decoded['records'][1]['raw_record']['v'] is False


def test_bad_table_reference_rejected_and_legacy_expanded_payload_not_reinterpreted():
    _,_,_,model,_=prepared();wire=json.loads(model.messages[1]['content'])
    wire['data']['records']['rows'][0][2]=-1
    with pytest.raises(ValueError,match='outside'):unpack_input(wire)
    assert unpack_input({'records':[],'legacy':'yes'})=={'records':[],'legacy':'yes'}


def test_evidence_enum_is_shared_and_old_preparation_retains_old_contract():
    _,p,g,model,units=prepared()
    schema=model.response_schema
    assert schema['$defs']['EvidenceId']['enum']==list(units)
    for name in ['DirectLink','ScopedMaterialGap']:
        assert schema['$defs'][name]['properties']['evidence_ids']['items']=={'$ref':'#/$defs/EvidenceId'}
    assert schema['$defs']['ContentAnnotation']['properties']['evidence_ids']['items']=={'$ref':'#/$defs/NativeEvidenceId'}
    raw=p.model_dump(mode='json');raw.pop('model_input_contract')
    old=EvidencePreparation.model_validate(raw)
    legacy,_=selection_input(old,g,p.research_question)
    assert legacy.input_contract=='expanded_v1'
    assert 'input_contract' not in json.loads(legacy.messages[1]['content'])
    assert 'source_subject_ref' not in legacy.response_schema['$defs']['ContentAnnotation']['properties']
    legacy_defs = legacy.response_schema['$defs']
    assert legacy_defs['ContentAnnotation']['properties']['evidence_ids']['items']['enum'] == [
        key for key, unit in units.items() if unit['reference_scope'] == 'raw_record'
    ]
    assert legacy_defs['ScopedMaterialGap']['properties']['evidence_ids']['items']['enum'] == list(units)


def annotation_payload(key,**changes):
    annotation=dict(evidence_ids=[key],entity_ref=None,content_types=['other'],subject_scope='gene_level',interpretation='Native source field')
    annotation.update(changes)
    return ScopedMaterialHandoff(claims=[dict(question_id='Q1',evidence_ids=[key])],review_flags=[],content_annotations=[annotation])


def test_native_annotation_schema_excludes_metadata_but_claims_keep_it():
    _,p,g,model,units=prepared()
    schema=model.response_schema
    excluded=schema['$defs']['NativeEvidenceId']['allOf'][1]['not']['enum']
    assert set(excluded)=={k for k,u in units.items() if u['reference_scope']!='raw_record'}
    assert set(schema['$defs']['EvidenceId']['enum'])-set(excluded)=={
        k for k,u in units.items() if u['reference_scope']=='raw_record'}
    metadata=excluded[0]
    assert metadata in schema['$defs']['EvidenceId']['enum']
    selection=annotation_payload(metadata)
    with pytest.raises(ValueError,match='native source fields'):
        assemble_question_selection(selection,g.group_id,p.research_question,units,
            {r.snapshot_id:r for r in p.source_records})


def test_source_native_scope_legal_without_inventing_planner_target():
    _,p,g,model,units=prepared();subjects=group_source_subjects(p,g)
    assert subjects and not p.query_context['manifest'].get('entities')
    ref=next(iter(subjects));key=next(k for k,u in units.items() if u['reference_scope']=='raw_record')
    annotation=annotation_payload(key,source_subject_ref=ref)
    validate_content_annotation_scopes(annotation,p.query_context,source_subjects=subjects,units=units)
    props=model.response_schema['$defs']['ContentAnnotation']['properties']
    assert props['entity_ref']['enum']==[None]
    assert props['source_subject_ref']['enum']==[None,ref]


@pytest.mark.parametrize('change',['foreign_run','wrong_target','both_namespaces','isoform_claim','wrong_evidence'])
def test_source_native_scope_rejects_wrong_entity_run_reference_or_isoform(change):
    _,p,g,_,units=prepared();subjects=group_source_subjects(p,g);ref=next(iter(subjects))
    key=next(k for k,u in units.items() if u['reference_scope']=='raw_record')
    selection=annotation_payload(key,source_subject_ref=ref)
    a=selection.content_annotations[0]
    if change=='foreign_run':a.source_subject_ref=ref.replace(p.source_run_id,'foreign')
    elif change=='wrong_target':a.source_subject_ref=ref+'-OTHER'
    elif change=='both_namespaces':a.entity_ref='named'
    elif change=='isoform_claim':a.subject_scope='source_explicit_isoform'
    else:units=deepcopy(units);units[key]['snapshot_ids']=['wrong-target-snapshot']
    with pytest.raises(ValueError):validate_content_annotation_scopes(selection,p.query_context,source_subjects=subjects,units=units)


def test_cross_run_record_and_native_target_map_damage_rejected():
    e,p,g,_,_=prepared()
    p.source_records[0].execution_origin['run_id']='foreign'
    with pytest.raises(ValueError):selection_input(p,g,p.research_question)
    e,p,g,_,_=prepared();p.query_context['native_target_scopes'][g.snapshot_ids[0]]='OTHER'
    with pytest.raises(ValueError):verify_execution_preparation(p,e)
    with pytest.raises(ValueError):group_source_subjects(p,g)


def test_v3_selection_reassembly_and_verification_use_same_source_scope(monkeypatch):
    _,p,g,_,units=prepared();key=next(k for k,u in units.items() if u['reference_scope']=='raw_record')
    ref=next(iter(group_source_subjects(p,g)))
    selection=annotation_payload(key,source_subject_ref=ref)
    import run_nim_evidence
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',lambda *args:dict(status='response_received',finish_reason='stop',response_text=selection.model_dump_json()))
    result=run_prepared_group(p,g.group_id,object())
    assert result['status']=='references_validated' and result['output_contract']=='source_material_v4'
    binding=result['structured_output']['content_annotations'][0]['source_subject_binding']
    assert binding['source_run_id']==p.source_run_id and binding['native_target_id']==g.target_id
    assert verify_evidence_result(p,result)['status']!='failed'
    tampered=deepcopy(result);tampered['structured_output']['content_annotations'][0]['source_subject_binding']['native_target_id']='OTHER'
    assert verify_evidence_result(p,tampered)['status']=='failed'
    tampered=deepcopy(result);tampered['input_contract']='expanded_v1'
    assert verify_evidence_result(p,tampered)['status']=='failed'


def test_known_planner_target_cannot_claim_other_native_target():
    _,p,g,_,units=prepared();p.query_context['manifest']['entities']=[dict(entity_ref='named',entity_type='target',entity_id='OTHER')]
    key=next(k for k,u in units.items() if u['reference_scope']=='raw_record')
    selection=annotation_payload(key,entity_ref='named')
    with pytest.raises(ValueError,match='contradicts'):validate_content_annotation_scopes(selection,p.query_context,
        source_subjects=group_source_subjects(p,g),units=units)


def test_new_molecular_schema_excludes_disease_even_unconfirmed_without_regrading_legacy():
    _,p,g,_,units=prepared();p.query_context['manifest']['entities']=[dict(entity_ref='d',entity_type='disease')]
    key=next(k for k,u in units.items() if u['reference_scope']=='raw_record')
    selection=annotation_payload(key,entity_ref='d',subject_scope='unconfirmed')
    validate_content_annotation_scopes(selection,p.query_context)  # Historical rule remains unchanged.
    with pytest.raises(ValueError,match='target entity'):validate_content_annotation_scopes(selection,p.query_context,
        source_subjects=group_source_subjects(p,g),units=units)


def test_empty_selection_is_background_but_keeps_all_native_audit_and_model_path(monkeypatch,tmp_path):
    e=execution();p=prepare_execution(e);g=p.groups[0]
    filters=dict(targets=[],unknown_targets=[],selection_input_keys=[[g.disease_id,g.target_id]],scope='Retrieved only')
    e['steps'].append(dict(step_id='filter',operation='filter_targets',status='success',result=filters))
    import run_nim_evidence
    from test_execution_organization import fake_stream
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',fake_stream)
    report=organize_execution(e,object(),tmp_path/'org.json')
    assert report['target_selection_results'][0]['result']['targets']==[]
    assert len(report['groups'])==len(p.groups)  # No bypass/short-circuit of the existing model flow.
    selected=next(r['result']['structured_output'] for r in report['groups'] if r['result']['group_id']==g.group_id)
    assert selected['selection_scope']['membership']=='excluded'
    assert selected['selection_scope']['material_role']=='candidate_background'
    assert selected['selection_scope']['qualified_count']==0
    after=prepare_execution(e)
    assert [r.raw_record for r in after.source_records]==[r.raw_record for r in p.source_records]


def test_selection_membership_does_not_merge_same_target_across_diseases():
    _,p,g,_,_=prepared();g=g.model_copy(update={'disease_id':''})
    p.query_context['executed_filters']=[dict(step_id='f',result=dict(
        targets=[dict(disease_id='D1',target_id=g.target_id)],unknown_targets=[],
        selection_input_keys=[['D1',g.target_id],['D2',g.target_id]]))]
    scope=group_selection_scope(p,g)
    assert scope['membership']=='mixed' and scope['material_role']=='candidate_background'
    assert [r['membership'] for r in scope['candidate_memberships']]==['selected','excluded']
