from copy import deepcopy
import json
from pathlib import Path

import pytest

from evidence_selection import ScopedMaterialHandoff
from execution_coverage import reconcile_gap, validate_coverage_selection
from execution_organization import prepare_execution, organize_execution
import run_nim_evidence


def native():
    return json.loads((Path(__file__).parent/'fixtures/gtc_coverage_native.json').read_text(encoding='utf-8'))


def by_symbol(prep, symbol):
    row = next(r for r in prep.source_records if (r.raw_record.get('target') or {}).get('approvedSymbol') == symbol)
    tid = row.raw_record['target']['id']
    return next(g for g in prep.groups if g.target_id == tid)


def reference(entry, pointer=None):
    return {**{k:entry[k] for k in ('coverage_id','subject_ref','source','material_kind')},
            'required_pointer':pointer}


def gap(refs):
    return dict(question_id='Q1', evidence_ids=[], missing_material='Requested native material', coverage_refs=refs)


@pytest.mark.parametrize('symbol', ['TOP1','POLE','POLD1'])
def test_saved_problem_flags_are_preserved_and_explicit_cross_group_scope_is_recognized(symbol):
    e = native()
    p = prepare_execution(e)
    group = by_symbol(p, symbol)
    flags = next(f['flags'] for f in e['original_flags'] if f['target_id'] == group.target_id)
    original = deepcopy(flags[0])
    assert reconcile_gap(original, p.query_context['execution_coverage'], group, p.source_records)['state'] == 'coverage_unmapped_requires_review'
    other = by_symbol(p, 'POLE' if symbol == 'TOP1' else 'TOP1')
    entry = next(c for c in p.query_context['execution_coverage']
                 if c['subject_ref']=='target:'+other.target_id and c['material_kind']=='target_drugs')
    annotated = {**original, 'coverage_refs':[reference(entry, '/drug')]}
    result = reconcile_gap(annotated, p.query_context['execution_coverage'], group, p.source_records)
    assert result['original_flag'] == annotated
    assert original == flags[0]
    assert result['references'][0]['state'] == 'collected_elsewhere'
    assert result['references'][0]['reuse_snapshot_ids'] == entry['snapshot_ids']


def test_truncation_empty_and_missing_field_are_not_complete():
    p = prepare_execution(native())
    index = p.query_context['execution_coverage']
    group = by_symbol(p, 'TOP1')
    truncated = next(c for c in index if c['material_kind']=='association_collection')
    assert reconcile_gap(gap([reference(truncated)]), index, group, p.source_records)['references'][0]['state']=='partial'
    empty = next(c for c in index if c['state']=='empty_result' and c['material_kind']=='target_drugs')
    r = reconcile_gap(gap([reference(empty)]), index, group, p.source_records)['references'][0]
    assert r['state']=='empty_result' and not r['reuse_snapshot_ids']
    drugs = next(c for c in index if c['subject_ref']=='target:'+group.target_id and c['material_kind']=='target_drugs')
    r = reconcile_gap(gap([reference(drugs, '/resultsSection')]), index, group, p.source_records)['references'][0]
    assert r['state']=='unverified_field_reference' and not r['reuse_snapshot_ids']
    r = reconcile_gap(gap([reference(drugs)]), index, group, p.source_records)['references'][0]
    assert r['state']=='collected_in_group' and r['detail_state']=='not_specified'
    assert 'require review' in r['judgment']


@pytest.mark.parametrize('field,bad', [('subject_ref','target:WRONG'),('source','europe_pmc'),('material_kind','tractability'),('coverage_id','invented')])
def test_different_entity_source_or_material_cannot_match(field,bad):
    p = prepare_execution(native())
    index = p.query_context['execution_coverage']
    entry = next(c for c in index if c['material_kind']=='target_drugs')
    ref = reference(entry)
    ref[field] = bad
    selected = ScopedMaterialHandoff.model_validate({'claims':[], 'review_flags':[gap([ref])]})
    with pytest.raises(ValueError):
        validate_coverage_selection(selected,index)


def test_same_material_kind_on_another_target_does_not_supply_requested_detail():
    p = prepare_execution(native())
    index = p.query_context['execution_coverage']
    empty = next(c for c in index if c['state']=='empty_result' and c['material_kind']=='target_drugs')
    other = next(c for c in index if c['state']=='collected' and c['material_kind']=='target_drugs')
    assert empty['subject_ref'] != other['subject_ref']
    group = by_symbol(p,'TOP1')
    result = reconcile_gap(gap([reference(empty, '/drug')]),index,group,p.source_records)
    assert result['references'][0]['state']=='empty_result'
    assert not result['references'][0]['matched_snapshot_ids']


def test_failed_and_unexecuted_operations_remain_distinct():
    e = native()
    e['calls'][0].update(status='error',error='source unavailable')
    e['steps'][0]['status']='error'
    e['steps'].append(dict(step_id='blocked',operation='retrieve_tractability',status='blocked'))
    e['manifest']['plan_steps'].append(dict(step_id='blocked',operation='retrieve_tractability',sources=['open_targets']))
    index=prepare_execution(e).query_context['execution_coverage']
    assert any(c['state']=='source_error' for c in index)
    assert any(c['state']=='not_collected' for c in index)


def test_saved_coverage_index_cannot_change_native_subject_scope():
    from execution_organization import verify_execution_preparation
    e=native()
    p=prepare_execution(e)
    p.query_context['execution_coverage'][0]['subject_ref']='target:WRONG'
    with pytest.raises(ValueError,match='Coverage index differs'):
        verify_execution_preparation(p,e)


def test_exact_cross_group_material_reuses_existing_context_without_network(monkeypatch,tmp_path):
    e = native()
    p = prepare_execution(e)
    receiving = by_symbol(p,'POLE')
    providing = by_symbol(p,'TOP1')
    index = p.query_context['execution_coverage']
    entry = next(c for c in index if c['subject_ref']=='target:'+providing.target_id and c['material_kind']=='target_drugs')
    seen=[]
    def model(client,messages):
        from evidence_packet import unpack_input
        data=unpack_input(json.loads(messages[1]['content']))
        seen.append(data['group']['group_id'])
        flags=[]
        if data['group']['group_id']==receiving.group_id and not data['group']['context_snapshot_ids']:
            flags=[gap([reference(entry,'/drug')])]
        return dict(status='response_received',finish_reason='stop',response_text=json.dumps({
            'claims':[dict(question_id='Q1',evidence_ids=[next(iter(data['catalog']))])], 'review_flags':flags}))
    monkeypatch.setattr(run_nim_evidence,'stream_nim_completion',model)
    import automatic_followup
    for tool in automatic_followup.TOOLS:
        monkeypatch.setitem(automatic_followup.TOOLS,tool,lambda **kw:pytest.fail('No new source call'))
    report=organize_execution(e,object(),tmp_path/'out.json')
    assert report['followup']['network_calls']==0
    assert report['followup']['reorganized_groups']==[receiving.group_id]
    assert seen.count(receiving.group_id)==2
    assert report['initial_groups'][next(i for i,g in enumerate(report['initial_groups']) if g['group_id']==receiving.group_id)]['result']['structured_output']['review_flags']
    assert report['initial_coverage_decisions'][0]['references'][0]['state']=='collected_elsewhere'
    assert report['followup']['remaining_flags']==[]
