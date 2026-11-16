from copy import deepcopy
import json
from pathlib import Path

import pytest

from evidence_selection import ScopedMaterialHandoff
from evidence_preparation import selection_input
from execution_coverage import reconcile_gap, validate_coverage_selection
from execution_organization import prepare_execution
from test_execution_coverage import native, by_symbol, reference, gap


def subject_entry(p, symbol='TOP1'):
    group=by_symbol(p,symbol)
    entry=next(c for c in p.query_context['execution_coverage']
               if c['subject_ref']=='target:'+group.target_id and c['material_kind']=='target_drugs')
    return group,entry


def test_saved_whole_call_pointers_are_unverified_not_missing_and_keep_collection_state():
    p=prepare_execution(native())
    cases=json.loads((Path(__file__).parent/'fixtures/gtc_wrong_coverage_fields.json').read_text(encoding='utf-8'))['cases']
    groups={g.group_id:g for g in p.groups}
    checked=0
    for case in cases:
        for flag in [case['original_flag']]:
            if not any(r.get('required_pointer') in {'/disease/associatedTargets','/mechanisms','/target/drugAndClinicalCandidates'} for r in flag.get('coverage_refs') or []):
                continue
            decision=reconcile_gap(flag,p.query_context['execution_coverage'],groups[case['group_id']],p.source_records)
            for r in decision['references']:
                if r['required_pointer'] not in {'/disease/associatedTargets','/mechanisms','/target/drugAndClinicalCandidates'}:
                    continue
                assert r['field_reference_state']=='unverified_field_reference'
                assert r['state'] in {'partial','empty_result'}
                assert not r['absent_field_snapshot_ids']
                assert not r['reuse_snapshot_ids']
                checked+=1
    assert checked>0


def test_inventory_field_id_supplies_pointer_and_unknown_free_path_is_not_proof():
    p=prepare_execution(native())
    group,entry=subject_entry(p)
    field=next(f for f in entry['field_options'] if f['json_pointer']=='/drug')
    ref=reference(entry)
    ref['field_id']=field['field_id']
    selected=ScopedMaterialHandoff.model_validate({'claims':[],'review_flags':[gap([ref])]})
    validate_coverage_selection(selected,p.query_context['execution_coverage'])
    r=reconcile_gap(gap([ref]),p.query_context['execution_coverage'],group,p.source_records)['references'][0]
    assert r['required_pointer'] is None and r['resolved_pointer']=='/drug'
    assert r['field_reference_state']=='inventory_field'
    assert r['detail_state']=='all_returned_records_present'
    wrong=reconcile_gap(gap([reference(entry,'/invented_measurement')]),p.query_context['execution_coverage'],group,p.source_records)['references'][0]
    assert wrong['state']=='unverified_field_reference' and not wrong['absent_field_snapshot_ids']
    model_input,_=selection_input(p,group,p.research_question)
    schema=model_input.response_schema['$defs']['CoverageReference']['properties']
    assert schema['required_pointer']['enum']==[None]
    assert field['field_id'] in schema['field_id']['enum']


@pytest.mark.parametrize('change',['other_field','conflicting_pointer'])
def test_field_reference_cannot_cross_material_or_rewrite_pointer(change):
    p=prepare_execution(native())
    _,entry=subject_entry(p)
    field=next(f for f in entry['field_options'] if f['json_pointer']=='/drug')
    ref=reference(entry)
    ref['field_id']=field['field_id']
    if change=='other_field':
        other=next(c for c in p.query_context['execution_coverage'] if c['coverage_id']!=entry['coverage_id'] and c['field_options'])
        ref['field_id']=other['field_options'][0]['field_id']
    else:
        ref['required_pointer']='/target/drugAndClinicalCandidates'
    selection=ScopedMaterialHandoff.model_validate({'claims':[],'review_flags':[gap([ref])]})
    with pytest.raises(ValueError):
        validate_coverage_selection(selection,p.query_context['execution_coverage'])


@pytest.mark.parametrize('value,expected,usable',[(None,'null',False),(False,'false',True),(0,'zero',True),([], 'empty_array',False),({},'empty_object',False),('','empty_string',False),('observed','value',True),('unknown','value',True)])
def test_native_field_states_are_distinct_and_do_not_change_cache_value_policy(value,expected,usable):
    e=native()
    call=next(c for c in e['calls'] if len((c.get('raw_response',{}).get('target') or {}).get('drugAndClinicalCandidates',{}).get('rows',[]))>2)
    rows=call['raw_response']['target']['drugAndClinicalCandidates']['rows']
    rows[0]['measurement']=value
    # Same native collection has this field; its absence on other rows is known,
    # unlike a field never observed in or declared for this material scope.
    p=prepare_execution(e)
    entry=next(c for c in p.query_context['execution_coverage'] if c['call_id']==call['call_id'])
    group=next(g for g in p.groups if g.group_id in entry['group_ids'])
    field=next(f for f in entry['field_options'] if f['json_pointer']=='/measurement')
    ref={**reference(entry),'field_id':field['field_id']}
    result=reconcile_gap(gap([ref]),p.query_context['execution_coverage'],group,p.source_records)['references'][0]
    assert result['field_observations'][0]['state']==expected
    assert {r['state'] for r in result['field_observations'][1:]}=={'absent'}
    assert len(result['absent_field_snapshot_ids'])==len(rows)-1
    assert bool(result['matched_snapshot_ids'])==usable
    assert not result['unverified_field_snapshot_ids']
    assert next(r for r in p.source_records if r.snapshot_id==result['field_observations'][0]['snapshot_id']).raw_record['measurement']==value


def test_unresolvable_nested_path_is_not_assumed_to_be_known_absent():
    p=prepare_execution(native())
    group,entry=subject_entry(p)
    result=reconcile_gap(gap([reference(entry,'/drug/never_declared_field')]),p.query_context['execution_coverage'],group,p.source_records)['references'][0]
    assert result['field_reference_state']=='unverified_field_reference'
    assert not result['absent_field_snapshot_ids']
    invalid=reconcile_gap(gap([reference(entry,'/invalid~2escape')]),p.query_context['execution_coverage'],group,p.source_records)['references'][0]
    assert invalid['field_reference_state']=='invalid_field_reference'
    assert invalid['detail_state']=='invalid_field_reference'
