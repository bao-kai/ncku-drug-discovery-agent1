from copy import deepcopy
import json
from pathlib import Path

import pytest

from schemas import QueryManifestV2
from operation_rules import evaluate_operation_rules
from planner_acceptance import RepairFeedback
from human_entity_context import HumanConfirmedEntityContext
from natural_language_planner import (NaturalLanguageResearchPlanner,QueryManifestPatch,RepairScope,
    build_repair_scope,validate_patch_scope,apply_manifest_patch,build_repair_operation_catalog)
from test_executable_inventory import runtime_payload
from test_natural_language_planner import FakeAgent


def feedback(rules):
    return RepairFeedback(incorrect=[dict(area=f.rule_id,problem=f'step_id={f.step_id}; '+f.problem,
        required_correction=f.required_correction) for f in rules.blocking_findings])


def scopes(plan):
    rules=evaluate_operation_rules(plan)
    return build_repair_scope(plan,None,rules,feedback(rules))


def entity(ref,kind):
    return dict(entity_ref=ref,mention=ref,entity_type=kind,resolution_status='unresolved')


def step(id,op,refs,deps=[]):
    return dict(step_id=id,operation=op,description=op,entity_refs=refs,depends_on=deps,
                sources=['open_targets'],outputs=[id+'_output'],
                safeguards=['association_score_not_probability','association_does_not_imply_efficacy'])


def branch_case(missing='disease',other_provider=True):
    p=runtime_payload();p['original_question']='da ta db tb'
    p['entities']=[entity('da','disease'),entity('ta','target'),entity('db','disease'),entity('tb','target')]
    existing='target' if missing=='disease' else 'disease'
    ref='ta' if existing=='target' else 'da'
    p['plan_steps']=[step('existing_a','resolve_'+existing,[ref]),
        step('consumer_a','retrieve_exact_pair',['da','ta'],['existing_a'])]
    if other_provider:p['plan_steps'].append(step('other','resolve_'+missing,['db' if missing=='disease' else 'tb']))
    base=QueryManifestV2.model_validate(p)
    provider=base.plan_steps[0].model_copy(deep=True,update=dict(step_id='needed',operation='resolve_'+missing,
        entity_refs=['da' if missing=='disease' else 'ta'],outputs=['needed_output']))
    consumer=base.plan_steps[1].model_copy(deep=True,update={'depends_on':['existing_a','needed']})
    patch=QueryManifestPatch(upsert_steps=[provider,consumer])
    return base,scopes(base),patch


def test_saved_failure_now_accepts_bounded_patch_but_does_not_prove_isoform_resolution():
    fixture=json.loads((Path(__file__).parent/'fixtures/q3_prerequisite_patch.json').read_text(encoding='utf-8'))
    base=QueryManifestV2.model_validate(fixture['base_plan']);patch=QueryManifestPatch.model_validate(fixture['failed_patch'])
    with pytest.raises(ValueError,match='new_step:resolve_disease_pdac'):
        validate_patch_scope(base,patch,RepairScope.model_validate(fixture['historical_scope']))
    context=HumanConfirmedEntityContext.model_validate(fixture['human_context'])
    planner=NaturalLanguageResearchPlanner(agent=object(),execution_mode=True)
    rules=planner._evaluate_rules(base,context);scope=build_repair_scope(base,None,rules,feedback(rules))
    validate_patch_scope(base,patch,scope)
    assert scope.new_operations==[] and scope.step_ids==[]
    assert all('resolve_target' not in p.provider_operations for p in scope.input_repairs)
    assert 'canonical=resolve_disease;' in build_repair_operation_catalog(base,feedback(rules),scope)
    merged=apply_manifest_patch(base,patch);remaining=planner._evaluate_rules(merged,context)
    assert {f.step_id for f in remaining.blocking_findings}=={'retrieve_publications_cldn18_2_pdac'}
    publication=next(s for s in merged.plan_steps if s.operation=='retrieve_publications').model_copy(
        deep=True,update={'depends_on':['resolve_disease_pdac']})
    followup=QueryManifestPatch(upsert_steps=[publication])
    next_scope=build_repair_scope(merged,None,remaining,feedback(remaining))
    validate_patch_scope(merged,followup,next_scope)
    repaired=apply_manifest_patch(merged,followup)
    assert planner._evaluate_rules(repaired,context).passed
    report=NaturalLanguageResearchPlanner(agent=FakeAgent([{'structured_response':base.model_dump()}]),
        patch_agent=FakeAgent([{'structured_response':patch.model_dump()},{'structured_response':followup.model_dump()}]),
        execution_mode=True).run_with_report(base.original_question,max_attempts=3,human_confirmed_context=context)
    assert report.status=='passed'
    assert report.final_plan.original_question==base.original_question
    iso=next(e for e in report.final_plan.entities if e.mention=='CLDN18.2')
    assert iso.entity_id is None and iso.resolution_status=='unresolved'


@pytest.mark.parametrize('missing',['disease','target'])
def test_generic_missing_input_is_scoped_even_when_operation_exists_on_another_entity(missing):
    base,scope,patch=branch_case(missing)
    validate_patch_scope(base,patch,scope)
    merged=apply_manifest_patch(base,patch)
    assert evaluate_operation_rules(merged).passed
    assert next(s for s in merged.plan_steps if s.step_id=='other')==base.plan_steps[-1]


@pytest.mark.parametrize('field,value',[('operation','retrieve_publications'),('sources',['europe_pmc']),
    ('filters',{'unexpected':1}),('outputs',['changed_output']),('entity_refs',['db','ta']),('depends_on',['needed'])])
def test_consumer_operation_conditions_entities_and_existing_dependencies_are_not_open(field,value):
    base,scope,patch=branch_case()
    patch.upsert_steps[1]=patch.upsert_steps[1].model_copy(update={field:value})
    with pytest.raises(ValueError):validate_patch_scope(base,patch,scope)


@pytest.mark.parametrize('change',['other_entity','unrelated_operation','unconnected','duplicate','provider_filter','foreign_dependency'])
def test_no_unrelated_wrong_scope_or_extra_provider_steps(change):
    base,scope,patch=branch_case()
    if change=='other_entity':patch.upsert_steps[0].entity_refs=['db']
    elif change=='unrelated_operation':patch.upsert_steps[0].operation='retrieve_target_drugs'
    elif change=='unconnected':patch.upsert_steps=patch.upsert_steps[:1]
    elif change=='duplicate':
        extra=patch.upsert_steps[0].model_copy(deep=True,update={'step_id':'duplicate','operation':'resolve_diseases'})
        patch.upsert_steps.append(extra);patch.upsert_steps[1].depends_on.append('duplicate')
    elif change=='provider_filter':patch.upsert_steps[0].filters={'all':[]}
    else:patch.upsert_steps[0].depends_on=['other']
    with pytest.raises(ValueError):validate_patch_scope(base,patch,scope)


def test_required_consumer_cannot_be_deleted_and_existing_provider_must_be_reused():
    base,scope,patch=branch_case()
    with pytest.raises(ValueError):validate_patch_scope(base,QueryManifestPatch(remove_step_ids=['consumer_a']),scope)
    provider=patch.upsert_steps[0].model_copy(deep=True,update={'step_id':'ready'})
    base.plan_steps.insert(0,provider)
    scope=scopes(base)
    consumer=base.plan_steps[2].model_copy(deep=True,update={'depends_on':['existing_a','ready']})
    validate_patch_scope(base,QueryManifestPatch(upsert_steps=[consumer]),scope)
    with pytest.raises(ValueError):validate_patch_scope(base,patch,scope)


def test_same_capability_for_two_entities_requires_both_but_allows_two_bounded_providers():
    p=runtime_payload();p['original_question']='da db';p['entities']=[entity('da','disease'),entity('db','disease')]
    p['plan_steps']=[step('consumer','discover_targets',['da','db'])]
    base=QueryManifestV2.model_validate(p);scope=scopes(base)
    a=step('a','resolve_disease',['da']);b=step('b','resolve_disease',['db'])
    consumer=base.plan_steps[0].model_copy(deep=True,update={'depends_on':['a']})
    patch=QueryManifestPatch.model_validate({'upsert_steps':[a,consumer.model_dump()]})
    with pytest.raises(ValueError,match='incomplete_input_entity_scope'):validate_patch_scope(base,patch,scope)
    patch=QueryManifestPatch.model_validate({'upsert_steps':[a,b,consumer.model_copy(update={'depends_on':['a','b']}).model_dump()]})
    validate_patch_scope(base,patch,scope)
    assert evaluate_operation_rules(apply_manifest_patch(base,patch)).passed


def test_feedback_for_unrelated_task_does_not_open_input_repairs():
    base,_,patch=branch_case()
    rules=evaluate_operation_rules(base)
    scope=build_repair_scope(base,None,rules,RepairFeedback(incorrect=[dict(area='ranking',problem='Fix ranking only',required_correction='ranking')]))
    assert scope.input_repairs==[]
    with pytest.raises(ValueError):validate_patch_scope(base,patch,scope)
