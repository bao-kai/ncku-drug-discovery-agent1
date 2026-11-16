from copy import deepcopy

import pytest

from natural_language_planner import (
    ENTITY_MENTION_POLICY, PLAN_ONLY_PROMPT, PLAN_PATCH_PROMPT,
    NaturalLanguageResearchPlanner, QueryManifestPatch, build_repair_scope,
    build_entity_repair_context, validate_patch_scope, apply_manifest_patch,
)
from operation_rules import evaluate_operation_rules
from planner_acceptance import RepairFeedback
from schemas import QueryManifestV2
from test_executable_inventory import runtime_payload
from test_natural_language_planner import FakeAgent


def repair_case(question, broad, association):
    payload=runtime_payload()
    payload['original_question']=question
    payload['entities'][0]['mention']=broad+' ('+association+')'
    base=QueryManifestV2.model_validate(payload)
    rules=evaluate_operation_rules(base)
    feedback=RepairFeedback(incorrect=[dict(area=f.rule_id,problem=f.problem,required_correction=f.required_correction)
                                      for f in rules.blocking_findings])
    scope=build_repair_scope(base,None,rules,feedback)
    original=base.entities[0]
    broad_entity=original.model_copy(update={'mention':broad})
    association_entity=original.model_copy(update={'entity_ref':'association_scope','mention':association})
    patch=QueryManifestPatch(upsert_entities=[broad_entity,association_entity],upsert_steps=[
        s.model_copy(update={'entity_refs':['association_scope']}) for s in base.plan_steps if original.entity_ref in s.entity_refs])
    return base,scope,patch


@pytest.mark.parametrize('question,broad,association',[
    ('Find drugs for condition A using associations with SUB_A.','condition A','SUB_A'),
    ('Study broad disorder and use subtype TYPE_B as the association scope.','broad disorder','TYPE_B'),
    ('Study condition C using SUB_C; retain condition C limitations.','condition C','SUB_C'),
])
def test_noncontiguous_scope_repair_can_split_literals_and_rebind_existing_operations(question,broad,association):
    base,scope,patch=repair_case(question,broad,association)
    assert scope.allow_new_entities
    assert scope.entity_reference_step_ids
    validate_patch_scope(base,patch,scope)
    merged=apply_manifest_patch(base,patch)
    assert {e.mention for e in merged.entities}=={broad,association}
    assert all('association_scope' in s.entity_refs for s in merged.plan_steps if s.entity_refs)
    assert evaluate_operation_rules(merged).passed
    report=NaturalLanguageResearchPlanner(
        agent=FakeAgent([{'structured_response':base.model_dump()}]),
        patch_agent=FakeAgent([{'structured_response':patch.model_dump()}]),execution_mode=True,
    ).run_with_report(question,max_attempts=2)
    assert report.status=='passed'  # Mock transport/permission test, not model ability or research acceptance.
    assert all(e.entity_id is None for e in report.final_plan.entities)
    context=build_entity_repair_context(question,merged,scope)
    occurrences=next(e['literal_occurrences'] for e in context['declared_mentions'] if e['mention']==broad)
    assert len(occurrences)==(2 if broad=='condition C' else 1)
    assert all(question[o['start']:o['end']]==broad for o in occurrences)


def test_multi_disease_repair_does_not_authorize_changing_the_other_entity_branch():
    question='Study condition A via SUB_A, and independent condition B via SUB_B.'
    base,scope,patch=repair_case(question,'condition A','SUB_A')
    other=base.entities[0].model_copy(update={'entity_ref':'other_scope','mention':'SUB_B'})
    base.entities.append(other)
    other_step=base.plan_steps[0].model_copy(update={'step_id':'other_resolve','entity_refs':['other_scope'],'outputs':['other_identity']})
    base.plan_steps.append(other_step)
    assert other_step.step_id not in scope.entity_reference_step_ids
    validate_patch_scope(base,patch,scope)
    merged=apply_manifest_patch(base,patch)
    assert merged.plan_steps[-1].entity_refs==['other_scope']
    wrong=patch.model_copy(deep=True)
    wrong.upsert_steps.append(other_step.model_copy(update={'entity_refs':['association_scope']}))
    with pytest.raises(ValueError,match='outside the authorized repair scope'):
        validate_patch_scope(base,wrong,scope)


@pytest.mark.parametrize('field,value',[('sources',['europe_pmc']),('depends_on',[])])
def test_entity_reference_permission_does_not_allow_source_or_graph_rewrites(field,value):
    base,scope,patch=repair_case('Study condition A via SUB_A.','condition A','SUB_A')
    step=base.plan_steps[1].model_copy(update={field:value})
    bad=QueryManifestPatch(upsert_steps=[step])
    with pytest.raises(ValueError,match='outside the authorized repair scope'):
        validate_patch_scope(base,bad,scope)


def test_incorrect_extra_entity_can_still_be_removed():
    payload=runtime_payload()
    payload['entities'].append(dict(entity_ref='incorrect_extra',entity_type='disease',mention='invented extra',resolution_status='unresolved'))
    report=NaturalLanguageResearchPlanner(agent=FakeAgent([{'structured_response':payload}]),
        patch_agent=FakeAgent([{'structured_response':{'remove_entity_refs':['incorrect_extra']}}]),
        execution_mode=True).run_with_report(payload['original_question'],max_attempts=2)
    assert report.status=='passed'
    assert all(e.entity_ref!='incorrect_extra' for e in report.final_plan.entities)


def test_fresh_and_patch_prompts_share_literal_scope_policy_without_biomedical_fewshot():
    assert ENTITY_MENTION_POLICY in PLAN_ONLY_PROMPT and ENTITY_MENTION_POLICY in PLAN_PATCH_PROMPT
    assert 'contiguous' in ENTITY_MENTION_POLICY and 'different research roles' in ENTITY_MENTION_POLICY
    assert not any(name in ENTITY_MENTION_POLICY for name in ['PDAC','KRAS','CLDN18'])
