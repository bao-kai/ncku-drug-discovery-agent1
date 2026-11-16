import json
from pathlib import Path

import pytest

from execution_capabilities import validate_runtime_repair
from human_entity_context import HumanResearchClarification
from natural_language_planner import NaturalLanguageResearchPlanner, build_repair_scope
from operation_rules import evaluate_question_requirements
from plan_execution import PlanExecutor, execution_issues
from schemas import QueryManifestV2
from test_agent1_new_bindings import entity, payload, step
from test_human_entity_context import context, payload as scope_payload
from test_natural_language_planner import FakeAgent


ROOT = Path(__file__).resolve().parents[1] / 'tests/fixtures/replay'


def saved(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))


def findings(plan, clarification=None, confirmed=None):
    return evaluate_question_requirements(plan, clarification, confirmed)


def status_for(checks, excerpt):
    return next(f.requirement_status for f in checks if f.requirement_excerpt == excerpt)


def test_saved_q4_prose_is_not_a_filter_and_bound_corrected_plan_is_supported():
    original = QueryManifestV2.model_validate(saved('q4_prose_only_plan.json'))
    clarification = HumanResearchClarification.model_validate(
        saved('q4_clarification.json'))
    planner = NaturalLanguageResearchPlanner(agent=object(), execution_mode=True)
    rules = planner._evaluate_rules(original, research_clarification=clarification)
    assert not rules.passed
    assert {'original_question', 'clarification'} <= {
        f.requirement_origin for f in rules.blocking_findings}
    scope = build_repair_scope(original, None, rules)
    assert scope.new_operations == ['filter_targets']
    preflight = PlanExecutor().run(original, research_clarification=clarification)
    assert preflight['status'] == 'not_executable'
    assert preflight['calls'] == []
    assert any(f['requirement_status'] == 'violated' for f in preflight['requirement_findings'])
    corrected = QueryManifestV2.model_validate(
        saved('q4_filter_plan.json'))
    assert status_for(findings(corrected, clarification), clarification.user_response) == 'covered'
    raw = corrected.model_dump()
    filter_step = raw['plan_steps'][-2]
    pocket_step = {**filter_step, 'step_id': 'pocket_filter', 'depends_on': [filter_step['step_id']],
                   'filters': {'all': [filter_step['filters']['all'].pop()]}, 'outputs': ['pocket_and_score']}
    raw['plan_steps'].insert(-1, pocket_step)
    raw['plan_steps'][-1]['depends_on'] = ['pocket_filter']
    chained = QueryManifestV2.model_validate(raw)
    assert status_for(findings(chained, clarification), clarification.user_response) == 'covered'
    assert planner._evaluate_rules(chained, research_clarification=clarification).passed
    chained.plan_steps[-1].depends_on = [chained.plan_steps[1].step_id]
    assert status_for(findings(chained, clarification), clarification.user_response) == 'violated'


def test_saved_q6_proposed_step_cannot_be_deleted_and_missing_clarification_is_visible():
    original = QueryManifestV2.model_validate(saved('q6_proposed_plan.json'))
    clarification = HumanResearchClarification.model_validate(
        saved('q6_clarification.json'))
    deleted = original.model_copy(deep=True)
    removed = {s.step_id for s in deleted.plan_steps if s.operation_status == 'proposed'}
    deleted.plan_steps = [s for s in deleted.plan_steps if s.step_id not in removed]
    for s in deleted.plan_steps:
        s.depends_on = [d for d in s.depends_on if d not in removed]
    with pytest.raises(ValueError, match='cannot be removed'):
        validate_runtime_repair(original, deleted)
    assert status_for(findings(deleted, clarification), clarification.user_response) == 'violated'


def association_plan(question, field='association_score', value=.42):
    p = payload([entity('d', 'DiseaseA', 'disease')], [
        step('resolve', 'resolve_disease', ['d'], ['open_targets']),
        step('discover', 'discover_targets', ['d'], ['open_targets'], ['resolve']),
        {**step('filter', 'filter_targets', [], [], ['discover']),
         'filters': {'all': [{'field': field, 'operator': 'gt', 'value': value}]}},
    ])
    p.original_question = question
    p.global_safeguards += ['association_score_not_probability', 'association_does_not_imply_efficacy']
    return p


@pytest.mark.parametrize(('clause', 'value', 'status'), [
    ('association score > .420', .42, 'covered'),
    ('association score > 0.42', .4, 'violated'),
    ('association score >= 0.42', .42, 'unsupported'),
    ('association score > 0.42 or with prior drugs', .42, 'uncertain'),
    ('association score > 0.42 without prior drugs', .42, 'uncertain'),
])
def test_numeric_operator_value_and_ambiguous_boolean_scope(clause, value, status):
    plan = association_plan('List DiseaseA targets with ' + clause, value=value)
    checks = findings(plan)
    assert checks[0].requirement_status == status
    if status == 'covered':
        planner = NaturalLanguageResearchPlanner(agent=object(), execution_mode=True)
        assert planner._evaluate_rules(plan).passed


def test_confirmed_scope_resolver_does_not_count_for_other_disease_retrieval():
    raw = scope_payload()
    raw['plan_steps'].append(step('discover', 'discover_targets', ['other'], ['open_targets'], ['resolve_other']))
    plan = QueryManifestV2.model_validate(raw)
    assert status_for(findings(plan, confirmed=context()), '病甲 / SUB_A') == 'violated'
    plan.plan_steps[-1].entity_refs = ['broad']
    plan.plan_steps[-1].depends_on = ['resolve_broad']
    assert status_for(findings(plan, confirmed=context()), '病甲 / SUB_A') == 'covered'


def test_comparison_and_exhaustiveness_reports_which_scope_is_missing():
    p = payload([entity('a', 'GENEA'), entity('b', 'GENEB')], [
        step('papers', 'retrieve_publications', ['a'], ['europe_pmc']),
    ])
    p.original_question = 'Compare GENEA and GENEB using all isoform publications'
    checks = findings(p)
    assert status_for(checks, 'Compare GENEA and GENEB') == 'violated'
    assert status_for(checks, 'all') == 'unsupported'
    assert status_for(checks, 'isoform') == 'unsupported'
    p.plan_steps[0].entity_refs = ['a', 'b']
    assert status_for(findings(p), 'Compare GENEA and GENEB') == 'covered'
    assert all(e.entity_id is None for e in p.entities)


def test_full_rewrite_cannot_drop_confirmed_expression_order(monkeypatch):
    monkeypatch.setenv('PLANNER_REPAIR_MODE', 'full_rewrite')
    original = QueryManifestV2.model_validate(saved('q6_proposed_plan.json'))
    clarification = HumanResearchClarification.model_validate(
        saved('q6_clarification.json')).for_run(original.original_question, 'test')
    raw = original.model_dump()
    expression = next(s for s in raw['plan_steps'] if s['operation_status'] == 'proposed')
    expression.update(operation='retrieve_normal_tissue_expression', operation_status='approved',
                      proposed_operation_name=None, proposal_reason=None, sources=['open_targets'],
                      filters={}, aggregation=None,
                      ranking={'primary_metric': 'expression_median', 'direction': 'descending'})
    baseline = QueryManifestV2.model_validate(raw)
    # An independent source defect legitimately triggers a repair.
    baseline.plan_steps[1].sources = ['open_targets']
    regressed = baseline.model_copy(deep=True)
    regressed.plan_steps[1].sources = ['europe_pmc']
    next(s for s in regressed.plan_steps if s.operation == expression['operation']).ranking = None
    agent = FakeAgent([{'structured_response': p.model_dump()} for p in (baseline, regressed)])
    planner = NaturalLanguageResearchPlanner(agent=agent, execution_mode=True)
    report = planner.run_with_report(baseline.original_question, max_attempts=2,
                                     research_clarification=clarification)
    assert report.final_plan is None
    assert any('Repair removed executable coverage' in error for error in report.attempts[-1].errors)


def test_explicit_isoform_label_cannot_silently_be_replaced_by_gene():
    p = payload([entity('gene', 'GENE')], [
        step('papers', 'retrieve_publications', ['gene'], ['europe_pmc']),
    ])
    p.original_question = 'Find publications on isoform GENE.2 of GENE'
    assert status_for(findings(p), 'isoform GENE.2') == 'violated'
    p.entities[0].mention = 'GENE.2'
    checks = findings(p)
    assert not any(f.requirement_status == 'violated' for f in checks)
    assert status_for(checks, 'isoform') == 'unsupported'


@pytest.mark.parametrize(('question', 'excerpt', 'status'), [
    ('Find publications on GENEA; do not investigate expression in normal tissues.',
     'expression in normal tissues', 'uncertain'),
    ('查詢 GENEA 文獻；不要查它在哪些正常組織高表現。', '在哪些正常組織高表現', 'uncertain'),
    ('Report GENEA expression in normal tissues.', 'expression in normal tissues', 'violated'),
    ('GENEA 在哪些正常組織高表現？', '在哪些正常組織高表現', 'violated'),
    ('Find publications on GENEA; do not compare GENEA and GENEB.', 'compare GENEA and GENEB', 'uncertain'),
    ('找出 GENEA 文獻；不要比較 GENEA 與 GENEB。', '比較 GENEA 與 GENEB', 'uncertain'),
    ('Find publications on GENEA and exclude isoform GENEA.2.', 'isoform', 'uncertain'),
    ('Quote “Which normal tissues express GENEA?”', 'Which normal tissues express', 'uncertain'),
    ('If asked, report GENEA expression in normal tissues.', 'expression in normal tissues', 'uncertain'),
])
def test_only_affirmative_requests_impose_obligations(question, excerpt, status):
    p = payload([entity('a', 'GENEA')], [step('papers', 'retrieve_publications', ['a'], ['europe_pmc'])])
    p.original_question = question
    checks = findings(p)
    check = next(f for f in checks if f.requirement_excerpt.startswith(excerpt))
    assert check.requirement_status == status
    if status == 'uncertain':
        assert execution_issues(p) == []


def two_diseases():
    p = association_plan('List DiseaseA targets with association score > 0.42; '
                         'also retrieve DiseaseB targets as background.')
    raw = p.model_dump()
    raw['entities'].append(entity('b', 'DiseaseB', 'disease'))
    raw['plan_steps'][2:2] = [
        step('rb', 'resolve_disease', ['b'], ['open_targets']),
        step('db', 'discover_targets', ['b'], ['open_targets'], ['rb']),
    ]
    return raw


def test_background_filter_cannot_satisfy_main_disease_and_background_rank_does_not_interfere():
    raw = two_diseases()
    raw['plan_steps'][-1]['depends_on'] = ['db']
    raw['plan_steps'].append({
        **step('rank_background', 'rank_targets', [], [], ['db']),
        'ranking': {'primary_metric': 'association_score', 'direction': 'descending'},
    })
    wrong = QueryManifestV2.model_validate(raw)
    assert status_for(findings(wrong), 'association score > 0.42') == 'violated'
    assert any('matching executable filter chain' in issue for issue in execution_issues(wrong))

    raw['plan_steps'][-2]['depends_on'] = ['discover']
    correct = QueryManifestV2.model_validate(raw)
    assert status_for(findings(correct), 'association score > 0.42') == 'covered'
    assert execution_issues(correct) == []


def test_different_disease_filters_cannot_be_combined_as_and():
    raw = two_diseases()
    question = '列出與 DiseaseA 關聯分數 > 0.42、且具有小分子結合口袋的標的；另查 DiseaseB。'
    raw['original_question'] = question
    clarification = HumanResearchClarification(
        original_question=question, confirmation_question='口袋證據口徑？',
        user_response='納入來源明示的預測口袋，與實驗結構證據分開標示')
    raw['plan_steps'].append({
        **step('pocket_b', 'filter_targets', [], [], ['db']),
        'filters': {'all': [{'field': 'small_molecule_pocket_evidence', 'operator': 'is_true', 'value': True}]},
    })
    p = QueryManifestV2.model_validate(raw)
    assert status_for(findings(p, clarification), clarification.user_response) == 'violated'
    p.plan_steps[-1].depends_on = ['db', 'filter']
    assert status_for(findings(p, clarification), clarification.user_response) == 'violated'


def test_multiscope_clause_with_no_clear_disease_binding_is_uncertain():
    raw = two_diseases()
    raw['original_question'] = ('List targets with association score > 0.42; '
                                'consider DiseaseA and DiseaseB.')
    assert status_for(findings(QueryManifestV2.model_validate(raw)), 'association score > 0.42') == 'uncertain'
