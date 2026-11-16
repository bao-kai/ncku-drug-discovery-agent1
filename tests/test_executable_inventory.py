from copy import deepcopy
from types import SimpleNamespace

import pytest

from execution_capabilities import INVENTORY_AGGREGATION, runtime_findings, validate_runtime_repair
from target_evidence_inventory import target_inventory, rank_inventory
from plan_execution import PlanExecutor, execution_issues
from natural_language_planner import NaturalLanguageResearchPlanner
from schemas import QueryManifestV2, RankingSpecV2
from operation_rules import evaluate_operation_rules
from test_natural_language_planner import _valid_payload, FakeAgent
from test_plan_execution import fake_graphql


def runtime_payload():
    p = _valid_payload()
    p['plan_steps'][4]['aggregation'] = deepcopy(INVENTORY_AGGREGATION)
    p['plan_steps'][5]['ranking']['secondary_metrics'] = ['known_drug_count']
    return p


def test_runtime_planner_and_executor_agree_on_same_plan():
    p = runtime_payload()
    report = NaturalLanguageResearchPlanner(agent=FakeAgent([{'structured_response': p}]), execution_mode=True).run_with_report(
        p['original_question'], max_attempts=1)
    assert report.status == 'passed', report.model_dump()
    assert report.execution_checked
    assert execution_issues(report.final_plan) == []


@pytest.mark.parametrize('operation', ['resolve_target_variants', 'retrieve_drug_targets'])
@pytest.mark.parametrize('replacement', ['remove', 'rank_targets'])
def test_runtime_repair_cannot_hide_an_unavailable_capability(operation, replacement):
    payload = runtime_payload()
    extra = deepcopy(payload['plan_steps'][-1])
    extra.update(step_id='unavailable_step', operation=operation, ranking=None,
                 outputs=['requested_material'])
    payload['plan_steps'].append(extra)
    base = QueryManifestV2.model_validate(payload)
    candidate = base.model_copy(deep=True)
    if replacement == 'remove':
        candidate.plan_steps.pop()
    else:
        candidate.plan_steps[-1] = candidate.plan_steps[-2].model_copy(
            update={'step_id': 'unavailable_step', 'outputs': ['requested_material']})
    with pytest.raises(ValueError, match='Preserve the unresolved capability'):
        validate_runtime_repair(base, candidate)


def test_runtime_local_patch_deletion_fails_instead_of_passing_validation():
    payload = runtime_payload()
    payload['plan_steps'].append({
        'step_id': 'unavailable_step', 'operation': 'resolve_target_variants',
        'description': 'Apply the requested source-based threshold.',
        'depends_on': ['step_6'], 'sources': [], 'outputs': ['filtered_targets'],
    })
    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([{'structured_response': payload}]),
        patch_agent=FakeAgent([{'structured_response': {'remove_step_ids': ['unavailable_step']}}]),
        execution_mode=True,
    ).run_with_report(payload['original_question'], max_attempts=2)
    assert report.status == 'failed'
    assert report.final_plan is None
    assert 'Preserve the unresolved capability' in '\n'.join(report.attempts[1].errors)
    assert any(s.step_id == 'unavailable_step' for s in report.attempts[0].candidate_plan.plan_steps)


def test_runtime_full_rewrite_cannot_drop_an_unavailable_capability(monkeypatch):
    monkeypatch.setenv('PLANNER_REPAIR_MODE', 'full_rewrite')
    payload = runtime_payload()
    payload['plan_steps'].append({
        'step_id': 'unavailable_step', 'operation': 'resolve_target_variants',
        'description': 'Apply the requested source-based threshold.',
        'depends_on': ['step_6'], 'sources': [], 'outputs': ['filtered_targets'],
    })
    report = NaturalLanguageResearchPlanner(agent=FakeAgent([
        {'structured_response': payload}, {'structured_response': runtime_payload()},
    ]), execution_mode=True).run_with_report(payload['original_question'], max_attempts=2)
    assert report.status == 'failed'
    assert 'Preserve the unresolved capability' in '\n'.join(report.attempts[1].errors)


def test_runtime_repair_keeps_unavailable_operation_and_allows_binding_repairs():
    payload = runtime_payload()
    payload['plan_steps'][3]['sources'] = ['chembl']
    base = QueryManifestV2.model_validate(payload)
    candidate = QueryManifestV2.model_validate(runtime_payload())
    validate_runtime_repair(base, candidate)
    candidate.plan_steps[-1].operation = 'resolve_target_variants'
    validate_runtime_repair(candidate, candidate.model_copy(deep=True))


@pytest.mark.parametrize('bad', ['aggregation', 'metric', 'source', 'operation'])
def test_runtime_planner_rejects_unexecutable_contract(bad):
    p = runtime_payload()
    if bad == 'aggregation':
        p['plan_steps'][4]['aggregation'] = {'target_translation_layer': 'classify stronger evidence'}
    elif bad == 'metric':
        p['plan_steps'][5]['ranking']['secondary_metrics'] = ['tractability_score']
    elif bad == 'source':
        p['plan_steps'][3]['sources'] = ['chembl']
    else:
        p['plan_steps'][3]['operation'] = 'resolve_target_variants'
    report = NaturalLanguageResearchPlanner(agent=FakeAgent([{'structured_response': p}]), execution_mode=True).run_with_report(
        p['original_question'], max_attempts=1)
    assert report.status == 'failed'
    assert report.attempts[0].schema_valid
    assert report.attempts[0].operation_rule_evaluation.blocking_findings
    assert execution_issues(QueryManifestV2.model_validate(p))


def test_undefined_tractability_metric_not_allowed_even_after_retrieval():
    p = runtime_payload()
    p['plan_steps'][5]['ranking']['secondary_metrics'] = ['tractability_score']
    assert not evaluate_operation_rules(QueryManifestV2.model_validate(p)).passed


def candidate(tid, score=.8):
    return {'disease_id': 'D', 'target_id': tid, 'overall_association_score': score,
            'source_ref': 'association', 'pointer': '/disease/associatedTargets/rows/0'}


def test_inventory_keeps_unknown_zero_false_indications_and_deduplicates_drugs():
    rows = [candidate('A'), candidate('B'), candidate('C')]
    drug_packet = {'target_id': 'A', 'source_ref': 'drugs', 'pointer': '/target/drugAndClinicalCandidates',
                  'records': [{'drug': {'id': 'CHEMBL1'}, 'diseases': [{'disease': {'id': 'OTHER'}}]},
                              {'drug': {'id': 'CHEMBL1'}}, {'drug': {'id': 'CHEMBL2'}}],
                  'total_count': 10, 'truncated': True}
    empty_packet = dict(drug_packet, target_id='B', source_ref='empty', records=[], total_count=0, truncated=False)
    tract = {'target_id': 'A', 'source_ref': 'tract', 'pointer': '/target/tractability',
             'records': [{'modality': 'SM', 'label': 'Ligand', 'value': False}]}
    upstream = [{'targets': rows}, {'target_material': [drug_packet, empty_packet]}, {'target_material': [tract]}]
    before = deepcopy(upstream)
    a,b,c = target_inventory(upstream)
    assert upstream == before
    assert a['known_drug_count'] == 2 and a['known_drug_count_is_lower_bound'] is True
    assert b['known_drug_count'] == 0
    assert c['known_drug_count'] is None
    assert a['tractability_evidence'][0]['records'][0]['value'] is False
    assert a['drug_evidence'][0]['records'][0]['diseases'][0]['disease']['id'] == 'OTHER'
    assert 'tractability_score' not in a
    assert c['evidence_dimensions']['tractability'] == 'not_collected'


def test_secondary_sort_and_unknowns_are_not_zero():
    rows = [dict(candidate('A'), known_drug_count=None), dict(candidate('B'), known_drug_count=0),
            dict(candidate('C'), known_drug_count=3), dict(candidate('D', .9), known_drug_count=0)]
    spec = RankingSpecV2(primary_metric='association_score', direction='descending', secondary_metrics=['known_drug_count'])
    assert [r['target_id'] for r in rank_inventory(rows, spec)] == ['D', 'C', 'B', 'A']
    spec.direction = 'ascending'
    assert [r['target_id'] for r in rank_inventory(rows, spec)] == ['B', 'C', 'A', 'D']


def test_chembl_only_cannot_supply_ot_drug_count():
    p = runtime_payload()
    p['plan_steps'][2]['sources'] = ['chembl']
    assert any('known_drug_count requires' in x['problem'] for x in runtime_findings(QueryManifestV2.model_validate(p)))


def test_bilingual_subtype_resolves_literal_alias_not_broader_disease(monkeypatch):
    import plan_execution as execution
    p = runtime_payload()
    p['original_question'] = '胰臟癌(pancreatic ductal adenocarcinoma, PDAC)有哪些標的？'
    p['entities'][0]['mention'] = '胰臟癌(pancreatic ductal adenocarcinoma, PDAC)'
    p['plan_steps'] = p['plan_steps'][:2]
    searches = []
    def graphql(query, variables):
        if 'queryString' in variables:
            searches.append(variables['queryString'])
            return {'search': {'hits': [{'id': 'MONDO_0005184', 'name': 'pancreatic ductal adenocarcinoma'},
                                       {'id': 'BROAD', 'name': 'pancreatic cancer'}]}}
        return fake_graphql(query, variables)
    monkeypatch.setattr(execution.ot, '_graphql', graphql)
    report = PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'completed', report
    assert searches == ['pancreatic ductal adenocarcinoma']
    assert report['calls'][-1]['arguments']['variables']['efoId'] == 'MONDO_0005184'


def test_real_executor_stratifies_and_ranks_without_extra_queries(monkeypatch):
    import plan_execution as execution
    p = runtime_payload()
    p['entities'][0]['entity_id'] = 'D'
    p['plan_steps'][2]['sources'] = ['open_targets']
    monkeypatch.setattr(execution.ot, '_graphql', fake_graphql)
    report = PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'completed', report
    assert len(report['calls']) == 4
    last = report['steps'][-1]['result']['targets'][0]
    assert last['known_drug_count'] == 1
    assert last['drug_evidence'][0]['source_ref'] == report['calls'][2]['call_id']
    assert last['tractability_evidence'][0]['records'][0]['value'] is True
