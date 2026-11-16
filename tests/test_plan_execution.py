from copy import deepcopy
import json

import pytest

import plan_execution as execution
from schemas import QueryManifestV2
from test_natural_language_planner import _valid_payload


def plan():
    p = _valid_payload()
    p['original_question'] = 'PDAC known targets'
    p['global_safeguards'].append('association_score_not_probability')
    p['plan_steps'] = p['plan_steps'][:4]
    p['plan_steps'][2]['sources'] = ['open_targets']
    p['entities'][0].update(mention='PDAC', entity_id='MONDO_PD', resolution_status='unresolved')
    return p


def fake_graphql(query, variables):
    if 'DiseaseIdentity' in query:
        return {'disease': {'id': variables['efoId'], 'name': 'PDAC'}}
    if 'associatedTargets' in query:
        return {'disease': {'id': variables['efoId'], 'name': 'PDAC',
                           'associatedTargets': {'count': 100, 'rows': [
                               {'target': {'id': 'ENSG_K', 'approvedSymbol': 'KRAS'},
                                'score': .8, 'datasourceScores': [{'id': 'somatic', 'score': .8}]}]}}}
    if 'drugAndClinicalCandidates' in query:
        return {'target': {'id': variables['ensemblId'], 'drugAndClinicalCandidates':
                          {'count': 1, 'rows': [{'drug': {'id': 'CHEMBL1'},
                                               'diseases': [{'disease': {'id': 'OTHER'}}]}]}}}
    if 'tractability' in query:
        return {'target': {'id': variables['ensemblId'], 'tractability':
                          [{'label': 'High-Quality Ligand', 'modality': 'SM', 'value': True}]}}
    raise AssertionError(query)


def run(payload, monkeypatch):
    monkeypatch.setattr(execution.ot, '_graphql', fake_graphql)
    return execution.PlanExecutor().run(QueryManifestV2.model_validate(payload))


def test_dependency_ids_raw_provenance_and_indication_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(execution.ot, '_graphql', fake_graphql)
    path = tmp_path / 'execution.json'
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(plan()), path)
    assert report['status'] == 'completed', report['preflight_issues']
    assert len(report['calls']) == 4
    assert report['calls'][1]['arguments']['variables'] == {'efoId': 'MONDO_PD', 'pageSize': 10}
    assert report['calls'][2]['arguments']['variables'] == {'ensemblId': 'ENSG_K'}
    material = report['steps'][2]['result']['target_material'][0]
    assert material['records'][0]['diseases'][0]['disease']['id'] == 'OTHER'
    call = next(c for c in report['calls'] if c['call_id'] == material['source_ref'])
    assert call['raw_response']['target']['drugAndClinicalCandidates']['rows'] == material['records']
    assert report['steps'][1]['result']['collections'][0]['truncated'] is True
    assert json.loads(path.read_text(encoding='utf-8'))['run_id'] == report['run_id']


@pytest.mark.parametrize('change', ['operation', 'source', 'filter', 'metric', 'aggregation'])
def test_unsupported_never_calls_api(monkeypatch, change):
    p = plan()
    s = p['plan_steps'][2]
    if change == 'operation':
        s['operation'] = 'invented_operation'
    elif change == 'source':
        s['sources'] = ['europe_pmc']
    elif change == 'filter':
        s['filters'] = {'min_evidence_score': .5}
    elif change == 'metric':
        s['ranking'] = {'primary_metric': 'tractability_score', 'direction': 'descending'}
    else:
        s['aggregation'] = {'method': 'make a conclusion'}
    def forbidden(*a, **kw):
        pytest.fail('No API call allowed before executable preflight')
    monkeypatch.setattr(execution.ot, '_graphql', forbidden)
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'not_executable'
    assert report['calls'] == []


def test_error_blocks_dependents_but_independent_branch_runs(monkeypatch):
    def fake(query, variables):
        if 'drugAndClinicalCandidates' in query:
            raise RuntimeError('503')
        return fake_graphql(query, variables)
    monkeypatch.setattr(execution.ot, '_graphql', fake)
    p = plan()
    p['plan_steps'].append(dict(step_id='blocked', operation='retrieve_tractability',
                               description='Tractability', depends_on=['step_3'],
                               entity_refs=['disease_1'], sources=['open_targets'], outputs=['blocked_output']))
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'incomplete'
    assert [s['status'] for s in report['steps']] == ['success', 'success', 'error', 'success', 'blocked']
    assert 'raw_response' not in report['calls'][2]


def test_ambiguous_resolution_preserves_candidates_and_does_not_choose_first(monkeypatch):
    p = plan()
    p['entities'][0]['entity_id'] = None
    def native(query, variables):
        if 'queryString' in variables:
            return {'search': {'hits': [
                {'id': 'WRONG1', 'name': 'pancreatic cancer'}, {'id': 'WRONG2', 'name': 'other'}]}}
        return {'disease': {'id': variables['efoId'], 'name': 'Nonmatching disease', 'synonyms': []}}
    monkeypatch.setattr(execution.ot, '_graphql', native)
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'incomplete'
    assert len(report['calls']) == 3
    assert report['calls'][0]['raw_response']['search']['hits'][0]['id'] == 'WRONG1'
    assert {c['arguments']['variables']['efoId'] for c in report['calls'][1:]} == {'WRONG1','WRONG2'}


def test_source_error_is_not_empty_success(monkeypatch):
    p = plan()
    p['plan_steps'] = [dict(step_id='trials', operation='retrieve_clinical_trials', description='Trials',
                            entity_refs=['disease_1'], sources=['clinicaltrials_gov'], outputs=['trials'])]
    monkeypatch.setattr(execution, 'search_clinical_trials', lambda **kw: {'status': 'error', 'error': 'timeout', 'records': []})
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'incomplete'
    assert report['calls'][0]['raw_response']['status'] == 'error'


def test_two_search_sources_one_page_each_no_hidden_filters(monkeypatch):
    p = plan()
    p['plan_steps'] = [dict(step_id='safety', operation='retrieve_safety_evidence', description='Source material',
                            entity_refs=['disease_1'], sources=['clinicaltrials_gov', 'europe_pmc'], outputs=['safety'])]
    received = []
    def fake(**kwargs):
        received.append(kwargs)
        return {'status': 'success', 'records': [{'id': 'record'}], 'truncated': True, 'total_count': 21}
    monkeypatch.setattr(execution, 'search_clinical_trials', fake)
    monkeypatch.setattr(execution, 'search_publications', fake)
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'completed', report['preflight_issues']
    assert received == [{'condition': '"PDAC"', 'intervention': '', 'term': ''},
                        {'query': '"PDAC" AND (safety OR toxicity OR "adverse event" OR "adverse events")'}]
    assert len(report['calls']) == 2
    assert all(p['interpretation_status'] == 'source_material_only'
               for p in report['steps'][0]['result']['source_material'])


def test_chembl_uses_native_protein_mapping_and_limits(monkeypatch):
    p = plan()
    p['plan_steps'][2]['sources'] = ['chembl']
    def gql(query, variables):
        if 'proteinIds' in query:
            return {'target': {'id': variables['id'], 'proteinIds': [{'id': 'P12345', 'source': 'uniprot_swissprot'}]}}
        return fake_graphql(query, variables)
    calls = []
    def get(path, params):
        calls.append((path, params))
        page = {'next': None, 'total_count': 1}
        if path == 'target.json':
            return {'targets': [{'target_chembl_id': 'CHEMBL_TARGET', 'target_type': 'PROTEIN COMPLEX',
                                 'target_components': [{'accession': 'P12345'}]}], 'page_meta': page}
        return {'mechanisms': [{'target_chembl_id': 'CHEMBL_TARGET', 'molecule_chembl_id': 'CHEMBL_DRUG'}], 'page_meta': page}
    monkeypatch.setattr(execution.ot, '_graphql', gql)
    monkeypatch.setattr(execution.chembl, '_get', get)
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'completed', report
    assert calls[0][1]['target_components__accession__in'] == 'P12345'
    assert calls[1][1]['target_chembl_id__in'] == 'CHEMBL_TARGET'
    assert all(args['limit'] == 20 for _, args in calls)
    assert report['steps'][2]['result']['chembl_material']['target_records'][0]['target_type'] == 'PROTEIN COMPLEX'


def test_raw_identity_mismatch_fails_before_downstream(monkeypatch):
    monkeypatch.setattr(execution.ot, '_graphql', lambda *a, **kw: {'disease': {'id': 'WRONG', 'name': 'PDAC'}})
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(plan()))
    assert report['status'] == 'incomplete'
    assert len(report['calls']) == 1


def test_ranking_only_within_dependency_candidates(monkeypatch):
    p = plan()
    p['plan_steps'] = p['plan_steps'][:2]
    p['plan_steps'].append(dict(step_id='rank', operation='rank_targets', description='Rank',
                               depends_on=['step_2'], entity_refs=['disease_1'], sources=[], outputs=['ranked'],
                               ranking={'primary_metric': 'overall_association_score', 'direction': 'descending'}))
    report = run(p, monkeypatch)
    assert report['status'] == 'completed'
    assert len(report['calls']) == 2
    assert report['steps'][-1]['result']['targets'][0]['target_id'] == 'ENSG_K'


def test_pai1_exact_pair_uses_ids_without_discovery_window(monkeypatch):
    p = plan()
    p['original_question'] = 'PDAC and PAI-1 association'
    p['entities'].append(dict(entity_ref='pai1', entity_type='target', mention='PAI-1',
                              entity_id='ENSG00000106366', resolution_status='unresolved'))
    p['plan_steps'] = [dict(step_id='resolve', operation='resolve_disease_target', description='Resolve both',
                            entity_refs=['disease_1', 'pai1'], sources=['open_targets'], outputs=['identities']),
                       dict(step_id='pair', operation='retrieve_exact_pair', description='Exact pair',
                            depends_on=['resolve'], entity_refs=['disease_1', 'pai1'],
                            sources=['open_targets'], outputs=['association'])]
    def fake(query, variables):
        if 'TargetIdentity' in query:
            return {'target': {'id': variables['ensemblId'], 'approvedSymbol': 'SERPINE1'}}
        if 'associatedTargets' in query:
            assert 'Bs: [$ensemblId]' in query
            assert variables == {'efoId': 'MONDO_PD', 'ensemblId': 'ENSG00000106366'}
            return {'disease': {'id': 'MONDO_PD', 'associatedTargets': {'count': 0, 'rows': []}}}
        return fake_graphql(query, variables)
    monkeypatch.setattr(execution.ot, '_graphql', fake)
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'completed', report['preflight_issues']
    assert len(report['calls']) == 3
    assert report['steps'][-1]['result']['collections'][0]['total_count'] == 0

    p['plan_steps'].append(dict(
        step_id='rank_pair', operation='rank_targets', description='Rank exact-pair results',
        depends_on=['pair'], sources=[], outputs=['ranked_pair'],
        ranking={'primary_metric': 'overall_association_score', 'direction': 'descending'},
    ))
    ranked = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert ranked['status'] == 'completed', ranked['preflight_issues']
    assert len(ranked['calls']) == 3
    assert ranked['steps'][-1]['result']['targets'] == []


def test_exact_pairs_rank_nonempty_results_with_native_source_refs(monkeypatch):
    p = plan()
    p['original_question'] = 'PDAC association with PAI-1 and KRAS'
    p['entities'] = [p['entities'][0], *[
        dict(entity_ref=ref, entity_type='target', mention=mention,
             entity_id=target_id, resolution_status='unresolved')
        for ref, mention, target_id in [
            ('pai1', 'PAI-1', 'ENSG00000106366'), ('kras', 'KRAS', 'ENSG00000133703')
        ]
    ]]
    refs = ['disease_1', 'pai1', 'kras']
    p['plan_steps'] = [
        dict(step_id='resolve', operation='resolve_disease_targets', description='Resolve IDs',
             entity_refs=refs, sources=['open_targets'], outputs=['identities']),
        dict(step_id='pairs', operation='retrieve_exact_pairs', description='Retrieve exact pairs',
             depends_on=['resolve'], entity_refs=refs, sources=['open_targets'], outputs=['association']),
        dict(step_id='rank', operation='rank_targets', description='Rank association scores',
             depends_on=['pairs'], sources=[], outputs=['ranked'],
             ranking={'primary_metric': 'overall_association_score', 'direction': 'descending'}),
    ]
    native = {'ENSG00000106366': ('SERPINE1', 0.31), 'ENSG00000133703': ('KRAS', 0.82)}

    def fake(query, variables):
        if 'TargetIdentity' in query:
            target_id = variables['ensemblId']
            return {'target': {'id': target_id, 'approvedSymbol': native[target_id][0]}}
        if 'associatedTargets' in query:
            assert 'Bs: [$ensemblId]' in query
            target_id = variables['ensemblId']
            assert variables == {'efoId': 'MONDO_PD', 'ensemblId': target_id}
            symbol, score = native[target_id]
            return {'disease': {'id': 'MONDO_PD', 'associatedTargets': {'count': 1, 'rows': [
                {'target': {'id': target_id, 'approvedSymbol': symbol},
                 'score': score, 'datasourceScores': []}
            ]}}}
        return fake_graphql(query, variables)

    monkeypatch.setattr(execution.ot, '_graphql', fake)
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'completed', report['preflight_issues']
    assert [call['step_id'] for call in report['calls']] == ['resolve'] * 3 + ['pairs'] * 2
    pair_calls = report['calls'][-2:]
    assert [call['arguments']['variables']['ensemblId'] for call in pair_calls] == list(native)
    ranked = report['steps'][-1]['result']['targets']
    assert [(row['disease_id'], row['target_id'], row['overall_association_score'],
             row['source_ref'], row['pointer']) for row in ranked] == [
        ('MONDO_PD', 'ENSG00000133703', 0.82, pair_calls[1]['call_id'], '/disease/associatedTargets/rows/0'),
        ('MONDO_PD', 'ENSG00000106366', 0.31, pair_calls[0]['call_id'], '/disease/associatedTargets/rows/0'),
    ]


def test_candidate_data_cannot_leak_from_unrelated_branch(monkeypatch):
    p = plan()
    p['plan_steps'][2]['depends_on'] = ['step_1']
    report = run(p, monkeypatch)
    assert report['status'] == 'not_executable'
    assert report['calls'] == []


def test_no_chembl_mapping_is_not_no_drugs(monkeypatch):
    p = plan()
    p['plan_steps'][2]['sources'] = ['chembl']
    def fake(query, variables):
        if 'proteinIds' in query:
            return {'target': {'id': variables['id'], 'proteinIds': []}}
        return fake_graphql(query, variables)
    monkeypatch.setattr(execution.ot, '_graphql', fake)
    monkeypatch.setattr(execution.chembl, '_get', lambda *a, **kw: pytest.fail('No query without mapped IDs'))
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'completed'
    material = report['steps'][2]['result']['chembl_material']
    assert material['status'] == 'mapping_unavailable'
    assert 'records' not in material


def test_termination_filter_is_native_status_not_safety_interpretation(monkeypatch):
    p = plan()
    p['plan_steps'] = [dict(step_id='stop', operation='retrieve_trial_terminations', description='Terminations',
                            entity_refs=['disease_1'], sources=['clinicaltrials_gov'], outputs=['stops'])]
    def search(**kw):
        assert kw['overall_statuses'] == ['TERMINATED']
        return {'status': 'not_found', 'records': [], 'total_count': 0, 'truncated': False}
    monkeypatch.setattr(execution, 'search_clinical_trials', search)
    report = execution.PlanExecutor().run(QueryManifestV2.model_validate(p))
    assert report['status'] == 'completed'
    assert report['steps'][0]['result']['source_material'][0]['material']['status'] == 'not_found'
