from copy import deepcopy
import json
from pathlib import Path

import pytest

import query_opentargets as ot
from evidence_preparation import resolve_json_pointer


def fixture():
    return json.loads((Path(__file__).parent/'fixtures/gtc_alias_native.json').read_text(encoding='utf-8'))


def inputs(name):
    data=fixture()
    kind='disease' if name=='PDAC' else 'target'
    candidates=data['saved_candidates'].get(name)
    if candidates is None:
        candidates=next(c['raw_response']['search']['hits'] for c in data['identity_calls']
                        if c['variables'].get('queryString')==name)
    rows={c['raw_response'][kind]['id']:c['raw_response'][kind] for c in data['identity_calls'] if kind in c['raw_response']}
    return kind,candidates,rows


@pytest.mark.parametrize('name,expected', [('PDAC','MONDO_0005184'),('SHP2','ENSG00000179295'),('FAK','ENSG00000169398')])
def test_saved_candidates_resolve_only_by_unique_native_alias_with_pointer(name,expected):
    kind,candidates,rows=inputs(name)
    candidate,proof,identity=ot.choose_source_candidate([name],kind,candidates,rows.__getitem__)
    assert candidate['id']==expected
    assert proof['basis']=='source_identity_alias'
    assert ot._normalized_entity_name(resolve_json_pointer(identity,proof['json_pointer']))==ot._normalized_entity_name(name)


def test_unique_search_candidate_without_source_alias_is_not_accepted():
    kind,candidates,rows=inputs('PDAC')
    rows=deepcopy(rows)
    rows['MONDO_0005184']['synonyms']=[]
    with pytest.raises(ot.EntityResolutionError):
        ot.choose_source_candidate(['PDAC'],kind,candidates,rows.__getitem__)


@pytest.mark.parametrize('relation',['hasRelatedSynonym','hasBroadSynonym','hasNarrowSynonym'])
def test_nonexact_disease_relations_are_not_equivalence(relation):
    kind,candidates,rows=inputs('PDAC')
    rows=deepcopy(rows)
    rows['MONDO_0005184']['synonyms']=[dict(terms=['PDAC'],relation=relation)]
    with pytest.raises(ot.EntityResolutionError):
        ot.choose_source_candidate(['PDAC'],kind,candidates,rows.__getitem__)


def test_two_source_confirmed_aliases_remain_ambiguous():
    kind,candidates,rows=inputs('SHP2')
    rows=deepcopy(rows)
    rows['ENSG00000137078']['synonyms'].append(dict(label='SHP2',source='synthetic_conflict_for_regression'))
    with pytest.raises(ot.EntityResolutionError):
        ot.choose_source_candidate(['SHP2'],kind,candidates,rows.__getitem__)


def test_gene_identity_does_not_establish_cldn18_2_isoform():
    kind,candidates,rows=inputs('CLDN18')
    with pytest.raises(ot.EntityResolutionError):
        ot.choose_source_candidate(['CLDN18.2'],kind,candidates,rows.__getitem__)


def test_candidate_identity_mismatch_is_not_accepted():
    kind,candidates,rows=inputs('PDAC')
    row=deepcopy(rows[candidates[0]['id']])
    row['id']='WRONG'
    with pytest.raises(ot.OpenTargetsError,match='identity'):
        ot.choose_source_candidate(['PDAC'],kind,candidates,lambda _:row)


def test_standalone_resolver_and_executor_share_native_alias_verification(monkeypatch):
    from plan_execution import PlanExecutor
    from schemas import QueryManifestV2
    from test_executable_inventory import runtime_payload
    kind,candidates,rows=inputs('SHP2')
    def graphql(query,variables):
        if 'queryString' in variables:
            return {'search':{'hits':candidates}}
        return {'target':rows[variables['ensemblId']]}
    monkeypatch.setattr(ot,'_graphql',graphql)
    result=ot.resolve_entity('SHP2','target')
    assert result['id']=='ENSG00000179295'
    assert result['resolution_evidence']['native_identity']==rows[result['id']]
    payload=runtime_payload()
    payload['original_question']='Resolve SHP2'
    payload['entities']=[dict(entity_ref='target',entity_type='target',mention='SHP2',resolution_status='unresolved')]
    payload['plan_steps']=[dict(step_id='resolve',operation='resolve_target',description='Resolve the requested target',
                              entity_refs=['target'],sources=['open_targets'],outputs=['resolved_target'])]
    report=PlanExecutor().run(QueryManifestV2.model_validate(payload))
    assert report['status']=='completed'
    entity=report['steps'][0]['result']['entities'][0]
    assert entity['id']==result['id']
    aliases=[c for c in report['calls'] if 'ensemblId' in c['arguments']['variables']]
    assert {c['raw_response']['target']['id'] for c in aliases}=={c['id'] for c in candidates}
    assert entity['resolution_evidence']['source_ref'] in {c['call_id'] for c in aliases}
