from copy import deepcopy

import pytest

import plan_execution as execution
from natural_language_planner import NaturalLanguageResearchPlanner
from schemas import QueryManifestV2
from test_executable_inventory import runtime_payload
from test_natural_language_planner import FakeAgent


def entity(mention='廣義疾病', candidates=('broad disease',), kind='disease', ref='disease_1'):
    return dict(entity_ref=ref, entity_type=kind, mention=mention, resolution_status='unresolved',
                search_candidates=[dict(text=n, generation_basis='Unverified translation hypothesis') for n in candidates])


def plan(entities):
    p = runtime_payload()
    p['original_question'] = '；'.join(e['mention'] for e in entities)
    p['entities'] = entities
    template = p['plan_steps'][0]
    p['plan_steps'] = [dict(**{**deepcopy(template), 'step_id': f'resolve_{i}',
                              'entity_refs': [e['entity_ref']], 'outputs': [f'identity_{i}'],
                              'operation': 'resolve_target' if e['entity_type']=='target' else 'resolve_disease'})
                       for i,e in enumerate(entities)]
    return QueryManifestV2.model_validate(p)


def source(monkeypatch, searches, identities):
    def graphql(query, variables):
        if 'queryString' in variables:
            return {'search': {'hits': deepcopy(searches.get(variables['queryString'], []))}}
        kind = 'disease' if 'efoId' in variables else 'target'
        return {kind: deepcopy(identities[next(iter(variables.values()))])}
    monkeypatch.setattr(execution.ot, '_graphql', graphql)


def disease(id='D1', name='broad disease', literal=None):
    return dict(id=id, name=name, description='Source definition', parents=[dict(id='PARENT',name='Parent disease')], ancestors=['ANCESTOR'],
                synonyms=[] if literal is None else [dict(relation='hasExactSynonym', terms=[literal])])


def test_source_backed_original_alias_verifies_and_preserves_all_raw_calls(monkeypatch, tmp_path):
    native = disease(literal='廣義疾病')
    source(monkeypatch, {'broad disease':[dict(id='D1',name='broad disease')]}, {'D1':native})
    report = execution.PlanExecutor().run(plan([entity()]), tmp_path/'execution.json')
    assert report['status']=='completed'
    audit=report['entity_resolution_attempts'][0]
    assert audit['scope_status']=='verified' and audit['identity_status']=='verified'
    assert audit['scope_proof']['matched_term']=='廣義疾病'
    assert report['manifest']['entities'][0]['canonical_name'] is None
    assert [c['raw_response'] for c in report['calls']][-1]=={'disease':native}
    assert len(report['calls'])==3


@pytest.mark.parametrize('candidate_name', ['broad disease','narrow subtype'])
def test_english_existence_never_confirms_chinese_equivalence_or_broad_scope(monkeypatch, candidate_name):
    native=disease(name=candidate_name)
    source(monkeypatch, {candidate_name:[dict(id='D1',name=candidate_name)]}, {'D1':native})
    report=execution.PlanExecutor().run(plan([entity(candidates=(candidate_name,))]))
    assert report['status']=='incomplete'
    audit=report['entity_resolution_attempts'][0]
    assert (audit['status'],audit['identity_status'],audit['scope_status'])==('needs_review','verified','needs_review')
    assert audit['source_scope']['parents']==[dict(id='PARENT',name='Parent disease')]
    assert 'entities' not in report['steps'][0].get('result',{})
    assert report['manifest']['entities'][0]['mention']=='廣義疾病'


def test_distinct_disease_roles_do_not_share_search_candidates(monkeypatch):
    source(monkeypatch, {'broad disease':[dict(id='B',name='broad disease')],
                         'SUBTYPE':[dict(id='S',name='SUBTYPE')]},
           {'B':disease(id='B'), 'S':disease(id='S',name='SUBTYPE')})
    p=plan([entity(), entity('SUBTYPE',(),ref='subtype_scope')])
    report=execution.PlanExecutor().run(p)
    assert [s['status'] for s in report['steps']]==['error','success']
    assert report['steps'][1]['result']['entities'][0]['id']=='S'
    assert [s['text'] for s in report['entity_resolution_attempts'][0]['searches']]==['廣義疾病','broad disease']


@pytest.mark.parametrize('mode,expected', [('ambiguous','ambiguous'),('mismatch','identity_mismatch'),('zero','zero_candidates')])
def test_failure_states_are_distinct_and_keep_trace(monkeypatch, mode, expected):
    leads=('gene A','gene B') if mode=='ambiguous' else ('gene A',)
    hits={name:[dict(id=f'T{i}',name=name)] for i,name in enumerate(leads)} if mode!='zero' else {}
    identities={f'T{i}':dict(id='WRONG' if mode=='mismatch' else f'T{i}',approvedSymbol=name,
                              approvedName=name,synonyms=[]) for i,name in enumerate(leads)}
    source(monkeypatch,hits,identities)
    report=execution.PlanExecutor().run(plan([entity('中文標的',leads,'target')]))
    audit=report['entity_resolution_attempts'][0]
    assert report['status']=='incomplete' and audit['status']==expected
    assert len(audit['searches'])==1+len(leads)
    assert all('raw_response' in c for c in report['calls'])
    assert report['manifest']['entities'][0]['mention']=='中文標的'


def test_chinese_target_requires_native_literal_alias(monkeypatch):
    native=dict(id='T',approvedSymbol='GENE',approvedName='Gene name',synonyms=[dict(label='中文標的',source='HGNC')])
    source(monkeypatch,{'GENE':[dict(id='T',name='GENE')]},{'T':native})
    report=execution.PlanExecutor().run(plan([entity('中文標的',('GENE',),'target')]))
    assert report['status']=='completed'
    assert report['entity_resolution_attempts'][0]['scope_proof']['source']=='HGNC'


def test_search_name_alone_does_not_override_conflicting_native_identity(monkeypatch):
    source(monkeypatch, {'broad disease':[dict(id='D1',name='broad disease')]},
           {'D1':disease(name='different disease')})
    report=execution.PlanExecutor().run(plan([entity()]))
    audit=report['entity_resolution_attempts'][0]
    assert report['status']=='incomplete' and audit['status']=='identity_mismatch'
    assert audit['identity_status']=='unresolved'


def test_same_planner_output_keeps_hypothesis_but_clears_unverified_identity():
    p=plan([entity()]).model_dump()
    p['entities'][0].update(entity_id='MODEL_GUESS',canonical_name='broad disease',resolution_status='verified')
    report=NaturalLanguageResearchPlanner(agent=FakeAgent([{'structured_response':p}]),execution_mode=True).run_with_report(
        p['original_question'],max_attempts=1)
    assert report.status=='passed'
    e=report.final_plan.entities[0]
    assert e.entity_id is None and e.canonical_name is None and e.resolution_status=='unresolved'
    assert e.search_candidates[0].text=='broad disease'
