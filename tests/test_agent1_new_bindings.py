from copy import deepcopy
import json
from pathlib import Path

import pytest

import plan_execution as execution
from schemas import QueryManifestV2
from target_evidence_inventory import filter_inventory,target_inventory
from execution_organization import prepare_execution
from evidence_selection import ScopedMaterialHandoff,assemble_question_selection,build_catalog,validate_content_annotation_scopes
from human_entity_context import HumanConfirmedEntityContext
from test_executable_inventory import runtime_payload


def row(id,score,disease='D',**extra):
    return dict(disease_id=disease,target_id=id,overall_association_score=score,
                source_ref='native-call',pointer='/native/rows/'+id,**extra)


def gt(value,field='overall_association_score'):
    return dict(field=field,operator='gt',value=value)


def test_strict_boundary_unknown_and_conjunction_keep_provenance():
    rows=[row('A',.69,known_drug_count=3),row('B',.7,known_drug_count=3),
          row('C',.71,known_drug_count=2),row('D',None,known_drug_count=2),
          row('E',.9,known_drug_count=0)]
    original=deepcopy(rows)
    result=filter_inventory(rows,{'all':[gt(.7),gt(0,'known_drug_count')]})
    assert result['targets']==[rows[2]] and result['unknown_targets']==[rows[3]]
    assert rows==original and result['targets'][0]['source_ref']=='native-call'


@pytest.mark.parametrize('value',[True,'0.7',float('nan'),float('inf')])
def test_invalid_numeric_threshold_and_native_value_rejected(value):
    with pytest.raises(ValueError): filter_inventory([row('A',.8)],{'all':[gt(value)]})
    with pytest.raises(ValueError): filter_inventory([row('A',value)],{'all':[gt(.7)]})


def test_selection_does_not_resurrect_excluded_ancestor_or_mix_diseases():
    a=row('same',.69); b=row('same',.9,'OTHER')
    filtered=filter_inventory([a],{'all':[gt(.7)]})
    result=target_inventory([{'targets':[a,b]},filtered])
    assert [(r['disease_id'],r['target_id']) for r in result]==[('OTHER','same')]


def test_predicted_pocket_is_not_an_executable_physical_pocket_filter():
    with pytest.raises(ValueError,match='no pocket predicate'):
        filter_inventory([row('A',.9)],{'all':[gt(0,'physical_pocket')]})


def test_datasource_scores_and_human_scope_require_native_upstream_before_execution(monkeypatch):
    e=[entity('d','condition','disease','D'),entity('t','target',id='T')]
    p=payload(e,[step('target','resolve_target',['t'],['open_targets']),
        step('scores','retrieve_datasource_scores',['d','t'],['open_targets'],['target'])])
    assert any('resolved_disease' in error for error in execution.execution_issues(p))
    p=payload([entity('d','condition','disease','D')],[step('papers','retrieve_publications',['d'],['europe_pmc'])])
    c=HumanConfirmedEntityContext(original_question=p.original_question,confirmation_question='condition scope',
        user_response='confirmed',approval_basis='explicit_user_scope_confirmation',run_id='test-run',
        entities=[dict(entity_type='disease',mentions=['condition'],entity_id='D',source_name='condition')])
    monkeypatch.setattr(execution,'search_publications',lambda **kwargs:pytest.fail('No source API before missing native check'))
    result=execution.PlanExecutor().run(p,run_id='test-run',human_confirmed_context=c)
    assert result['status']=='not_executable' and result['calls']==[]
    assert any('upstream native resolution' in error for error in result['preflight_issues'])


def test_multi_target_content_annotations_cannot_hide_subject_binding():
    selection=ScopedMaterialHandoff(claims=[dict(question_id='Q1',evidence_ids=['E1'])],review_flags=[],
      content_annotations=[dict(evidence_ids=['E1'],content_types=['expression'],subject_scope='source_explicit_isoform',interpretation='Source wording')])
    context=dict(manifest=dict(entities=[entity('a','A'),entity('b','B')]))
    with pytest.raises(ValueError,match='explicit entity_ref'):
        validate_content_annotation_scopes(selection,context)
    selection.content_annotations[0].entity_ref='a'
    validate_content_annotation_scopes(selection,context)
    selection.content_annotations[0].entity_ref='foreign'
    with pytest.raises(ValueError,match='unknown entity'):
        validate_content_annotation_scopes(selection,context)


def test_exact_pair_requires_the_selected_target_binding_before_any_source_call(monkeypatch):
    p = payload(
        [entity('d', 'disease', 'disease', 'D'), entity('a', 'A', id='A'), entity('b', 'B', id='B')],
        [step('resolve', 'resolve_disease_target', ['d', 'a'], ['open_targets']),
         step('pair', 'retrieve_exact_pair', ['d', 'b'], ['open_targets'], ['resolve'])],
    )
    monkeypatch.setattr(execution.ot, '_graphql', lambda *args: pytest.fail('Preflight must stop calls'))
    report = execution.PlanExecutor().run(p)
    assert report['status'] == 'not_executable'
    assert report['calls'] == []
    findings = execution.runtime_findings(p)
    assert any(f.get('missing') == ['resolved_target'] and f.get('input_entity_refs') == ['b']
               for f in findings)
    p.plan_steps[-1].entity_refs = ['d']
    assert any('explicit target' in issue for issue in execution.execution_issues(p))


def test_known_drug_count_filter_requires_open_targets_not_chembl():
    p = payload(
        [entity('d', 'disease', 'disease', 'D')],
        [step('resolve', 'resolve_disease', ['d'], ['open_targets']),
         step('find', 'discover_targets', ['d'], ['open_targets'], ['resolve']),
         step('drugs', 'retrieve_target_drugs', [], ['chembl'], ['find']),
         {**step('filter', 'filter_targets', [], [], ['drugs']),
          'filters': {'all': [gt(0, 'known_drug_count')]}}],
    )
    assert any('requires upstream retrieve_target_drugs from open_targets' in issue
               for issue in execution.execution_issues(p))
    p.plan_steps[2].sources = ['open_targets']
    p.global_safeguards += ['association_score_not_probability', 'association_does_not_imply_efficacy']
    assert execution.execution_issues(p) == []


def test_human_disease_scope_verified_then_isoform_topic_search_needs_no_gene_id(monkeypatch):
    p=payload([entity('d','病甲','disease','D'),entity('iso','NAMED.2')],
        [step('resolve','resolve_disease',['d'],['open_targets']),
         step('papers','retrieve_publications',['d','iso'],['europe_pmc'],['resolve'])])
    c=HumanConfirmedEntityContext(original_question=p.original_question,confirmation_question='Research scope?',
        user_response='confirmed',approval_basis='explicit_user_scope_confirmation',run_id='test-run',
        entities=[dict(entity_type='disease',mentions=['病甲'],entity_id='D',source_name='Source disease')])
    calls=[]
    def gql(query,variables):
        calls.append(('identity',variables));return {'disease':dict(id='D',name='Source disease')}
    def search(**kwargs):
        calls.append(('papers',kwargs));return dict(source_url='https://example.org/search',status='not_found',records=[],total_count=0,truncated=False)
    monkeypatch.setattr(execution.ot,'_graphql',gql);monkeypatch.setattr(execution,'search_publications',search)
    result=execution.PlanExecutor().run(p,run_id='test-run',human_confirmed_context=c)
    assert result['status']=='completed'
    assert calls==[('identity',{'efoId':'D'}),('papers',{'query':'"Source disease" AND "NAMED.2"'})]
    assert result['manifest']['entities'][1]['entity_id'] is None
    assert not result['steps'][1]['result']['source_material'][0]['subject_scope'][1]['canonical_identity_verified']


def payload(entities,steps):
    p=runtime_payload();p['original_question']=' '.join(e['mention'] for e in entities)
    p['entities']=entities;p['plan_steps']=steps
    return QueryManifestV2.model_validate(p)


@pytest.mark.parametrize('scopes,expected', [
    ([['a'], ['b']], [('', '"GENEA"'), ('', '"GENEB"')]),
    ([['d', 'a']], [('"Condition A"', '"GENEA"')]),
    ([['a', 'b']], [('', '"GENEA" AND "GENEB"')]),
])
def test_text_search_scope_is_consistent_in_both_source_requests(monkeypatch, scopes, expected):
    import query_clinicaltrials as ct
    from unittest.mock import Mock

    def complete_page(url, *, params, timeout):
        calls.append((url, params))
        raw = {'totalCount': 0} if url == ct.ENDPOINT else {'hitCount': 0, 'resultList': {'result': []}}
        return Mock(json=lambda: raw)

    calls = []
    monkeypatch.setattr(ct.requests, 'get', complete_page)
    plan = payload(
        [entity('d', 'Condition A', 'disease'), entity('a', 'GENEA'), entity('b', 'GENEB')],
        [step(f'safety_{index}', 'retrieve_safety_evidence', refs, ['clinicaltrials_gov', 'europe_pmc'])
         for index, refs in enumerate(scopes)],
    )
    report = execution.PlanExecutor().run(plan)
    assert report['status'] == 'completed', report
    assert len(calls) == 2 * len(expected)
    for index, ((condition, term), refs) in enumerate(zip(expected, scopes)):
        (ct_url, trial), (pmc_url, literature) = calls[index * 2:index * 2 + 2]
        assert ct_url == ct.ENDPOINT and pmc_url != ct_url
        assert trial.get('query.cond', '') == condition
        assert trial['query.term'] == term
        assert 'filter.overallStatus' not in trial
        prefix = ' AND '.join(value for value in (condition, term) if value)
        assert literature['query'] == prefix + ' AND (safety OR toxicity OR "adverse event" OR "adverse events")'
        assert trial['pageSize'] == literature['pageSize'] == 20
        packets = report['steps'][index]['result']['source_material']
        for packet in packets:
            assert [item['entity_ref'] for item in packet['subject_scope']] == refs
            assert packet['search_scope']['entity_operator'] == 'AND'
        assert packets[0]['search_scope']['topic_expression'] is None
        assert packets[1]['search_scope']['topic_expression'].startswith('(safety OR')


@pytest.mark.parametrize('operation,suffix', [
    ('retrieve_publications', ''),
    ('retrieve_negative_publications', ' AND (negative OR failure OR ineffective OR terminated)'),
])
def test_literature_topic_depends_on_operation_not_description(monkeypatch, operation, suffix):
    from unittest.mock import Mock

    search = Mock(return_value={'status': 'not_found', 'records': [], 'total_count': 0, 'truncated': False})
    monkeypatch.setattr(execution, 'search_publications', search)
    p = payload(
        [entity('a', 'GENEA'), entity('d', 'Condition A', 'disease')],
        [{**step('papers', operation, ['a', 'd'], ['europe_pmc']),
          'description': 'Find safety reports and return only ten records.'}],
    )
    assert execution.PlanExecutor().run(p)['status'] == 'completed'
    search.assert_called_once_with(query='"GENEA" AND "Condition A"' + suffix)


def entity(ref,mention,kind='target',id=None):
    return dict(entity_ref=ref,entity_type=kind,mention=mention,entity_id=id,resolution_status='unresolved')


def step(id,op,refs,sources,depends=[]):
    return dict(step_id=id,operation=op,description=op,entity_refs=refs,sources=sources,
                depends_on=depends,outputs=[id+'_output'])


def test_executor_actually_filters_then_queries_only_selected_ids(monkeypatch):
    e=entity('d','disease','disease','D')
    steps=[step('resolve','resolve_disease',['d'],['open_targets']),
           step('discover','discover_targets',['d'],['open_targets'],['resolve']),
           step('filter','filter_targets',['d'],[],['discover']),
           step('tract','retrieve_tractability',[],['open_targets'],['filter'])]
    steps[1]['safeguards']=['association_score_not_probability','association_does_not_imply_efficacy']
    steps[2]['filters']={'all':[gt(.7)]}
    queried=[]
    def gql(query,variables):
        if 'associatedTargets' in query:
            native_rows=[dict(target=dict(id=n,approvedSymbol=n),score=s,datasourceScores=[])
                         for n,s in [('A',.7),('B',.71),('C',.69)]]
            return {'disease':dict(id='D',associatedTargets=dict(count=3,rows=native_rows))}
        if 'tractability' in query:
            queried.append(variables['ensemblId']);return {'target':dict(id=variables['ensemblId'],tractability=[])}
        return {'disease':dict(id='D',name='disease')}
    monkeypatch.setattr(execution.ot,'_graphql',gql)
    result=execution.PlanExecutor().run(payload([e],steps))
    assert result['status']=='completed',result
    assert queried==['B']
    assert len(result['calls'][1]['raw_response']['disease']['associatedTargets']['rows'])==3
    assert result['steps'][2]['result']['targets'][0]['target_id']=='B'


def test_generic_publications_keeps_unresolved_isoform_literal_and_native_response(monkeypatch):
    native=json.loads((Path(__file__).parent/'fixtures/agent1_bindings_native.json').read_text(encoding='utf-8'))['publication_response']
    calls=[]
    def search(**args): calls.append(args);return deepcopy(native)
    monkeypatch.setattr(execution,'search_publications',search)
    result=execution.PlanExecutor().run(payload([entity('iso','CLDN18.2')],
        [step('papers','retrieve_publications',['iso'],['europe_pmc'])]))
    assert result['status']=='completed',result
    assert calls==[{'query':'"CLDN18.2"'}]
    assert result['calls'][0]['raw_response']==native
    scope=result['steps'][0]['result']['source_material'][0]['subject_scope'][0]
    assert scope['literal_mention']=='CLDN18.2' and not scope['canonical_identity_verified']
    assert result['manifest']['entities'][0]['entity_id'] is None
    prep=prepare_execution(result)
    assert prep.query_context['retrieval_subject_scopes'][0]['subject_scope'][0]==scope


def test_content_annotation_is_model_interpretation_not_identity_or_exclusive_source_class(monkeypatch):
    native=dict(status='success',source_url='https://example.org/papers',records=[dict(id='1',source='MED',title='Isoform observation',abstractText='Isoform mutation and expression observations')],total_count=1,truncated=False)
    monkeypatch.setattr(execution,'search_publications',lambda **kwargs:deepcopy(native))
    report=execution.PlanExecutor().run(payload([entity('x','named isoform')],
        [step('papers','retrieve_publications',['x'],['europe_pmc'])]))
    prep=prepare_execution(report);g=prep.groups[0]
    records={r.snapshot_id:r for r in prep.source_records};units=build_catalog(g,records,prep.source_collections,prep.source_run_id)
    key=next(k for k,v in units.items() if v['json_pointer']=='/abstractText')
    selection=ScopedMaterialHandoff(claims=[dict(question_id='Q1',evidence_ids=[key])],review_flags=[],
        content_annotations=[dict(evidence_ids=[key],content_types=['mutation','expression'],subject_scope='unconfirmed',interpretation='Content hypotheses from this source field only')])
    output=assemble_question_selection(selection,g.group_id,prep.research_question,units,records)
    annotation=output['content_annotations'][0]
    assert annotation['content_types']==['mutation','expression'] and annotation['source_channels']==['literature']
    assert annotation['classification_basis']=='organizer_model_interpretation'
    assert not annotation['canonical_isoform_identity_verified']
    selection.content_annotations[0].evidence_ids=['foreign']
    with pytest.raises(ValueError,match='Unknown evidence ID'):
        assemble_question_selection(selection,g.group_id,prep.research_question,units,records)


def test_baseline_native_replay_preserves_units_unknowns_and_separate_groups(monkeypatch):
    native=json.loads((Path(__file__).parent/'fixtures/agent1_bindings_native.json').read_text(encoding='utf-8'))['expression_response']
    rows=deepcopy(native['target']['baselineExpression']['rows'])
    target=native['target']['id']
    source=dict(target=dict(id=target,baselineExpression=dict(count=len(rows),rows=rows)))
    def gql(query,variables):
        if 'baselineExpression' in query:return deepcopy(source)
        return {'target':dict(id=target,approvedSymbol='PTPN11')}
    monkeypatch.setattr(execution.ot,'_graphql',gql)
    p=payload([entity('x','PTPN11',id=target)],
        [step('resolve','resolve_target',['x'],['open_targets']),
         step('expr','retrieve_normal_tissue_expression',['x'],['open_targets'],['resolve'])])
    result=execution.PlanExecutor().run(p)
    assert result['status']=='completed',result
    assert result['calls'][1]['raw_response']==source
    assert result['steps'][1]['result']['expression_collections'][0]['complete_native_collection']
    prep=prepare_execution(result)
    assert len(prep.source_records)==len(rows)
    for group in prep.groups:
        packet=[r.raw_record for r in prep.source_records if r.snapshot_id in group.snapshot_ids]
        assert len({(r['datasourceId'],r['datatypeId'],r['unit']) for r in packet})==1
    assert all(r.raw_record['targetFromSourceId'] is None for r in prep.source_records)
    assert all(c['material_kind']=='normal_tissue_expression' for c in prep.query_context['execution_coverage'])


def test_baseline_wrong_target_or_nonprogress_page_is_not_completed(monkeypatch):
    target='T'
    p=payload([entity('x','target',id=target)],
        [step('resolve','resolve_target',['x'],['open_targets']),
         step('expr','retrieve_normal_tissue_expression',['x'],['open_targets'],['resolve'])])
    for row_target in ['WRONG','T']:
        def gql(query,variables):
            return {'target':dict(id='T',approvedSymbol='target',baselineExpression=dict(count=501,rows=[dict(targetId=row_target,unit=None)]))}
        monkeypatch.setattr(execution.ot,'_graphql',gql)
        result=execution.PlanExecutor().run(p)
        assert result['status']=='incomplete'
        assert 'mismatch' in result['steps'][-1]['error'] or 'no progress' in result['steps'][-1]['error']


def test_baseline_uses_row_offset_not_page_number(monkeypatch):
    offsets=[]
    def gql(query,variables):
        if 'baselineExpression' not in query:return {'target':dict(id='T',approvedSymbol='target')}
        offset=variables['rowOffset'];offsets.append(offset)
        rows=[dict(targetId='T',unit='native',tissueBiosampleFromSource=str(i)) for i in range(offset,min(offset+500,1001))]
        return {'target':dict(id='T',baselineExpression=dict(count=1001,rows=rows))}
    monkeypatch.setattr(execution.ot,'_graphql',gql)
    p=payload([entity('x','target',id='T')],[step('resolve','resolve_target',['x'],['open_targets']),
         step('expr','retrieve_normal_tissue_expression',['x'],['open_targets'],['resolve'])])
    result=execution.PlanExecutor().run(p)
    assert result['status']=='completed' and offsets==[0,500,1000]
    assert result['steps'][-1]['result']['expression_collections'][0]['returned_count']==1001
