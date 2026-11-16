from copy import deepcopy
import json
from pathlib import Path

import pytest

import plan_execution as execution
from execution_organization import prepare_execution, verify_execution_preparation
from human_entity_context import HumanResearchClarification
from natural_language_planner import NaturalLanguageResearchPlanner
from target_evidence_inventory import (
    filter_inventory, target_inventory, pocket_assessments, rank_expression_material,
)
from test_agent1_new_bindings import payload, entity, step, row, gt
from test_natural_language_planner import FakeAgent


POCKET='predicted_small_molecule_pocket'
PREDICATE=dict(field=POCKET,operator='is_true',value=True)


def packet(target,high=False,medium=False,structure=False):
    return dict(target_id=target,source_ref='tract-'+target,pointer='/target/tractability',records=[
        dict(label=label,modality='SM',value=value) for label,value in [
            ('High-Quality Pocket',high),('Med-Quality Pocket',medium),('Structure with Ligand',structure)]])


@pytest.mark.parametrize('high,medium,expected',[(True,False,True),(False,True,True),(False,False,False),
    (None,False,None),(None,None,None),(None,True,True)])
def test_native_predicted_pocket_tristate_and_separate_structure(high,medium,expected):
    result=pocket_assessments([packet('A',high,medium,True)])
    assert result[POCKET] is expected
    assert result['experimental_structure_with_ligand'] is True
    assert result['pocket_flag_evidence']['High-Quality Pocket'][0]['pointer'].endswith('/0/value')


def test_unrelated_modality_drug_and_other_target_cannot_qualify():
    a=packet('A');a['records'] += [dict(label='High-Quality Pocket',modality='AB',value=True),
                                 dict(label='Approved Drug',modality='SM',value=True)]
    upstream=[dict(targets=[row('A',.71),row('B',.7),row('C',.9,'OTHER')]),
              dict(target_material=[a,packet('B',True),packet('C',True)])]
    rows=target_inventory(upstream)
    assert rows[0][POCKET] is False
    filtered=filter_inventory(rows[:2],{'all':[gt(.7),PREDICATE]})
    assert filtered['targets']==[] and filtered['unknown_targets']==[]
    # This is a valid empty selection; unrelated disease/target cannot leak back in.
    result=target_inventory([*upstream,filtered])
    assert [(r['disease_id'],r['target_id']) for r in result]==[('OTHER','C')]


def test_null_false_zero_threshold_and_provenance_are_not_conflated():
    rows=[row('A',.71,**{POCKET:True}),row('B',.7,**{POCKET:True}),
          row('C',.9,**{POCKET:None}),row('D',.9,**{POCKET:False})]
    result=filter_inventory(rows,{'all':[gt(.7),PREDICATE]})
    assert result['targets']==[rows[0]] and result['unknown_targets']==[rows[2]]
    with pytest.raises(ValueError):filter_inventory([row('A',.9,**{POCKET:1})],{'all':[PREDICATE]})
    with pytest.raises(ValueError):filter_inventory(rows,{'all':[{**PREDICATE,'operator':'gt'}]})


def test_pocket_metric_needs_tractability_ancestor_and_matching_entity():
    p=payload([entity('d','condition','disease','D')],[step('r','resolve_disease',['d'],['open_targets']),
        step('discover','discover_targets',['d'],['open_targets'],['r']),
        step('filter','filter_targets',['d'],[],['discover'])])
    p.plan_steps[-1].filters={'all':[PREDICATE]}
    assert any(POCKET in error for error in execution.execution_issues(p))


def material(tissue,median,**changes):
    raw=dict(targetId='T',targetFromSourceId=None,datasourceId='gtex',datatypeId='bulk rna-seq',unit='TPM',
             tissueBiosample=dict(biosampleId=tissue,biosampleName=tissue),tissueBiosampleFromSource=tissue,
             celltypeBiosample=None,celltypeBiosampleFromSource=None,qualityControls='[]',median=median,
             specificity_score=999)
    raw.update(changes)
    return dict(record=raw,source_ref='native-call',pointer='/native/'+tissue)


def test_relative_median_groups_ties_zero_unknown_and_native_values():
    rows=[material('B',2),material('A',2),material('Z',0),material('M',None),material('N',float('nan')),
          material('PROTEIN',999,datatypeId='proteomics',unit='PPB'),
          material('OTHER',100,targetId='OTHER'),material('CELL',100,celltypeBiosampleFromSource='C'),
          material('QC',100,qualityControls='["flag"]'),material('UNIT',100,unit=None)]
    result=rank_expression_material(rows,'descending')
    assert len(result['groups'])==6
    group=next(g for g in result['groups'] if len(g['rows'])==5)
    assert [r['tissue_from_source'] for r in group['rows']]==['A','B','Z','M','N']
    assert [r['source_median'] for r in group['rows']]==[2,2,0,None,None]
    assert [r['position'] for r in group['rows']]==[1,2,3,None,None]
    assert next(g for g in result['groups'] if not g['comparable'])['rows'][0]['source_median'] is None
    assert rows[0]['record']['median']==2


def test_expression_ranking_contract_and_executor_native_replay(monkeypatch):
    fixture=json.loads((Path(__file__).parent/'fixtures/agent1_bindings_native.json').read_text(encoding='utf-8'))
    raw=fixture['expression_response']['target']
    target=raw['id'];native_rows=deepcopy(raw['baselineExpression']['rows'])
    p=payload([entity('x','target',id=target)],[step('resolve','resolve_target',['x'],['open_targets']),
        step('expr','retrieve_normal_tissue_expression',['x'],['open_targets'],['resolve'])])
    p.plan_steps[-1].ranking=dict(primary_metric='expression_median',direction='descending',secondary_metrics=[])
    # Assignment does not run Pydantic parsing, so parse the complete manifest.
    p=type(p).model_validate(p.model_dump())
    def gql(query,variables):
        if 'baselineExpression' in query:
            return dict(target=dict(id=target,baselineExpression=dict(count=len(native_rows),rows=deepcopy(native_rows))))
        return dict(target=dict(id=target,approvedSymbol='target'))
    monkeypatch.setattr(execution.ot,'_graphql',gql)
    assert execution.execution_issues(p)==[]
    c=HumanResearchClarification(original_question=p.original_question,confirmation_question='Use relative source medians?',
        user_response='Relative ranking separately by measurement',run_id='run-one')
    result=execution.PlanExecutor().run(p,run_id='run-one',research_clarification=c)
    assert result['status']=='completed'
    ranking=result['steps'][-1]['result']['expression_ranking']
    assert sum(len(g['rows']) for g in ranking['groups'])==len(native_rows)
    for group in ranking['groups']:
        values=[r['source_median'] for r in group['rows'] if r['source_median'] is not None]
        assert values==sorted(values,reverse=True)
    prep=prepare_execution(result)
    assert verify_execution_preparation(prep,result)['status']=='passed'
    assert prep.query_context['relative_expression_rankings'][0]['result']==ranking
    assert prep.query_context['research_clarification']['run_id']=='run-one'
    record_map={r.snapshot_id:r for r in prep.source_records}
    for group,model_input in zip(prep.groups,prep.model_inputs):
        from evidence_packet import unpack_input
        data=unpack_input(json.loads(model_input.messages[1]['content']))
        origins={(record_map[sid].execution_origin['call_id'],record_map[sid].execution_origin['json_pointer']) for sid in group.snapshot_ids}
        for entry in data['query_context']['relative_expression_rankings']:
            assert all((r['source_ref'],r['pointer']) in origins for g in entry['result']['groups'] for r in g['rows'])
    forged=deepcopy(result);forged['research_clarification']['run_id']='foreign'
    with pytest.raises(ValueError,match='different run'):prepare_execution(forged)


def test_research_clarification_is_question_run_scoped_and_visible_without_rewriting_question():
    p=payload([entity('t','target',id='T')],[step('r','resolve_target',['t'],['open_targets'])])
    c=HumanResearchClarification(original_question=p.original_question,confirmation_question='Use relative source medians?',
        user_response='Relative ranking separately by measurement',run_id='run-one')
    fake=FakeAgent([dict(structured_response=p.model_dump(mode='json'))])
    report=NaturalLanguageResearchPlanner(agent=fake,execution_mode=True).run_with_report(
        p.original_question,max_attempts=1,research_clarification=c)
    assert report.status=='passed' and not report.blind_first_attempt
    assert report.final_plan.original_question==p.original_question
    assert c.user_response in json.dumps(fake.calls)
    with pytest.raises(ValueError,match='different question'):c.for_run('OTHER','run-one')
    with pytest.raises(ValueError,match='different run'):c.for_run(p.original_question,'run-two')
