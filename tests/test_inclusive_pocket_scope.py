from copy import deepcopy

import pytest

from target_evidence_inventory import pocket_assessments,filter_inventory,target_inventory
from test_clarified_pocket_expression import packet
from test_agent1_new_bindings import row,gt,payload,entity,step
from plan_execution import execution_issues

FIELD='small_molecule_pocket_evidence'
PREDICATE=dict(field=FIELD,operator='is_true',value=True)


@pytest.mark.parametrize('predicted,experimental',[(p,e) for p in [True,False,None] for e in [True,False,None]])
def test_inclusive_truth_table_keeps_native_evidence_classes_separate(predicted,experimental):
    result=pocket_assessments([packet('A',predicted,False,experimental)])
    expected=True if predicted is True or experimental is True else False if predicted is False and experimental is False else None
    assert result[FIELD] is expected
    assert result['predicted_small_molecule_pocket'] is predicted
    assert result['experimental_structure_with_ligand'] is experimental
    filtered=filter_inventory([row('A',.70001,**result)],{'all':[gt(.7),PREDICATE]})
    assert len(filtered['targets'])==(1 if expected is True else 0)
    assert len(filtered['unknown_targets'])==(1 if expected is None else 0)


def test_inclusive_does_not_change_strict_association_boundary_or_cross_target_scope():
    rows=[row('EXPERIMENT',.71),row('PREDICTED',.7),row('NONE',.9),row('UNKNOWN',.9),row('FOREIGN',.99,'OTHER')]
    packets=[packet('EXPERIMENT',False,False,True),packet('PREDICTED',True),
             packet('NONE',False,False,False),packet('UNKNOWN',None,False,False),packet('FOREIGN',True)]
    upstream=[dict(targets=rows),dict(target_material=packets)]
    original=deepcopy(upstream)
    inventory=target_inventory(upstream)
    filtered=filter_inventory(inventory[:4],{'all':[gt(.7),PREDICATE]})
    assert [r['target_id'] for r in filtered['targets']]==['EXPERIMENT']
    assert [r['target_id'] for r in filtered['unknown_targets']]==['UNKNOWN']
    assert upstream==original
    selected=filtered['targets'][0]
    assert selected['predicted_pocket_flags']=={'High-Quality Pocket':False,'Med-Quality Pocket':False}
    assert selected['pocket_flag_evidence']['Structure with Ligand'][0]['source_ref']=='tract-EXPERIMENT'


def test_approved_drug_or_wrong_modality_does_not_become_pocket_evidence():
    p=packet('A');p['records'] += [dict(label='Approved Drug',modality='SM',value=True),
                                 dict(label='Structure with Ligand',modality='AB',value=True)]
    assert pocket_assessments([p])[FIELD] is False
    assert pocket_assessments([])[FIELD] is None


def test_inclusive_metric_requires_tractability_before_execution():
    p=payload([entity('d','condition','disease','D')],[step('r','resolve_disease',['d'],['open_targets']),
        step('discover','discover_targets',['d'],['open_targets'],['r']),
        step('filter','filter_targets',['d'],[],['discover'])])
    p.plan_steps[-1].filters={'all':[PREDICATE]}
    assert any(FIELD in issue for issue in execution_issues(p))
