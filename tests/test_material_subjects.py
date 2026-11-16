import json
from pathlib import Path

import pytest

from evidence_preparation import EvidencePreparation, selection_input
from evidence_selection import ScopedMaterialHandoff, assemble_question_selection
from evidence_verification import verify_evidence_result
from test_nim_evidence import fixture_response


@pytest.mark.parametrize('repetition', [1, 2, 3])
def test_real_empty_gap_links_bind_to_question_study_without_rewriting_model(repetition):
    root = Path(__file__).resolve().parents[1] / 'tests/fixtures/replay/'
    prep = EvidencePreparation.model_validate_json((root / 'references.preparation.json').read_text('utf8'))
    old = json.loads((root / f'references_{repetition}.json').read_text('utf8'))
    group = next(g for g in prep.groups if g.group_id == old['group_id'])
    prepared, units = selection_input(prep, group, old['research_question'])
    assert list(json.loads(prepared.messages[1]['content'])['subjects']) == ['NCT01767311']
    selection = ScopedMaterialHandoff.model_validate_json(old['response_text'])
    assert selection.review_flags[0].evidence_ids == []
    result = assemble_question_selection(selection, group.group_id, old['research_question'], units,
                                         {r.snapshot_id:r for r in prep.source_records})
    assert result['review_flags'][0]['subject_ref'] == 'NCT01767311'
    assert result['review_flags'][0]['evidence_ids'] == ['E29']
    assert selection.review_flags[0].evidence_ids == []
    old.update(output_contract='source_material_v2', structured_output=result)
    report = verify_evidence_result(prep, old)
    assert all(c['status'] == 'passed' for c in report['automated_checks'])


def assemble_gap(question, subject=None, ids=None):
    prep, _ = fixture_response()
    record = next(r for r in prep.source_records if r.source == 'open_targets')
    units = {key: dict(reference_scope='raw_record', json_pointer='/clinicalReportId',
                       field_state='populated', value=nct.lower(), snapshot_ids=[record.snapshot_id])
             for key,nct in [('E1','NCT00000001'),('E2','NCT00000002')]}
    selection = ScopedMaterialHandoff.model_validate({'claims': [], 'review_flags': [{
        'question_id':'Q1', 'evidence_ids':ids or [], 'subject_ref':subject,
        'missing_material':'尚未取得的結果表'}]})
    return assemble_question_selection(selection, prep.groups[0].group_id, question, units,
                                       {record.snapshot_id:record})


def test_multiple_studies_require_explicit_subject():
    with pytest.raises(ValueError, match='Multiple study subjects'):
        assemble_gap('比較 NCT00000001 與 NCT00000002')
    result = assemble_gap('比較 NCT00000001 與 NCT00000002', 'NCT00000002')
    assert result['review_flags'][0]['evidence_ids'] == ['E2']


def test_unknown_or_conflicting_subject_is_rejected():
    with pytest.raises(ValueError, match='Unknown subject_ref'):
        assemble_gap('NCT00000001 結果', 'NCT00000002')
    with pytest.raises(ValueError, match='conflicts'):
        assemble_gap('NCT00000001 結果', ids=['E2'])


def test_requested_study_without_retrieved_locator_does_not_inherit_another_study():
    result = assemble_gap('NCT99999999 結果')
    assert result['review_flags'][0]['subject_ref'] == 'NCT99999999'
    assert result['review_flags'][0]['evidence_ids'] == []
