import pytest
from pydantic import ValidationError

from schemas import DataSourceEvidence, EvidenceInterpretation


def test_datasource_score_must_be_bounded():
    with pytest.raises(ValidationError):
        DataSourceEvidence(datasource_id="example", score=1.2)


def test_interpretation_keeps_confidence_pending():
    result = EvidenceInterpretation(
        disease="PDAC",
        target="KRAS",
        summary="Evidence is present.",
        supporting_evidence=["Open Targets association evidence."],
        weak_or_missing_evidence=["No team-defined confidence framework."],
        data_limitations=["Research-use dataset."],
    )
    assert result.confidence_score is None
    assert result.confidence_score_status == "pending_research_team_definition"
