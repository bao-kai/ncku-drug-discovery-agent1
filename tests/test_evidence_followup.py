"""Offline checks for the exact-ID supplementation case, never live APIs."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
import requests

from evidence_followup import (
    EvidenceFollowupCase, STATUS_PATH, STOP_REASON_PATH, UPDATED_PATH,
    apply_trial_lookup, build_followup_case, execute_trial_followup,
)
from query_clinicaltrials import ClinicalTrialLookup, get_clinical_trial
from schemas import Agent1RunOutput


def saved_run(reason="Sponsor's Decision"):
    return Agent1RunOutput.model_validate({
        "producer": "agent1", "run_id": "db81c97d-967a-4014-99aa-7a5c749d766b",
        "generated_at": "2026-08-06T14:01:29Z",
        "agent_1_evidence": {
            "disease_query": "Alzheimer", "target_query": "APP",
            "disease_id": "MONDO_0004975", "disease_name": "Alzheimer disease",
            "target_id": "ENSG00000142192", "target_symbol": "APP", "target_name": "APP",
            "fetched_target_count": 1,
            "datasource_evidence": [{
                "datasource_id": "clinical_precedence", "total_evidence_count": 1,
                "fetched_evidence_count": 1, "truncated": False,
                "records": [{"id": "E1", "clinicalReportId": "nct04241068",
                             "drugFromSource": "aducanumab", "clinicalStage": "PHASE_3",
                             "trialWhyStopped": reason}],
            }],
        },
    })


def lookup(status="success", overall_status="TERMINATED", why_stopped="Sponsor's Decision"):
    return ClinicalTrialLookup(
        record_id="NCT04241068", source_url="https://clinicaltrials.gov/api/v2/studies/NCT04241068",
        retrieved_at=datetime.now(timezone.utc), status=status,
        raw_record={"protocolSection": {
            "identificationModule": {"nctId": "NCT04241068"},
            "statusModule": {
                "overallStatus": overall_status, "whyStopped": why_stopped,
                "lastUpdatePostDateStruct": {"date": "2025-02-21"},
            },
        }} if status == "success" else None,
        error="ConnectionError" if status == "error" else None,
    )


@patch("query_clinicaltrials.requests.get")
def test_lookup_normalizes_id_preserves_raw_payload_and_timeout(get):
    raw = lookup().raw_record
    get.return_value = Mock(status_code=200, json=lambda: raw)
    result = get_clinical_trial(" nct04241068 ")
    get.assert_called_once_with(
        "https://clinicaltrials.gov/api/v2/studies/NCT04241068", timeout=30,
    )
    assert result.status == "success"
    assert result.raw_record == raw


@patch("query_clinicaltrials.requests.get")
def test_invalid_id_never_starts_retrieval(get):
    with pytest.raises(ValueError, match="Expected a ClinicalTrials"):
        get_clinical_trial("NCT04241068/other")
    get.assert_not_called()


@patch("query_clinicaltrials.requests.get")
def test_not_found_is_not_a_negative_scientific_claim(get):
    get.return_value = Mock(status_code=404)
    result = get_clinical_trial("NCT04241068")
    assert result.status == "not_found"
    assert result.raw_record is None


@patch("query_clinicaltrials.requests.get", side_effect=requests.Timeout("timed out"))
def test_network_error_is_not_empty_evidence(_get):
    result = get_clinical_trial("NCT04241068")
    assert result.status == "error"
    assert "Timeout" in result.error


@patch("query_clinicaltrials.requests.get")
def test_wrong_study_id_is_rejected(get):
    raw = lookup().raw_record
    raw["protocolSection"]["identificationModule"]["nctId"] = "NCT00000001"
    get.return_value = Mock(status_code=200, json=lambda: raw)
    result = get_clinical_trial("NCT04241068")
    assert result.status == "error"
    assert result.raw_record is None


def test_blank_stop_reason_does_not_generate_terminated_or_safety_claims():
    case = build_followup_case(saved_run(reason=None), "E1")
    assert {f.predicate for f in case.claims} == {
        "drugFromSource", "clinicalReportId", "clinicalStage",
    }
    assert case.review_flags[0].field_name == STATUS_PATH
    assert case.supplemental_requests[0].record_id == "NCT04241068"


def test_success_keeps_original_and_links_new_facts_and_flag_resolution():
    run = saved_run()
    original = run.model_dump()
    base = build_followup_case(run, "E1")
    before = base.model_dump()
    result = apply_trial_lookup(base, base.supplemental_requests[0].request_id, lookup())
    assert run.model_dump() == original
    assert base.model_dump() == before
    assert result.source_records[0] == base.source_records[0]
    assert len(result.source_records) == 2
    assert result.supplemental_requests[0].status == "completed"
    facts = {f.predicate: f for f in result.claims}
    assert facts[STATUS_PATH].value == "TERMINATED"
    assert facts[STOP_REASON_PATH].value == "Sponsor's Decision"
    assert facts[UPDATED_PATH].value == "2025-02-21"
    assert facts[STATUS_PATH].evidence_refs[0].snapshot_id == result.source_records[1].snapshot_id
    assert result.review_flags[0].resolved_by_snapshot_id == result.source_records[1].snapshot_id


@pytest.mark.parametrize("status", ["not_found", "error"])
def test_failed_followup_preserves_gap_and_adds_no_facts(status):
    base = build_followup_case(saved_run(), "E1")
    result = apply_trial_lookup(base, base.supplemental_requests[0].request_id, lookup(status=status))
    assert result.claims == base.claims
    assert result.source_records == base.source_records
    assert result.review_flags[0].resolved_by_snapshot_id is None
    assert result.supplemental_requests[0].status == status


def test_partial_result_keeps_missing_fields_explicit():
    base = build_followup_case(saved_run(), "E1")
    result = apply_trial_lookup(
        base, base.supplemental_requests[0].request_id,
        lookup(overall_status=None, why_stopped=None),
    )
    assert result.supplemental_requests[0].status == "partial"
    assert result.supplemental_requests[0].missing_fields == [STATUS_PATH, STOP_REASON_PATH]
    assert result.review_flags[0].resolved_by_snapshot_id is None


def test_fabricated_fact_value_fails_source_validation():
    case = build_followup_case(saved_run(), "E1")
    payload = case.model_dump()
    payload["claims"][0]["value"] = "invented drug"
    with pytest.raises(ValueError, match="differs from the referenced raw field"):
        EvidenceFollowupCase.model_validate(payload)


def test_source_field_cannot_be_relabeled_as_a_scientific_conclusion():
    payload = build_followup_case(saved_run(), "E1").model_dump()
    payload["claims"][0]["predicate"] = "proves_efficacy"
    with pytest.raises(ValueError, match="predicate must identify"):
        EvidenceFollowupCase.model_validate(payload)


def test_existing_status_does_not_create_a_false_missing_status_flag():
    run = saved_run()
    run.agent_1_evidence.datasource_evidence[0].records[0]["overallStatus"] = "COMPLETED"
    with pytest.raises(ValueError, match="already contains"):
        build_followup_case(run, "E1")


def test_duplicate_execution_does_not_call_the_network():
    base = build_followup_case(saved_run(), "E1")
    request_id = base.supplemental_requests[0].request_id
    result = apply_trial_lookup(base, request_id, lookup())
    with patch("evidence_followup.get_clinical_trial") as get:
        with pytest.raises(ValueError, match="pending"):
            execute_trial_followup(result, request_id)
        get.assert_not_called()
