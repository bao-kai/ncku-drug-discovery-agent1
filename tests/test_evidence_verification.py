"""Offline regressions for source verification and explicit wording review."""

import json
from copy import deepcopy

import pytest

from evidence_followup import apply_trial_lookup, build_followup_case
from evidence_preparation import prepare_evidence
from evidence_verification import SourceSupportReview, verify_evidence_result
from test_evidence_followup import lookup, saved_run


def prepared_result():
    run = saved_run()
    row = run.agent_1_evidence.datasource_evidence[0].records[0]
    row["diseaseFromSourceMappedId"] = run.agent_1_evidence.disease_id
    row["targetFromSourceId"] = run.agent_1_evidence.target_id
    prepared = prepare_evidence(run)
    group = prepared.groups[0]
    payload = {
        "group_id": group.group_id,
        "claims": [{"subject": "NCT04241068", "predicate": "reported_stop_reason",
                    "statement": "來源記載 Sponsor's Decision。", "qualifications": [],
                    "evidence_refs": [{"snapshot_id": group.snapshot_ids[0],
                                       "json_pointer": "/trialWhyStopped"}]}],
        "review_flags": [],
    }
    result = {"source_run_id": prepared.source_run_id, "group_id": group.group_id,
              "finish_reason": "stop", "response_text": json.dumps(payload),
              "status": "references_validated"}
    return prepared, payload, result


def test_valid_references_without_wording_review_remain_pending():
    prepared, _, result = prepared_result()
    report = verify_evidence_result(prepared, result)
    assert report["status"] == "needs_review"
    item = report["item_reviews"][0]
    assert item["reference_status"] == "passed"
    assert item["source_support"] == "needs_review"
    assert item["source_values"][0]["raw_value"] == "Sponsor's Decision"


def test_explicit_review_preserves_wording_source_values_and_original_inputs():
    prepared, _, result = prepared_result()
    before = deepcopy(result)
    report = verify_evidence_result(prepared, result, reviews={
        "/claims/0": SourceSupportReview(decision="supported", reason="Exact source wording is retained."),
    }, reviewer="offline_test_reviewer")
    assert result == before
    assert report["status"] == "source_reviewed_within_scope"
    assert report["reviewer"] == "offline_test_reviewer"
    assert report["scope"].endswith("not_biomedical_truth")


def test_existing_reference_does_not_make_an_unsupported_conclusion_pass():
    prepared, payload, result = prepared_result()
    payload["claims"][0]["statement"] = "試驗因安全性問題終止。"
    result["response_text"] = json.dumps(payload)
    report = verify_evidence_result(prepared, result, reviews={
        "/claims/0": SourceSupportReview(
            decision="unsupported", reason="Sponsor's Decision does not specify a safety cause."),
    }, reviewer="offline_test_reviewer")
    item = report["item_reviews"][0]
    assert item["reference_status"] == "passed"
    assert item["source_support"] == "unsupported"
    assert report["status"] == "failed"


@pytest.mark.parametrize("defect", ["unknown_snapshot", "missing_field"])
def test_bad_reference_reports_exact_item_and_prevents_supported_promotion(defect):
    prepared, payload, result = prepared_result()
    ref = payload["claims"][0]["evidence_refs"][0]
    if defect == "unknown_snapshot":
        ref["snapshot_id"] = "missing"
    else:
        ref["json_pointer"] = "/missing"
    result["response_text"] = json.dumps(payload)
    report = verify_evidence_result(prepared, result, reviews={
        "/claims/0": SourceSupportReview(decision="supported", reason="Erroneous review must not override broken links."),
    }, reviewer="offline_test_reviewer")
    assert report["status"] == "failed"
    item = report["item_reviews"][0]
    assert item["item_path"] == "/claims/0"
    assert item["source_support"] == "unsupported"
    assert item["source_values"][0]["error"]


def test_source_count_and_expanded_query_warning_remain_visible():
    prepared, _, result = prepared_result()
    prepared.source_collections[0]["collection_warnings"] = ["Expanded fields rejected; scalar fallback used"]
    report = verify_evidence_result(prepared, result)
    check = report["collection_checks"][0]
    assert check["count_consistent"] is True
    assert check["status"] == "needs_review"
    assert "scalar fallback" in check["collection_warnings"][0]
    prepared.source_collections[0]["fetched_evidence_count"] = 2
    assert verify_evidence_result(prepared, result)["status"] == "failed"


def test_verifier_recomputes_statistics_instead_of_trusting_a_cached_inventory():
    prepared, _, result = prepared_result()
    prepared.groups[0].scalar_value_counts["clinicalStage"][0]["count"] = 9
    report = verify_evidence_result(prepared, result)
    assert report["status"] == "failed"
    assert report["group_inventory"]["scalar_value_counts"]["clinicalStage"] == [
        {"value": "PHASE_3", "count": 1}]


def test_wrong_entity_fails_and_missing_entity_is_not_silently_verified():
    prepared, _, result = prepared_result()
    prepared.source_records[0].raw_record["targetFromSourceId"] = "WRONG_TARGET"
    report = verify_evidence_result(prepared, result)
    assert report["source_checks"][0]["status"] == "failed"
    assert report["status"] == "failed"
    prepared.source_records[0].raw_record.pop("targetFromSourceId")
    report = verify_evidence_result(prepared, result)
    assert report["source_checks"][0]["status"] == "needs_review"


def test_wrong_run_incomplete_response_and_invalid_schema_do_not_pass():
    prepared, _, result = prepared_result()
    for change in ({"source_run_id": "other"}, {"finish_reason": "length"},
                   {"response_text": "not JSON"}):
        report = verify_evidence_result(prepared, {**result, **change})
        assert report["status"] == "failed"


def test_review_flags_need_semantic_review_too_and_review_paths_are_checked():
    prepared, payload, result = prepared_result()
    payload["review_flags"] = [{
        "description": "缺少詳細停止原因", "missing_information": ["specific stop cause"],
        "evidence_refs": payload["claims"][0]["evidence_refs"],
    }]
    result["response_text"] = json.dumps(payload)
    reviews = {"/claims/0": SourceSupportReview(decision="supported", reason="Source wording copied.")}
    report = verify_evidence_result(prepared, result, reviews=reviews, reviewer="offline_test_reviewer")
    assert report["item_reviews"][1]["item_path"] == "/review_flags/0"
    assert report["item_reviews"][1]["source_support"] == "needs_review"
    assert report["status"] == "needs_review"
    with pytest.raises(ValueError, match="named reviewer"):
        verify_evidence_result(prepared, result, reviews=reviews)
    reviews["/claims/99"] = reviews["/claims/0"]
    with pytest.raises(ValueError, match="item absent"):
        verify_evidence_result(prepared, result, reviews=reviews, reviewer="offline_test_reviewer")


def test_supplement_remains_distinct_and_resolves_collection_gap_not_specific_cause():
    run = saved_run()
    row = run.agent_1_evidence.datasource_evidence[0].records[0]
    row.update(diseaseFromSourceMappedId=run.agent_1_evidence.disease_id,
               targetFromSourceId=run.agent_1_evidence.target_id)
    base = build_followup_case(run, "E1")
    case = apply_trial_lookup(base, base.supplemental_requests[0].request_id, lookup())
    prepared = prepare_evidence(run, case)
    group = next(g for g in prepared.groups if g.source == "clinicaltrials_gov")
    snapshot_id = group.snapshot_ids[0]
    payload = {"group_id": group.group_id, "claims": [], "review_flags": [{
        "description": "已取得狀態，但缺少具體停止原因。", "missing_information": ["specific stop cause"],
        "evidence_refs": [{"snapshot_id": snapshot_id,
                           "json_pointer": "/protocolSection/statusModule/whyStopped"}],
    }]}
    result = {"source_run_id": prepared.source_run_id, "group_id": group.group_id,
              "finish_reason": "stop", "response_text": json.dumps(payload)}
    report = verify_evidence_result(prepared, result, reviews={
        "/review_flags/0": SourceSupportReview(decision="supported", reason="Returned stop text remains nonspecific."),
    }, reviewer="offline_test_reviewer")
    assert report["status"] == "source_reviewed_within_scope"
    assert {s["source"] for s in report["source_checks"]} == {"open_targets", "clinicaltrials_gov"}
    assert report["item_reviews"][0]["source_values"][0]["raw_value"] == "Sponsor's Decision"
