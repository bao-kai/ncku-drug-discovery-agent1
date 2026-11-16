"""Checker contracts and isolation tests with local simulated responses."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from evidence_preparation import ModelEvidenceReference, resolve_evidence_reference, validate_organized_group
from evidence_semantic_checker import (
    build_checker_messages, run_semantic_check, validate_semantic_review,
    verification_with_model_review,
)
from evidence_verification import verify_evidence_result
from test_evidence_verification import prepared_result
from test_nim_evidence import FakeStream, chunk


def checker_response(prepared, candidate):
    return {"group_id": prepared.groups[0].group_id, "reviews": [{
        "item_path": "/claims/0", "decision": "supported", "reason": "與來源原文相符。",
        "evidence_refs": candidate["claims"][0]["evidence_refs"],
    }]}


def test_metadata_refs_resolve_exact_warning_and_native_null_remains_distinct():
    prepared, candidate, result = prepared_result()
    prepared.source_collections[0]["collection_warnings"] = ["Scalar fallback was used"]
    snapshot_id = prepared.groups[0].snapshot_ids[0]
    prepared.source_records[0].raw_record["warningMessage"] = None
    ref = ModelEvidenceReference(snapshot_id=snapshot_id, reference_scope="collection_metadata",
                                 json_pointer="/collection_warnings/0")
    assert resolve_evidence_reference(prepared, ref) == "Scalar fallback was used"
    assert resolve_evidence_reference(prepared, ModelEvidenceReference(
        snapshot_id=snapshot_id, json_pointer="/warningMessage")) is None
    candidate["claims"][0]["evidence_refs"] = [ref.model_dump()]
    assert validate_organized_group(candidate, prepared, candidate["group_id"])
    result["response_text"] = json.dumps(candidate)
    report = verify_evidence_result(prepared, result)
    assert report["item_reviews"][0]["source_values"][0]["raw_value"] == "Scalar fallback was used"
    assert report["item_reviews"][0]["reference_status"] == "passed"
    semantic = checker_response(prepared, candidate)
    semantic["reviews"][0]["evidence_refs"] = [ref.model_dump()]
    assert validate_semantic_review(semantic, prepared, result)
    prepared.source_collections.append(deepcopy(prepared.source_collections[0]))
    with pytest.raises(ValueError, match="one matching run/datasource"):
        resolve_evidence_reference(prepared, ref)


def test_wrong_run_cannot_anchor_collection_metadata():
    prepared, _, _ = prepared_result()
    source = prepared.source_records[0]
    source.source_run_id = "different-run"
    with pytest.raises(ValueError, match="one matching run/datasource"):
        resolve_evidence_reference(prepared, ModelEvidenceReference(
            snapshot_id=source.snapshot_id, reference_scope="collection_metadata", json_pointer=""))


def test_checker_receives_source_context_without_organizer_reasoning_or_review_answers():
    prepared, _, result = prepared_result()
    result["reasoning_text"] = "ORGANIZER_REASONING_MARKER"
    result["human_review"] = "REFERENCE_ANSWER_MARKER"
    before = deepcopy(result)
    messages = build_checker_messages(prepared, result)
    serialized = json.dumps(messages)
    assert "ORGANIZER_REASONING_MARKER" not in serialized
    assert "REFERENCE_ANSWER_MARKER" not in serialized
    data = json.loads(messages[1]["content"])
    assert data["required_item_paths"] == ["/claims/0"]
    assert data["source_records"][0]["raw_record"] == prepared.source_records[0].raw_record
    assert data["collection_metadata"][0]["document"] == prepared.source_collections[0]
    assert result == before


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "wrong_group", "bad_ref", "empty_reason"])
def test_incomplete_or_fabricated_reviews_are_rejected(defect):
    prepared, candidate, result = prepared_result()
    payload = checker_response(prepared, candidate)
    if defect == "missing":
        payload["reviews"] = []
    elif defect == "duplicate":
        payload["reviews"] *= 2
    elif defect == "extra":
        payload["reviews"][0]["item_path"] = "/claims/99"
    elif defect == "wrong_group":
        payload["group_id"] = "other"
    elif defect == "bad_ref":
        payload["reviews"][0]["evidence_refs"][0]["snapshot_id"] = "not-supplied"
    else:
        payload["reviews"][0]["reason"] = " "
    with pytest.raises(ValueError):
        validate_semantic_review(payload, prepared, result)


def test_model_supported_does_not_approve_items_without_source_comparison():
    prepared, candidate, result = prepared_result()
    client = Mock()
    client.chat.completions.create.return_value = FakeStream([
        chunk(reasoning="checker-only reasoning"),
        chunk(json.dumps(checker_response(prepared, candidate))), chunk(finish="stop"),
    ])
    semantic = run_semantic_check(prepared, result, client)
    assert semantic["status"] == "review_validated"
    assert semantic["reasoning_text"] == "checker-only reasoning"
    assert client.chat.completions.create.call_count == 1
    report = verification_with_model_review(prepared, result, semantic)
    assert report["item_reviews"][0]["source_support"] == "supported"
    assert report["item_reviews"][0]["model_review_evidence"][0]["value"] == "Sponsor's Decision"
    assert report["status"] == "needs_review"
    assert report["source_comparison_pending"] is True
    assert report["review_method"] == "model_initial"
    wrong_candidate = {**result, "response_text": result["response_text"] + " "}
    with pytest.raises(ValueError, match="not bound"):
        verification_with_model_review(prepared, wrong_candidate, semantic)


@pytest.mark.parametrize("failure", ["network", "length", "invalid"])
def test_checker_failure_keeps_source_support_pending_without_retry(failure):
    prepared, candidate, result = prepared_result()
    client = Mock()
    if failure == "network":
        client.chat.completions.create.side_effect = RuntimeError("fake-network-error")
    else:
        text = "not JSON" if failure == "invalid" else json.dumps(checker_response(prepared, candidate))
        client.chat.completions.create.return_value = FakeStream([
            chunk(text), chunk(finish="length" if failure == "length" else "stop"),
        ])
    semantic = run_semantic_check(prepared, result, client)
    assert semantic["status"] in {"api_error", "incomplete_response", "invalid_response"}
    assert semantic["structured_review"] is None
    report = verification_with_model_review(prepared, result, semantic)
    assert report["status"] == "needs_review"
    assert report["item_reviews"][0]["source_support"] == "needs_review"
    assert client.chat.completions.create.call_count == 1


def test_automated_source_failure_never_triggers_a_checker_request():
    prepared, _, result = prepared_result()
    prepared.source_records[0].raw_record["targetFromSourceId"] = "WRONG_TARGET"
    client = Mock()
    with pytest.raises(ValueError, match="Automated source/response checks"):
        run_semantic_check(prepared, result, client)
    client.chat.completions.create.assert_not_called()
