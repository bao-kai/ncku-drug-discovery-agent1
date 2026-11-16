"""Offline preparation checks; no inference and no external retrieval."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from evidence_followup import apply_trial_lookup, build_followup_case
from evidence_preparation import (
    MaterialHandoff, prepare_evidence, resolve_json_pointer, validate_organized_group,
)
from schemas import Agent1RunOutput
from test_evidence_followup import lookup, saved_run


def test_groups_keep_every_source_row_null_and_source_score_without_mutation():
    run = saved_run()
    source = run.agent_1_evidence.datasource_evidence[0]
    row = deepcopy(source.records[0])
    row.update(id="E2", drugFromSource=" ADUCANUMAB ", optional=None, count=0, positive=False)
    source.records.append(row)
    source.fetched_evidence_count = source.total_evidence_count = 2
    source.aggregated_score = 0.7
    source.evidence_fields = ["declared_but_absent"]
    before = run.model_dump()
    with patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
        result = prepare_evidence(run)
    assert run.model_dump() == before
    assert len(result.groups) == len(result.model_inputs) == 1
    group = result.groups[0]
    assert group.record_count == 2
    assert group.drug_name_variants == [" ADUCANUMAB ", "aducanumab"]
    assert [r.raw_record for r in result.source_records] == source.records
    assert group.field_inventory["optional"] == dict(populated=0, null=1, empty=0, absent=1)
    assert group.field_inventory["count"]["populated"] == 1
    assert group.field_inventory["positive"]["populated"] == 1
    assert group.scalar_value_counts["count"] == [{"value": 0, "count": 1}]
    assert group.scalar_value_counts["positive"] == [{"value": False, "count": 1}]
    assert group.scalar_value_counts["optional"] == [{"value": None, "count": 1}]
    assert group.field_inventory["declared_but_absent"]["absent"] == 2
    assert result.source_collections[0]["aggregated_score"] == 0.7
    data = json.loads(result.model_inputs[0].messages[1]["content"])
    assert data["records"][1]["raw_record"]["optional"] is None
    assert result.model_invoked is False


def test_no_synonym_merge_no_invented_record_id_and_source_pair_precedes_query():
    run = saved_run()
    source = run.agent_1_evidence.datasource_evidence[0]
    row = deepcopy(source.records[0])
    row.pop("id")
    row.update(drugFromSource="different label", diseaseFromSourceMappedId="MONDO_OTHER",
               targetFromSourceId="ENSG_OTHER")
    source.records.append(row)
    result = prepare_evidence(run)
    assert len(result.groups) == 2
    group = result.groups[1]
    assert (group.disease_id, group.target_id) == ("MONDO_OTHER", "ENSG_OTHER")
    assert result.source_records[1].record_id is None
    assert result.source_collections[0]["fetched_evidence_count"] == 1
    assert sum(g.record_count for g in result.groups) == 2  # actual count, not reported count


def test_supplement_has_bidirectional_context_and_keeps_original_source_versions():
    run = saved_run()
    base = build_followup_case(run, "E1")
    case = apply_trial_lookup(base, base.supplemental_requests[0].request_id, lookup())
    before = case.model_dump()
    result = prepare_evidence(run, case)
    assert case.model_dump() == before
    assert len(result.source_records) == len(result.groups) == 2
    original, supplement = result.source_records
    assert original.snapshot_id == case.source_records[0].snapshot_id
    assert supplement.raw_record == case.source_records[1].raw_record
    assert result.collection_review_flags[0]["resolved_by_snapshot_id"] == supplement.snapshot_id
    assert result.supplemental_requests[0]["status"] == "completed"
    for group, prepared in zip(result.groups, result.model_inputs):
        data = json.loads(prepared.messages[1]["content"])
        assert len(data["records"]) == 2
        assert len(group.context_snapshot_ids) == 1
        assert len(data["source_field_claims"]) == len(case.claims)
        for fact in data["source_field_claims"]:
            for ref in fact["evidence_refs"]:
                raw = next(r["raw_record"] for r in data["records"]
                           if r["snapshot_id"] == ref["snapshot_id"])
                assert resolve_json_pointer(raw, ref["json_pointer"]) == fact["value"]
        assert prepared.serialized_messages_bytes == len(
            json.dumps(prepared.messages, ensure_ascii=False).encode("utf-8"))
        for definition in prepared.response_schema["$defs"].values():
            field = definition.get("properties", {}).get("evidence_ids")
            if field is not None:
                assert field["items"]["enum"] == list(data["catalog"])


@pytest.mark.parametrize("status", ["error", "not_found"])
def test_failed_supplement_stays_a_collection_error_without_added_evidence(status):
    run = saved_run()
    case = build_followup_case(run, "E1")
    case = apply_trial_lookup(case, case.supplemental_requests[0].request_id, lookup(status=status))
    result = prepare_evidence(run, case)
    assert len(result.source_records) == 1
    assert result.supplemental_requests[0]["status"] == status
    assert result.collection_review_flags[0]["resolved_by_snapshot_id"] is None


def test_supplement_cannot_be_attached_to_another_pair_or_changed_original():
    run = saved_run()
    case = build_followup_case(run, "E1")
    other = run.model_copy(deep=True)
    other.agent_1_evidence.target_id = "ENSG_OTHER"
    with pytest.raises(ValueError, match="different disease/target"):
        prepare_evidence(other, case)
    other = run.model_copy(deep=True)
    other.agent_1_evidence.datasource_evidence[0].records[0]["clinicalStage"] = "APPROVAL"
    with pytest.raises(ValueError, match="does not match exactly one"):
        prepare_evidence(other, case)


def test_two_supplement_origins_cannot_silently_replace_the_same_source_snapshot():
    run = saved_run()
    case = build_followup_case(run, "E1")
    duplicate = case.source_records[0].model_copy(deep=True)
    duplicate.snapshot_id = "another-snapshot"
    case.source_records.append(duplicate)
    with pytest.raises(ValueError, match="origins must map to distinct"):
        prepare_evidence(run, case)


def test_empty_source_is_preserved_without_fabricating_a_model_task():
    run = saved_run()
    source = run.agent_1_evidence.datasource_evidence[0]
    source.records.clear()
    source.truncated = True
    source.collection_warnings.append("Incomplete source collection")
    result = prepare_evidence(run)
    assert result.groups == result.model_inputs == result.source_records == []
    assert result.source_collections[0]["truncated"] is True
    assert result.source_collections[0]["collection_warnings"] == ["Incomplete source collection"]


def test_json_pointer_handles_nested_arrays_and_escaped_keys():
    raw = {"a/b": {"~field": [None, {"name": "保留原文"}]}}
    assert resolve_json_pointer(raw, "/a~1b/~0field/1/name") == "保留原文"
    assert resolve_json_pointer(raw, "") is raw
    assert resolve_json_pointer(raw, "/a~1b/~0field/0") is None
    for path in ("a/b", "/bad~2", "/a~1b/~0field/01", "/a~1b/~0field/-1", "/missing"):
        with pytest.raises(ValueError):
            resolve_json_pointer(raw, path)


def model_response(result):
    group = result.groups[0]
    return {
        "group_id": group.group_id,
        "claims": [{"subject": "NCT04241068", "predicate": "reported_stop_reason",
                    "statement": "來源記載 Sponsor's Decision。", "qualifications": [],
                    "evidence_refs": [{"snapshot_id": group.snapshot_ids[0],
                                       "json_pointer": "/trialWhyStopped"}]}],
        "review_flags": [],
    }


def test_contract_checks_correct_refs_unknown_refs_paths_and_wrong_group():
    result = prepare_evidence(saved_run())
    payload = model_response(result)
    assert validate_organized_group(payload, result, result.groups[0].group_id).claims
    for change, error in (("unknown_snapshot", "outside the supplied"),
                          ("missing_field", "Source field does not exist"),
                          ("wrong_group", "group_id differs")):
        bad = deepcopy(payload)
        if change == "unknown_snapshot":
            bad["claims"][0]["evidence_refs"][0]["snapshot_id"] = "unknown"
        elif change == "missing_field":
            bad["claims"][0]["evidence_refs"][0]["json_pointer"] = "/nonexistent"
        else:
            bad["group_id"] = "another-group"
        with pytest.raises(ValueError, match=error):
            validate_organized_group(bad, result, result.groups[0].group_id)


def test_contract_rejects_cross_group_reference_and_unapproved_confidence_field():
    run = saved_run()
    source = run.agent_1_evidence.datasource_evidence[0]
    other = deepcopy(source.records[0])
    other.update(id="E2", drugFromSource="another drug")
    source.records.append(other)
    result = prepare_evidence(run)
    payload = model_response(result)
    payload["claims"][0]["evidence_refs"][0]["snapshot_id"] = result.groups[1].snapshot_ids[0]
    with pytest.raises(ValueError, match="outside the supplied"):
        validate_organized_group(payload, result, result.groups[0].group_id)
    payload = model_response(result)
    payload["claims"][0]["confidence"] = 0.9
    with pytest.raises(ValueError):
        validate_organized_group(payload, result, result.groups[0].group_id)


def test_cli_prepares_without_loading_model_code_and_refuses_input_overwrite(tmp_path):
    script = Path(__file__).resolve().parents[1] / "evidence_preparation.py"
    source = tmp_path / "evidence.json"
    output = tmp_path / "prepared.json"
    source.write_text(saved_run().model_dump_json(), encoding="utf-8")
    original = source.read_bytes()
    command = [sys.executable, str(script), "--evidence", str(source), "--output", str(output)]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["model_invoked"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["stage"] == "ready_for_model"
    process = subprocess.run(command[:-1] + [str(source)], capture_output=True, text=True, check=False)
    assert process.returncode != 0
    assert "Output must differ" in process.stderr
    assert source.read_bytes() == original
    process = subprocess.run(
        [sys.executable, "-c", "import sys, evidence_preparation; "
         "assert 'model_factory' not in sys.modules; assert 'langchain' not in sys.modules"],
        cwd=script.parent, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr


def test_real_saved_app_case_covers_all_rows_and_supplement_without_truncation():
    root = Path(__file__).resolve().parents[1]
    run = Agent1RunOutput.model_validate_json(
        (root / "tests/fixtures/replay/agent1_APP_full_evidence.json").read_text(encoding="utf-8"))
    case = build_followup_case(run, "6631aff5404c8963172da6ef343edcd5617b8c41")
    case = apply_trial_lookup(case, case.supplemental_requests[0].request_id, lookup())
    result = prepare_evidence(run, case)
    lecanemab = next(g for g in result.groups if g.drug_name_key == "lecanemab")
    assert lecanemab.record_count == 23
    assert {x["value"]: x["count"] for x in lecanemab.scalar_value_counts["clinicalStage"]}["UNKNOWN"] == 8
    assert {x["value"]: x["count"] for x in lecanemab.scalar_value_counts["score"]}[1] == 6
    assert len(result.source_records) == sum(
        len(s.records) for s in run.agent_1_evidence.datasource_evidence) + 1
    assigned = [i for group in result.groups for i in group.snapshot_ids]
    assert len(assigned) == len(set(assigned)) == len(result.source_records)
    assert {i for i in assigned} == {s.snapshot_id for s in result.source_records}
    for group, model_input in zip(result.groups, result.model_inputs):
        data = json.loads(model_input.messages[1]["content"])
        assert len(data["records"]) == group.record_count + len(group.context_snapshot_ids)
        assert [d["raw_record"] for d in data["records"]] == [
            next(s.raw_record for s in result.source_records if s.snapshot_id == i)
            for i in group.snapshot_ids + group.context_snapshot_ids]
