import json
from copy import deepcopy
from pathlib import Path

import pytest

from evidence_preparation import EvidencePreparation, parse_organized_group_json, resolve_json_pointer, selection_input
from evidence_selection import GroupSelection, MaterialHandoff, QuestionSelection, assemble_selection, assemble_question_selection
from evidence_verification import verify_evidence_result
from test_nim_evidence import fixture_response


def test_saved_response_with_one_trailing_fence_still_requires_one_complete_object():
    root = Path(__file__).resolve().parents[1]
    path = root / "tests/fixtures/replay/NCT04241068_unresolved_question_pilot.json"
    raw = json.loads(path.read_text(encoding="utf8"))["response_text"]
    assert raw.endswith("\n```")
    QuestionSelection.model_validate(parse_organized_group_json(raw))
    for invalid in ('{}\nexplanation\n```', '{}\n{}\n```', '{\n```', '{}\n```\n```', '[]\n```'):
        with pytest.raises(ValueError):
            parse_organized_group_json(invalid)


def test_saved_cases_preserve_source_rows_scope_and_whole_outcome():
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/replay/APP_ready_for_model_v3_20261106.json"
    prep = EvidencePreparation.model_validate_json(path.read_text(encoding="utf8"))
    records = {r.snapshot_id: r for r in prep.source_records}
    lecanemab = next(g for g in prep.groups if g.drug_name_key == "lecanemab")
    _, units = selection_input(prep, lecanemab, "開發階段？")
    stages = [u for u in units.values() if u["json_pointer"] == "/clinicalStage"]
    assert {u["value"]: u["record_count"] for u in stages}["APPROVAL"] == 4
    assert sum(u["record_count"] for u in stages) == 23
    for unit in stages:
        assert unit["snapshot_ids"] == [sid for sid in lecanemab.snapshot_ids
                                      if records[sid].raw_record["clinicalStage"] == unit["value"]]
    ct = next(g for g in prep.groups if g.source == "clinicaltrials_gov")
    prepared, units = selection_input(prep, ct, "核心期安全性？")
    assert json.loads(prepared.messages[1]["content"])["research_question"] == "核心期安全性？"
    endpoint = next(u for u in units.values() if u["json_pointer"].endswith("/outcomeMeasures/0"))
    value = endpoint["value"]
    catalog = json.loads(prepared.messages[1]["content"])["catalog"]
    for key, unit in units.items():
        if isinstance(unit.get("value"), dict) and "timeFrame" in unit["value"]:
            assert catalog[key]["timeFrame"] == unit["value"]["timeFrame"]
    endpoint_key = next(k for k, u in units.items() if u is endpoint)
    assert catalog[endpoint_key]["title"] == value["title"]
    assert value == resolve_json_pointer(records[ct.snapshot_ids[0]].raw_record, endpoint["json_pointer"])
    assert "Treatment-Emergent" in value["title"] and "causal relationship" in value["description"]
    assert "Week 118" in value["timeFrame"] and "core period" in value["populationDescription"]
    measurement = value["classes"][0]["categories"][0]["measurements"][0]
    denom = value["denoms"][0]["counts"][0]
    assert measurement["groupId"] == denom["groupId"]
    assert (measurement["value"], denom["value"]) == ("1549", "1696")
    assert all(u["record_role"] == "context" for u in units.values()
               if ct.context_snapshot_ids[0] in u["snapshot_ids"])


def test_model_cannot_supply_source_facts_or_declare_a_populated_field_missing():
    prep, response = fixture_response()
    group = prep.groups[0]
    _, units = selection_input(prep, group, "why?")
    records = {r.snapshot_id: r for r in prep.source_records}
    injected = deepcopy(response)
    injected["claims"][0]["value"] = "drug-related"
    with pytest.raises(ValueError):
        MaterialHandoff.model_validate(injected)
    legacy_item = {"interpretation": "舊格式", "evidence_ids": response["claims"][0]["evidence_ids"]}
    response = {"group_id": group.group_id, "claims": [legacy_item],
                "review_flags": [{**legacy_item, "missing_field_ids": legacy_item["evidence_ids"]}]}
    with pytest.raises(ValueError, match="Populated"):
        assemble_selection(GroupSelection.model_validate(response), group.group_id, units, records)


def test_verification_rebuilds_assembled_values_and_keeps_relevance_pending():
    prep, response = fixture_response()
    group = prep.groups[0]
    _, units = selection_input(prep, group, "why?")
    records = {r.snapshot_id: r for r in prep.source_records}
    result = dict(source_run_id=prep.source_run_id, group_id=group.group_id, finish_reason="stop",
                  output_contract="source_material_v1", research_question=prep.research_question,
                  response_text=json.dumps(response),
                  structured_output=assemble_question_selection(MaterialHandoff.model_validate(response),
                                      group.group_id, prep.research_question, units, records))
    report = verify_evidence_result(prep, result)
    assert report["source_comparison_pending"]
    key = response["claims"][0]["evidence_ids"][0]
    result["structured_output"]["source_facts"][key]["value"] = "different reason"
    assert verify_evidence_result(prep, result)["status"] == "failed"


def test_subquestion_cannot_inject_conclusions_or_claim_an_answer_without_evidence():
    prep, response = fixture_response()
    group = prep.groups[0]
    _, units = selection_input(prep, group, prep.research_question)
    records = {r.snapshot_id: r for r in prep.source_records}
    for extra in ({"group_id": "invented"}, {"missing_field_ids": []}):
        with pytest.raises(ValueError):
            MaterialHandoff.model_validate({**response, **extra})
    response["claims"][0]["question_id"] = "Q2"
    with pytest.raises(ValueError, match="Unknown question_id"):
        assemble_question_selection(MaterialHandoff.model_validate(response), group.group_id,
                                    prep.research_question, units, records)
    response["claims"][0]["evidence_ids"] = []
    with pytest.raises(ValueError):
        MaterialHandoff.model_validate(response)


def test_compound_input_binds_multiple_items_without_splitting_or_requoting():
    prep, response = fixture_response()
    question = "比較 KRAS 和 EGFR 的證據，並說明哪些關聯仍待判讀。"
    group = prep.groups[0]
    prepared, units = selection_input(prep, group, question)
    assert json.loads(prepared.messages[1]["content"])["questions"] == {"Q1": question}
    out = assemble_question_selection(MaterialHandoff.model_validate(response),
                                      group.group_id, question, units,
                                      {s.snapshot_id: s for s in prep.source_records})
    for item in out["claims"] + out["review_flags"]:
        assert out["questions"][item["question_id"]] == question
        assert "question_span" not in item


def test_stop_reason_can_have_a_narrower_pending_question_only_in_review_flags():
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "tests/fixtures/replay/NCT04241068_question_evidence_v1.json")
                       .read_text(encoding="utf8"))
    prep = EvidencePreparation.model_validate_json(
        (root / "tests/fixtures/replay/APP_ready_for_question_evidence_20261106.json").read_text(encoding="utf8"))
    assert all(c["status"] == "passed" for c in verify_evidence_result(prep, saved)["automated_checks"])
    group = next(g for g in prep.groups if g.group_id == saved["group_id"])
    question = saved["research_question"]
    _, units = selection_input(prep, group, question)
    refs = [k for k, unit in units.items() if unit["json_pointer"] == "/trialWhyStopped"]
    assert len(refs) == 1 and units[refs[0]]["value"] == "Sponsor's Decision"
    direct = {"question_span": "停試原因", "evidence_ids": refs}
    detail = "贊助者決定終止整項試驗的具體考量是什麼？"
    pending = {**direct, "unresolved_question": detail}
    selection = QuestionSelection.model_validate({"claims": [direct], "review_flags": [pending]})
    out = assemble_question_selection(selection, group.group_id, question, units,
                                      {s.snapshot_id: s for s in prep.source_records})
    assert detail not in question and out["review_flags"] == [pending] and out["claims"] == [direct]
    with pytest.raises(ValueError):
        QuestionSelection.model_validate({"claims": [pending], "review_flags": []})


def test_material_handoff_separates_locator_from_source_field_and_reviews_published_items():
    prep, response = fixture_response()
    group = prep.groups[0]
    _, units = selection_input(prep, group, prep.research_question)
    locator = next(k for k, u in units.items() if u["json_pointer"] == "/clinicalReportId")
    response["claims"][0]["evidence_ids"].append(locator)
    records = {s.snapshot_id: s for s in prep.source_records}
    out = assemble_question_selection(MaterialHandoff.model_validate(response), group.group_id,
                                      prep.research_question, units, records)
    assert [c["source_form"] for c in out["claims"]] == ["source_field", "reference"]
    assert out["claims"][1]["evidence_ids"] == [locator]
    result = dict(group_id=group.group_id, source_run_id=prep.source_run_id, research_question=prep.research_question,
                  status="references_validated", finish_reason="stop", response_text=json.dumps(response),
                  output_contract="source_material_v1", structured_output=out)
    report = verify_evidence_result(prep, result)
    assert [r["item_path"] for r in report["item_reviews"]] == ["/claims/0", "/claims/1", "/review_flags/0"]
    assert report["source_comparison_pending"]
    # The Agent 1 contract asks for missing material rather than scientific judgment.
    response["review_flags"][0] = {"question_id": "Q1", "evidence_ids": [], "unresolved_question": "治療有效嗎？"}
    with pytest.raises(ValueError):
        MaterialHandoff.model_validate(response)


def test_saved_material_gap_replay_links_exact_trial_without_changing_model_response():
    from evidence_followup import EvidenceFollowupCase
    from evidence_preparation import prepare_evidence
    from schemas import Agent1RunOutput

    root = Path(__file__).resolve().parents[1] / "tests/fixtures/replay"
    run = Agent1RunOutput.model_validate_json((root / "agent1_APP_full_evidence.json").read_text("utf8"))
    case = EvidenceFollowupCase.model_validate_json(
        (root / "NCT01767311_material_references.json").read_text("utf8"))
    saved = json.loads((root / "material_references_3.json").read_text("utf8"))
    question = saved["research_question"]
    prep = prepare_evidence(run, case, research_question=question)
    group = next(g for g in prep.groups if g.group_id == saved["group_id"])
    _, units = selection_input(prep, group, question)
    selection = MaterialHandoff.model_validate_json(saved["response_text"])
    original = selection.model_dump()
    assembled = assemble_question_selection(selection, group.group_id, question, units,
                                             {s.snapshot_id: s for s in prep.source_records})
    assert assembled["review_flags"][0]["evidence_ids"] == ["E29"]
    assert assembled["source_facts"]["E29"]["value"] == "nct01767311"
    assert selection.model_dump() == original
    assert original["review_flags"][0]["evidence_ids"] == []
    assert assembled["claims"] == saved["structured_output"]["claims"]
    saved["structured_output"] = assembled
    report = verify_evidence_result(prep, saved)
    gap, = [r for r in report["item_reviews"] if r["item_path"] == "/review_flags/0"]
    assert gap["reference_status"] == "passed"
    assert gap["source_values"][0]["evidence_id"] == "E29"
    assert all(c["status"] == "passed" for c in report["automated_checks"])
    # The verifier must reject tampering with the derived gap links too.
    saved["structured_output"]["review_flags"][0]["evidence_ids"] = []
    assert verify_evidence_result(prep, saved)["status"] == "failed"


@pytest.mark.parametrize("text, expected", [
    ("NCT01767311 的結果", ["trial"]),
    ("nct01767311的結果", ["trial"]),
    ("NCT017673110 的結果", []),
    ("XNCT01767311 的結果", []),
    ("NCT00000000 的結果", []),
    ("該研究的結果", []),
    ("33865446 人的觀察", []),
    ("PMID:33865446 的全文", ["paper"]),
    ("PMID：33865446的全文", ["paper"]),
    ("PMC123456 的全文", ["pmc"]),
])
def test_gap_linking_requires_explicit_typed_identifier_not_shared_question(text, expected):
    prep, _ = fixture_response()
    record = next(r for r in prep.source_records if r.source == "open_targets")
    units = {}
    for key, pointer, value in [
        ("trial", "/clinicalReportId", "nct01767311"),
        ("other_trial", "/clinicalReportId", "nct99999999"),
        ("paper", "/literature", ["33865446"]),
        ("pmc", "/pubMedCentralIds", ["PMC123456"]),
        ("non_locator", "/trialWhyStopped", "NCT01767311"),
    ]:
        units[key] = dict(reference_scope="raw_record", json_pointer=pointer, value=value,
                          field_state="populated", snapshot_ids=[record.snapshot_id])
    selection = MaterialHandoff.model_validate({
        "claims": [{"question_id": "Q1", "evidence_ids": ["other_trial"]}],
        "review_flags": [{"question_id": "Q1", "evidence_ids": [], "missing_material": text}]})
    out = assemble_question_selection(selection, prep.groups[0].group_id, "比較多項研究", units,
                                      {record.snapshot_id: record})
    assert out["review_flags"][0]["evidence_ids"] == expected


def test_registry_gap_links_and_existing_model_links_are_preserved_and_deduplicated():
    prep, _ = fixture_response()
    record = prep.source_records[0].model_copy(update={"source": "clinicaltrials_gov"})
    units = {"trial": dict(reference_scope="raw_record", field_state="populated",
                           json_pointer="/protocolSection/identificationModule",
                           value={"nctId": "NCT01767311"}, snapshot_ids=[record.snapshot_id])}
    selection = MaterialHandoff.model_validate({"claims": [], "review_flags": [
        {"question_id": "Q1", "evidence_ids": ["trial"], "missing_material": "NCT01767311 的原始結果"}]})
    out = assemble_question_selection(selection, prep.groups[0].group_id, "結果", units,
                                      {record.snapshot_id: record})
    assert out["review_flags"][0]["evidence_ids"] == ["trial"]
    assert out["source_facts"]["trial"]["value"] == {"nctId": "NCT01767311"}
