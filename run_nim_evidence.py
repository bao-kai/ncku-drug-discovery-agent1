"""Run the first prepared evidence group against NVIDIA's hosted NIM endpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evidence_preparation import (
    EvidencePreparation, parse_organized_group_json, selection_input,
)
from evidence_selection import (ScopedMaterialHandoff, assemble_question_selection, validate_content_annotation_scopes,
    group_source_subjects,group_selection_scope,attach_packet_scope)
from evidence_verification import verify_evidence_result
from nim_inference import MODEL, BASE_URL, GENERATION_PARAMETERS, stream_nim_completion
from filter_assessment import group_assessed_result


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "outputs/evidence_preparation/APP_ready_for_source_material_20261107.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs/evidence_organization/APP_nim_group_1.json"


def run_prepared_group(
    preparation: EvidencePreparation, group_id: str, client: Any, *, research_question: str | None = None,
) -> dict[str, Any]:
    """One API request; preserve reasoning separately from the final JSON answer."""
    groups = [g for g in preparation.groups if g.group_id == group_id]
    if len(groups) != 1:
        raise ValueError("Select exactly one prepared evidence group")
    question = preparation.research_question if research_question is None else research_question
    prepared, units = selection_input(preparation, groups[0], question)
    result: dict[str, Any] = {
        **stream_nim_completion(client, prepared.messages),
        "source_run_id": preparation.source_run_id,
        "group_id": group_id,
        "output_contract": "source_material_v4" if prepared.input_contract=='references_v1' else "source_material_v2",
        "input_contract": prepared.input_contract,
        "research_question": question,
        "validation_scope": "source_assembly_and_references_only",
        "structured_output": None,
    }
    if result["status"] != "response_received":
        return result
    try:
        selection = ScopedMaterialHandoff.model_validate(parse_organized_group_json(result["response_text"]))
        subjects=group_source_subjects(preparation,groups[0]) if prepared.input_contract=='references_v1' else None
        validate_content_annotation_scopes(selection,preparation.query_context,source_subjects=subjects,units=units)
        from automatic_followup import validate_followup_selection
        validate_followup_selection(selection, preparation.query_context.get('followup_candidates', []))
        from execution_coverage import validate_coverage_selection
        validate_coverage_selection(selection, preparation.query_context.get('execution_coverage', []))
        result["structured_output"] = assemble_question_selection(
            selection, group_id, question, units, {r.snapshot_id: r for r in preparation.source_records},
            assessed_result=group_assessed_result(preparation,groups[0]) if prepared.input_contract=='references_v1' else None,
        )
        if prepared.input_contract=='references_v1':
            attach_packet_scope(result['structured_output'],subjects,group_selection_scope(preparation,groups[0]))
    except ValueError as exc:
        result["status"] = "invalid_response"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return result
    result["status"] = "references_validated"
    return result


def main() -> None:
    # Fill NVIDIA_API_KEY in this project's .env; do not put the key in this file.
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(f"Fill NVIDIA_API_KEY in {PROJECT_ROOT / '.env'} before running.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install dependencies: .venv\\Scripts\\python.exe -m pip install -r requirements.txt") from exc
    try:
        preparation = EvidencePreparation.model_validate_json(INPUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot load prepared evidence: {exc}") from exc
    if not preparation.model_inputs:
        raise SystemExit("No prepared evidence group is available.")

    group_id = preparation.model_inputs[0].group_id
    print(f"Calling NVIDIA NIM for evidence group 1/{len(preparation.model_inputs)}: {group_id}", flush=True)
    # One selected group only. Disable hidden SDK retries; timeout is per network
    # operation, not a total wall-clock deadline for the complete streamed response.
    with OpenAI(base_url=BASE_URL, api_key=api_key, max_retries=0, timeout=1200) as client:
        result = run_prepared_group(preparation, group_id, client)
    result["prepared_input_file"] = str(INPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    verification = verify_evidence_result(preparation, result)
    verification_path = OUTPUT_PATH.with_suffix(".verification.json")
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Status: {result['status']}\nSaved result: {OUTPUT_PATH}")
    print(f"Source-support review: {verification['status']}\nSaved verification: {verification_path}")
    if result["status"] != "references_validated" or verification["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
