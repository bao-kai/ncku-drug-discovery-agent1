"""Source checks and explicit source-support review reports for model evidence.

Python checks the response, source identity, provenance and collection metadata.
Semantic decisions are supplied by a named reviewer; missing decisions remain
needs_review. This module neither invokes another model nor changes evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from evidence_selection import (
    GroupSelection, IndexedQuestionSelection, MaterialHandoff, ScopedMaterialHandoff, QuestionSelection, assemble_question_selection, assemble_selection, build_catalog,
)

from evidence_preparation import (
    EvidencePreparation, OrganizedGroupOutput, resolve_json_pointer,
    parse_organized_group_json, _inventory, _scalar_counts,
    resolve_evidence_reference,
)


class SourceSupportReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["supported", "unsupported", "needs_review"]
    reason: str


def verify_evidence_result(
    preparation: EvidencePreparation,
    model_result: dict[str, Any],
    *,
    reviews: dict[str, SourceSupportReview] | None = None,
    reviewer: str | None = None,
    review_method: Literal["source_comparison", "model_initial"] = "source_comparison",
) -> dict[str, Any]:
    """Build a report; supplied reviews address /claims/N or /review_flags/N.

    A supported decision means the supplied source supports that wording with
    its qualifications, not that the underlying biomedical claim is proven.
    """
    reviews = {path: SourceSupportReview.model_validate(value)
               for path, value in (reviews or {}).items()}
    if reviews and not reviewer:
        raise ValueError("Source-support decisions need a named reviewer")
    report: dict[str, Any] = {
        "schema_version": "0.1", "group_id": model_result.get("group_id"),
        "source_run_id": preparation.source_run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "review_method": review_method,
        "source_comparison_pending": True,
        "scope": "source_consistency_and_wording_support_not_biomedical_truth",
        "automated_checks": [], "source_checks": [], "collection_checks": [],
        "item_reviews": [], "status": "needs_review",
    }

    def check(name: str, passed: bool, message: str) -> None:
        report["automated_checks"].append({
            "check": name, "status": "passed" if passed else "failed", "message": message,
        })

    check("source_run", model_result.get("source_run_id") == preparation.source_run_id,
          "Model result must belong to the saved source run.")
    group = next((g for g in preparation.groups if g.group_id == report["group_id"]), None)
    check("group", group is not None, "Model result must identify an existing evidence group.")
    check("generation_finished", model_result.get("finish_reason") == "stop",
          "Incomplete generation or API errors do not count as verified evidence.")
    check("transport_completed", model_result.get("status") != "api_error",
          "Transport errors remain failures even if a stop event was received.")
    if group is None:
        report["status"] = "failed"
        return report
    source_map = {s.snapshot_id: s for s in preparation.source_records}
    check("snapshot_ids_unique", len(source_map) == len(preparation.source_records),
          "Snapshot IDs must identify distinct saved source records.")
    allowed = set(group.snapshot_ids + group.context_snapshot_ids)
    own_records = [source_map[i] for i in group.snapshot_ids if i in source_map]
    actual_inventory = _inventory(own_records, set(group.field_inventory))
    actual_scalar_counts = _scalar_counts(own_records)
    check("group_inventory", group.record_count == len(own_records)
          and group.field_inventory == actual_inventory
          and (not group.scalar_value_counts or group.scalar_value_counts == actual_scalar_counts),
          "Group counts and field statistics must agree with saved raw records.")
    report["group_inventory"] = {
        "record_count": len(own_records), "field_inventory": actual_inventory,
        "scalar_value_counts": actual_scalar_counts,
    }
    for snapshot_id in sorted(allowed):
        source = source_map.get(snapshot_id)
        if source is None:
            check("snapshot_present", False, f"Missing saved snapshot: {snapshot_id}")
            continue
        issues = []
        identity_errors = []
        if not source.source_url or (not source.record_id and not source.execution_origin):
            issues.append("Source URL or source-native record ID is missing.")
        raw = source.raw_record
        if source.execution_origin:
            origin = source.execution_origin
            if not origin.get('call_id') or 'json_pointer' not in origin or not origin.get('run_id'):
                identity_errors.append('Execution source locator is incomplete.')
            # Native structures vary by API endpoint. Exact pointer/value/metadata
            # matching against the original execution is checked by its adapter.
        elif source.source == "open_targets":
            if source.record_id and raw.get("id") != source.record_id:
                identity_errors.append("Raw evidence ID differs from the saved record ID.")
            for field, expected in (("diseaseFromSourceMappedId", group.disease_id),
                                    ("targetFromSourceId", group.target_id)):
                actual = raw.get(field)
                if actual in (None, ""):
                    issues.append(f"{field} is not provided; group scope alone does not verify this entity.")
                elif actual != expected:
                    identity_errors.append(f"{field}={actual!r} differs from group scope {expected!r}.")
        elif source.source == "clinicaltrials_gov":
            try:
                nct_id = resolve_json_pointer(raw, "/protocolSection/identificationModule/nctId")
            except ValueError:
                nct_id = None
            if nct_id != source.record_id:
                identity_errors.append("Returned registry NCT ID differs from the saved record ID.")
        report["source_checks"].append({
            "snapshot_id": snapshot_id, "source": source.source,
            "record_id": source.record_id, "source_url": source.source_url,
            "retrieved_at": source.retrieved_at.isoformat(),
            "status": "failed" if identity_errors else "needs_review" if issues else "passed",
            "issues": identity_errors + issues,
        })

    datasources = {source_map[i].datasource_id for i in allowed if i in source_map}
    for collection in preparation.source_collections:
        if collection["datasource_id"] not in datasources:
            continue
        actual_count = sum((s.source == "open_targets" or s.execution_origin is not None)
                           and s.datasource_id == collection["datasource_id"]
                           and s.source_run_id == preparation.source_run_id
                           for s in preparation.source_records)
        reported_count = collection["fetched_evidence_count"]
        warnings = collection.get("collection_warnings", [])
        truncated = collection["truncated"]
        report["collection_checks"].append({
            "datasource_id": collection["datasource_id"],
            "reported_total": collection["total_evidence_count"],
            "reported_fetched": reported_count, "saved_records": actual_count,
            "count_consistent": reported_count == actual_count,
            "truncated": truncated, "collection_warnings": warnings,
            "status": ("failed" if reported_count != actual_count else
                       "needs_review" if truncated or warnings else "passed"),
            "message": "Row-count completeness does not establish field completeness or scientific sufficiency.",
        })

    try:
        # Re-parse the actual final text; do not trust a cached success label.
        raw = parse_organized_group_json(model_result.get("response_text", ""))
        contract = model_result.get("output_contract")
        if contract in {'source_material_v3','source_material_v4'}:
            check('input_contract',preparation.model_input_contract=='references_v1' and model_result.get('input_contract')=='references_v1',
                  'New source scopes must use the matching self-contained input contract; historical outputs keep their original contract.')
        question_schemas = {"question_evidence_v1": QuestionSelection, "question_evidence_v2": IndexedQuestionSelection,
                            "source_material_v1": MaterialHandoff, "source_material_v2": ScopedMaterialHandoff,
                            "source_material_v3": ScopedMaterialHandoff,"source_material_v4":ScopedMaterialHandoff}
        question_contract = contract in question_schemas
        selection_contract = question_contract or model_result.get("output_contract") == "source_selection_v1"
        if selection_contract:
            if not allowed.issubset(source_map):
                raise ValueError("Missing source snapshot for evidence assembly")
            units = build_catalog(group, source_map, preparation.source_collections, preparation.source_run_id)
            if question_contract:
                question = model_result.get("research_question")
                if not isinstance(question, str):
                    raise ValueError("Missing research question for assembly")
                schema = question_schemas[contract]
                payload = schema.model_validate(raw)
                if isinstance(payload, ScopedMaterialHandoff):
                    if contract in {'source_material_v3','source_material_v4'}:
                        from evidence_selection import validate_content_annotation_scopes,group_source_subjects
                        validate_content_annotation_scopes(payload,preparation.query_context,
                            source_subjects=group_source_subjects(preparation,group),units=units)
                    elif any(a.source_subject_ref is not None for a in payload.content_annotations):
                        raise ValueError('Legacy output contract has no source_subject_ref')
                    from automatic_followup import validate_followup_selection
                    validate_followup_selection(payload, preparation.query_context.get('followup_candidates', []))
                    from execution_coverage import validate_coverage_selection
                    validate_coverage_selection(payload, preparation.query_context.get('execution_coverage', []))
                from filter_assessment import group_assessed_result
                assembled = assemble_question_selection(payload, group.group_id, question, units, source_map,
                    assessed_result=group_assessed_result(preparation,group) if contract=='source_material_v4' else None)
                if contract in {'source_material_v3','source_material_v4'}:
                    from evidence_selection import attach_packet_scope,group_selection_scope
                    attach_packet_scope(assembled,group_source_subjects(preparation,group),group_selection_scope(preparation,group))
            else:
                payload = GroupSelection.model_validate(raw)
                assembled = assemble_selection(payload, group.group_id, units, source_map)
            check("source_facts_match", json.dumps(assembled, sort_keys=True)
                  == json.dumps(model_result.get("structured_output"), sort_keys=True),
                  "Rebuild the selected facts, conditions, counts and provenance from the saved records.")
        else:
            payload = OrganizedGroupOutput.model_validate(raw)
        check("response_schema", True, "Response contains claims and review_flags in the expected format.")
    except ValueError as exc:
        check("response_schema", False, str(exc))
        report["status"] = "failed"
        return report
    if not question_contract:
        check("response_group", payload.group_id == group.group_id,
              "The response group ID must match the requested group.")
    known_paths = set()
    if question_contract:
        report["scope"] = "source_fields_and_selection_relevance_within_the_saved_group"
    if contract in {"source_material_v1", "source_material_v2", "source_material_v3", "source_material_v4"}:
        report["scope"] = "retrieved_material_identity_conditions_references_and_collection_gaps"
    for collection_name in ("claims", "review_flags"):
        items = getattr(payload, collection_name)
        if contract in {"source_material_v1", "source_material_v2", "source_material_v3", "source_material_v4"}:
            # Review the published groups after Python separates source forms.
            items = assembled[collection_name]
        for index, item in enumerate(items):
            path = f"/{collection_name}/{index}"
            known_paths.add(path)
            references = []
            if selection_contract:
                ids = item["evidence_ids"] if isinstance(item, dict) else item.evidence_ids
                for key in dict.fromkeys([*ids, *getattr(item, "missing_field_ids", [])]):
                    references.append({"evidence_id": key, **units[key], "status": "passed"})
            for ref in getattr(item, "evidence_refs", []):
                detail: dict[str, Any] = ref.model_dump()
                detail["status"] = "failed"
                if ref.snapshot_id not in allowed or ref.snapshot_id not in source_map:
                    detail["error"] = "Reference is outside this group's supplied evidence/context."
                else:
                    source = source_map[ref.snapshot_id]
                    try:
                        detail["raw_value"] = resolve_evidence_reference(preparation, ref)
                        detail["source_url"] = source.source_url
                        detail["retrieved_at"] = source.retrieved_at.isoformat()
                        detail["status"] = "passed"
                    except ValueError as exc:
                        detail["error"] = str(exc)
                references.append(detail)
            reference_status = "passed" if all(r["status"] == "passed" for r in references) else "failed"
            if question_contract and not references:
                reference_status = "none_selected"
            semantic = reviews.get(path)
            if reference_status == "failed":
                decision = "unsupported"
                reason = "Source links are invalid; this wording cannot be promoted as source-supported."
            elif semantic is None:
                decision = "needs_review"
                reason = ("Selection relevance, coverage and unresolved questions await source comparison."
                          if question_contract else "Source links exist; wording support has not been reviewed.")
            else:
                decision, reason = semantic.decision, semantic.reason
            report["item_reviews"].append({
                "item_path": path, "item": item if isinstance(item, dict) else item.model_dump(mode="json", exclude_unset=True),
                "reference_status": reference_status, "source_values": references,
                "source_support": decision, "reason": reason,
            })
    if set(reviews) - known_paths:
        raise ValueError("Source-support review addresses an item absent from the response")

    failed = any(c["status"] == "failed" for c in report["automated_checks"])
    failed |= any(c["status"] == "failed" for c in report["source_checks"])
    failed |= any(c["status"] == "failed" for c in report["collection_checks"])
    failed |= any(r["source_support"] == "unsupported" for r in report["item_reviews"])
    pending = any(c["status"] == "needs_review" for c in report["source_checks"])
    pending |= any(c["status"] == "needs_review" for c in report["collection_checks"])
    pending |= any(r["source_support"] == "needs_review" for r in report["item_reviews"])
    pending |= review_method == "model_initial"
    report["source_comparison_pending"] = (review_method == "model_initial" or not known_paths
        or any(r["source_support"] == "needs_review" for r in report["item_reviews"]))
    report["status"] = ("failed" if failed else "needs_review" if pending or not known_paths
                        else "source_reviewed_within_scope")
    return report
