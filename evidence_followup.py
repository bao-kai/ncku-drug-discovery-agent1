"""Minimal source-linked facts, flags, and single-study follow-up for Agent 1.

This is the deterministic factual baseline for the proposed evidence organizer.
It does not invoke a model, establish scientific claim taxonomy, or overwrite
raw evidence. Cross-record semantic organization and Parquet storage are separate
work; this module first makes one real supplemental-retrieval case inspectable.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from query_clinicaltrials import (
    ClinicalTrialLookup, get_clinical_trial, normalize_nct_id,
)
from schemas import Agent1RunOutput


class SourceSnapshot(BaseModel):
    snapshot_id: str
    source: Literal["open_targets", "clinicaltrials_gov"]
    record_id: str
    source_run_id: str | None = None
    datasource_id: str | None = None
    source_url: str
    retrieved_at: datetime
    raw_record: dict[str, Any]


class EvidenceReference(BaseModel):
    snapshot_id: str
    field_path: str


class StructuredFact(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    claim_kind: Literal["source_field_fact"] = "source_field_fact"
    subject_id: str
    predicate: str
    value: Any
    evidence_refs: list[EvidenceReference] = Field(min_length=1)
    generated_by: Literal["python"] = "python"


class ReviewFlag(BaseModel):
    flag_id: str = Field(default_factory=lambda: str(uuid4()))
    flag_type: Literal["field_not_collected"] = "field_not_collected"
    snapshot_id: str
    field_name: str
    generated_by: Literal["python"] = "python"
    resolved_by_snapshot_id: str | None = None


STATUS_PATH = "protocolSection.statusModule.overallStatus"
STOP_REASON_PATH = "protocolSection.statusModule.whyStopped"
UPDATED_PATH = "protocolSection.statusModule.lastUpdatePostDateStruct.date"


class SupplementalRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    origin_snapshot_id: str
    record_id: str
    source: Literal["clinicaltrials_gov"] = "clinicaltrials_gov"
    requested_fields: list[str] = Field(
        default_factory=lambda: [STATUS_PATH, STOP_REASON_PATH, UPDATED_PATH]
    )
    status: Literal["pending", "completed", "partial", "not_found", "error"] = "pending"
    missing_fields: list[str] = Field(default_factory=list)
    error: str | None = None
    result_snapshot_id: str | None = None


def _field_value(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Source field does not exist: {path}")
        value = value[part]
    return value


class EvidenceFollowupCase(BaseModel):
    schema_version: Literal["0.1"] = "0.1"
    disease_id: str
    target_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_records: list[SourceSnapshot]
    claims: list[StructuredFact] = Field(default_factory=list)
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    supplemental_requests: list[SupplementalRequest] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_links(self) -> "EvidenceFollowupCase":
        snapshots = {item.snapshot_id: item for item in self.source_records}
        if len(snapshots) != len(self.source_records):
            raise ValueError("Source snapshot IDs must be unique")
        for claim in self.claims:
            for ref in claim.evidence_refs:
                if claim.predicate != ref.field_path:
                    raise ValueError("Source-field fact predicate must identify the referenced field")
                if ref.snapshot_id not in snapshots:
                    raise ValueError("Claim references an unknown snapshot")
                raw_value = _field_value(snapshots[ref.snapshot_id].raw_record, ref.field_path)
                if type(raw_value) is not type(claim.value) or raw_value != claim.value:
                    raise ValueError("Source-field fact value differs from the referenced raw field")
        for flag in self.review_flags:
            if flag.snapshot_id not in snapshots:
                raise ValueError("Review flag references an unknown snapshot")
            if flag.resolved_by_snapshot_id and flag.resolved_by_snapshot_id not in snapshots:
                raise ValueError("Review flag resolution references an unknown snapshot")
        for request in self.supplemental_requests:
            origin = snapshots.get(request.origin_snapshot_id)
            if origin is None:
                raise ValueError("Supplemental request has no origin snapshot")
            if normalize_nct_id(origin.raw_record.get("clinicalReportId", "")) != request.record_id:
                raise ValueError("Supplemental request does not match the original report ID")
            if request.result_snapshot_id:
                result = snapshots.get(request.result_snapshot_id)
                if result is None or result.source != request.source or result.record_id != request.record_id:
                    raise ValueError("Supplemental result does not match the requested source/record")
        return self


def _fact(snapshot: SourceSnapshot, subject: str, path: str) -> StructuredFact:
    return StructuredFact(
        subject_id=subject,
        predicate=path,
        value=_field_value(snapshot.raw_record, path),
        evidence_refs=[EvidenceReference(snapshot_id=snapshot.snapshot_id, field_path=path)],
    )


def build_followup_case(run: Agent1RunOutput, evidence_id: str) -> EvidenceFollowupCase:
    """Prepare one known trial record; never treat absent stop text as termination."""
    evidence = run.agent_1_evidence
    matches = [
        (source, row)
        for source in evidence.datasource_evidence
        for row in source.records
        if row.get("id") == evidence_id
    ]
    if len(matches) != 1:
        raise ValueError("Select exactly one existing evidence record")
    source, row = matches[0]
    if row.get("overallStatus") not in (None, "", [], {}):
        raise ValueError("This missing-status follow-up case already contains overallStatus")
    nct_id = normalize_nct_id(row.get("clinicalReportId", ""))
    snapshot = SourceSnapshot(
        snapshot_id=str(uuid4()), source="open_targets", record_id=evidence_id,
        source_run_id=str(run.run_id), datasource_id=source.datasource_id,
        source_url=evidence.source_url, retrieved_at=evidence.retrieved_at,
        raw_record=deepcopy(row),
    )
    facts = [
        _fact(snapshot, nct_id, key)
        for key in ("drugFromSource", "clinicalReportId", "clinicalStage",
                    "trialWhyStopped", "trialStopReasonCategories")
        if row.get(key) not in (None, "", [], {})
    ]
    # The saved Open Targets record has no study-status field. This is a
    # collection gap, not a claim that the registry itself lacks the information.
    flags = [ReviewFlag(snapshot_id=snapshot.snapshot_id, field_name=STATUS_PATH)]
    request = SupplementalRequest(origin_snapshot_id=snapshot.snapshot_id, record_id=nct_id)
    return EvidenceFollowupCase(
        disease_id=evidence.disease_id, target_id=evidence.target_id,
        source_records=[snapshot], claims=facts, review_flags=flags,
        supplemental_requests=[request],
        limitations=[
            *evidence.limitations, *source.collection_warnings,
            "Source-field facts copy source statements; they do not establish efficacy or causality.",
            "A returned stop-reason string does not establish a more specific reason than its source text.",
        ],
    )


def apply_trial_lookup(
    case: EvidenceFollowupCase, request_id: str, lookup: ClinicalTrialLookup,
) -> EvidenceFollowupCase:
    """Return a new case with the supplement linked; preserve every original row."""
    result = case.model_copy(deep=True)
    matches = [item for item in result.supplemental_requests if item.request_id == request_id]
    if len(matches) != 1:
        raise ValueError("Unknown supplemental request")
    request = matches[0]
    if request.status != "pending":
        raise ValueError("Supplemental request has already been executed")
    if lookup.record_id != request.record_id:
        raise ValueError("Lookup ID differs from supplemental request")
    if lookup.status != "success":
        request.status = lookup.status
        request.error = lookup.error
        return EvidenceFollowupCase.model_validate(result.model_dump())
    raw = lookup.raw_record
    if raw is None or _field_value(raw, "protocolSection.identificationModule.nctId") != request.record_id:
        raise ValueError("Supplemental raw record has the wrong study ID")
    snapshot = SourceSnapshot(
        snapshot_id=str(uuid4()), source=lookup.source, record_id=lookup.record_id,
        source_url=lookup.source_url, retrieved_at=lookup.retrieved_at,
        raw_record=deepcopy(raw),
    )
    result.source_records.append(snapshot)
    request.result_snapshot_id = snapshot.snapshot_id
    for path in request.requested_fields:
        try:
            value = _field_value(raw, path)
        except ValueError:
            value = None
        if value in (None, "", [], {}):
            request.missing_fields.append(path)
        else:
            result.claims.append(_fact(snapshot, request.record_id, path))
            for flag in result.review_flags:
                if flag.snapshot_id == request.origin_snapshot_id and flag.field_name == path:
                    flag.resolved_by_snapshot_id = snapshot.snapshot_id
    request.status = "partial" if request.missing_fields else "completed"
    return EvidenceFollowupCase.model_validate(result.model_dump())


def execute_trial_followup(case: EvidenceFollowupCase, request_id: str) -> EvidenceFollowupCase:
    """Execute only the explicitly selected pending request; no recursive lookup."""
    request = next((r for r in case.supplemental_requests if r.request_id == request_id), None)
    if request is None or request.status != "pending":
        raise ValueError("Select a pending supplemental request")
    return apply_trial_lookup(case, request_id, get_clinical_trial(request.record_id))
