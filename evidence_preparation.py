"""Prepare source-linked evidence groups and model messages without inference.

The saved Agent 1 run remains authoritative. This module does not retrieve data,
construct a model client, prune evidence, or judge scientific sufficiency.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field

from evidence_followup import EvidenceFollowupCase
from evidence_selection import MaterialHandoff, ScopedMaterialHandoff, study_subjects, PROMPT, build_catalog, catalog_for_prompt, question_catalog
from schemas import Agent1RunOutput

DEFAULT_QUESTION = "針對查詢的疾病與標的，這組來源提供哪些相關引用與原始觀察？本次還缺哪些原始資料？"


class ModelEvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str
    reference_scope: Literal["raw_record", "collection_metadata"] = "raw_record"
    json_pointer: str = Field(description="RFC 6901 pointer into the chosen reference_scope")


class OrganizedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    predicate: str
    statement: str
    qualifications: list[str]
    evidence_refs: list[ModelEvidenceReference] = Field(min_length=1)


class OrganizationReviewFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    missing_information: list[str]
    evidence_refs: list[ModelEvidenceReference] = Field(min_length=1)


class OrganizedGroupOutput(BaseModel):
    """Provisional semantic response shape, not an approved scientific taxonomy."""

    model_config = ConfigDict(extra="forbid")
    group_id: str
    claims: list[OrganizedClaim]
    review_flags: list[OrganizationReviewFlag]


class PreparedSourceRecord(BaseModel):
    snapshot_id: str
    source: Literal["open_targets", "clinicaltrials_gov", "chembl", "europe_pmc"]
    record_id: str | None
    source_run_id: str | None = None
    datasource_id: str | None = None
    source_url: str
    retrieved_at: datetime
    raw_record: dict[str, Any]
    execution_origin: dict[str, str] | None = None


class EvidenceGroup(BaseModel):
    group_id: str
    source: str
    datasource_id: str | None
    disease_id: str
    target_id: str
    drug_name_key: str | None
    drug_name_variants: list[str]
    snapshot_ids: list[str]
    context_snapshot_ids: list[str]
    record_count: int
    field_inventory: dict[str, dict[str, int]]
    scalar_value_counts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class PreparedModelInput(BaseModel):
    group_id: str
    messages: list[dict[str, str]]
    serialized_messages_bytes: int
    response_schema: dict[str, Any]
    input_contract: Literal['expanded_v1','references_v1'] = 'expanded_v1'


class EvidencePreparation(BaseModel):
    research_question: str = DEFAULT_QUESTION
    model_input_contract: Literal['expanded_v1','references_v1'] = 'expanded_v1'
    schema_version: Literal["0.1"] = "0.1"
    stage: Literal["ready_for_model"] = "ready_for_model"
    model_invoked: Literal[False] = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_run_id: str
    query_context: dict[str, Any]
    source_collections: list[dict[str, Any]]
    source_records: list[PreparedSourceRecord]
    source_field_claims: list[dict[str, Any]]
    collection_review_flags: list[dict[str, Any]]
    supplemental_requests: list[dict[str, Any]]
    groups: list[EvidenceGroup]
    model_inputs: list[PreparedModelInput]
    limitations: list[str]





def parse_organized_group_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith(("```json\n", "```\n")) and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    elif text.endswith("\n```"):
        text = text[:-4].rstrip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected one JSON object")
    return payload


def resolve_json_pointer(record: Any, pointer: str) -> Any:
    """Resolve a source-field reference, including nested arrays and escaped keys."""
    if pointer == "":
        return record
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must be empty or start with /")
    value = record
    for token in pointer[1:].split("/"):
        # Reject invalid escapes rather than silently interpreting a wrong path.
        index = 0
        while index < len(token):
            if token[index] == "~":
                if index + 1 == len(token) or token[index + 1] not in "01":
                    raise ValueError("Invalid JSON Pointer escape")
                index += 1
            index += 1
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isascii() and key.isdecimal() and (
            key == "0" or not key.startswith("0")
        ) and int(key) < len(value):
            value = value[int(key)]
        else:
            raise ValueError(f"Source field does not exist: {pointer}")
    return value


def resolve_evidence_reference(
    preparation: EvidencePreparation, ref: ModelEvidenceReference,
) -> Any:
    """Resolve native data or the exact run/datasource metadata of its anchor."""
    matches = [s for s in preparation.source_records if s.snapshot_id == ref.snapshot_id]
    if len(matches) != 1:
        raise ValueError("Reference must identify exactly one source snapshot")
    source = matches[0]
    if ref.reference_scope == "raw_record":
        document = source.raw_record
    else:
        collections = [c for c in preparation.source_collections
                       if c["datasource_id"] == source.datasource_id
                       and source.source_run_id == preparation.source_run_id]
        if len(collections) != 1:
            raise ValueError("Metadata reference needs one matching run/datasource collection")
        document = collections[0]
    return resolve_json_pointer(document, ref.json_pointer)


def validate_organized_group(
    payload: dict[str, Any], preparation: EvidencePreparation, group_id: str,
) -> OrganizedGroupOutput:
    """Check a future response's shape and source links, not scientific entailment."""
    result = OrganizedGroupOutput.model_validate(payload)
    if result.group_id != group_id:
        raise ValueError("Response group_id differs from the requested group")
    group = next((g for g in preparation.groups if g.group_id == group_id), None)
    if group is None:
        raise ValueError("Unknown evidence group")
    allowed = set(group.snapshot_ids + group.context_snapshot_ids)
    for item in [*result.claims, *result.review_flags]:
        for ref in item.evidence_refs:
            if ref.snapshot_id not in allowed:
                raise ValueError("Evidence reference is outside the supplied group/context")
            resolve_evidence_reference(preparation, ref)
    return result


def _label(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _scope(row: dict[str, Any], disease_id: str, target_id: str) -> tuple[str, str, str | None]:
    # Explicit source IDs take precedence over the enclosing pair's query scope.
    return (
        _label(row.get("diseaseFromSourceMappedId")) or disease_id,
        _label(row.get("targetFromSourceId")) or target_id,
        _label(row.get("drugFromSource")),
    )


def _inventory(records: list[PreparedSourceRecord], declared: set[str]) -> dict[str, dict[str, int]]:
    fields = declared | {key for record in records for key in record.raw_record}
    result = {}
    for key in sorted(fields):
        counts = dict(populated=0, null=0, empty=0, absent=0)
        for record in records:
            if key not in record.raw_record:
                counts["absent"] += 1
            elif record.raw_record[key] is None:
                counts["null"] += 1
            elif record.raw_record[key] in ("", [], {}):
                counts["empty"] += 1
            else:
                counts["populated"] += 1
        result[key] = counts
    return result


def _scalar_counts(records: list[PreparedSourceRecord]) -> dict[str, list[dict[str, Any]]]:
    """Count top-level scalar values exactly; do not flatten nested source data."""
    counts: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        for field, value in record.raw_record.items():
            if isinstance(value, (dict, list)):
                continue
            marker = json.dumps(value, ensure_ascii=False)
            values = counts.setdefault(field, {})
            item = values.setdefault(marker, {"value": value, "count": 0})
            item["count"] += 1
    return {field: list(values.values()) for field, values in sorted(counts.items())}


def prepare_evidence(
    run: Agent1RunOutput, followup: EvidenceFollowupCase | None = None,
    *, research_question: str = DEFAULT_QUESTION,
) -> EvidencePreparation:
    """Group every saved row; optionally attach the already-collected supplement."""
    evidence = run.agent_1_evidence
    run_id = str(run.run_id)
    records: list[PreparedSourceRecord] = []
    scopes: dict[str, tuple[str, str, str | None]] = {}
    declared_fields: dict[str, set[str]] = {}
    collections = []
    for source_index, source in enumerate(evidence.datasource_evidence):
        collections.append(source.model_dump(mode="json", exclude={"records"}))
        for row_index, row in enumerate(source.records):
            snapshot_id = str(uuid5(NAMESPACE_URL, f"{run_id}/{source_index}/{row_index}"))
            record = PreparedSourceRecord(
                snapshot_id=snapshot_id, source="open_targets", record_id=_label(row.get("id")),
                source_run_id=run_id, datasource_id=source.datasource_id,
                source_url=evidence.source_url, retrieved_at=evidence.retrieved_at,
                raw_record=deepcopy(row),
            )
            records.append(record)
            scopes[snapshot_id] = _scope(row, evidence.disease_id, evidence.target_id)
            declared_fields[snapshot_id] = set(source.evidence_fields)

    claims, flags, requests = [], [], []
    context_links: dict[str, list[str]] = defaultdict(list)
    limitations = [*evidence.limitations]
    if followup is not None:
        # Revalidate mutable Pydantic inputs before trusting their source links.
        followup = EvidenceFollowupCase.model_validate(followup.model_dump())
        if (followup.disease_id, followup.target_id) != (evidence.disease_id, evidence.target_id):
            raise ValueError("Supplemental case belongs to a different disease/target pair")
        for snapshot in followup.source_records:
            if snapshot.source != "open_targets":
                continue
            matches = [r for r in records if r.record_id == snapshot.record_id
                       and r.datasource_id == snapshot.datasource_id
                       and r.source_run_id == snapshot.source_run_id]
            if len(matches) != 1 or matches[0].raw_record != snapshot.raw_record:
                raise ValueError("Supplemental origin does not match exactly one saved source record")
            record = matches[0]
            old_id = record.snapshot_id
            record.snapshot_id = snapshot.snapshot_id
            scopes[record.snapshot_id] = scopes.pop(old_id)
            declared_fields[record.snapshot_id] = declared_fields.pop(old_id)
        existing = {r.snapshot_id for r in records}
        if any(s.snapshot_id not in existing for s in followup.source_records
               if s.source == "open_targets"):
            raise ValueError("Supplemental origins must map to distinct saved source records")
        for snapshot in followup.source_records:
            if snapshot.source == "open_targets":
                continue
            links = [r for r in followup.supplemental_requests
                     if r.result_snapshot_id == snapshot.snapshot_id]
            if len(links) != 1 or links[0].origin_snapshot_id not in existing:
                raise ValueError("Supplemental snapshot needs one traceable original record")
            if snapshot.snapshot_id in existing:
                raise ValueError("Duplicate source snapshot ID")
            origin_id = links[0].origin_snapshot_id
            record = PreparedSourceRecord.model_validate(snapshot.model_dump())
            records.append(record)
            scopes[record.snapshot_id] = scopes[origin_id]
            declared_fields[record.snapshot_id] = set()
            context_links[record.snapshot_id].append(origin_id)
            context_links[origin_id].append(record.snapshot_id)
            existing.add(record.snapshot_id)
        claims = [c.model_dump(mode="json") for c in followup.claims]
        flags = [f.model_dump(mode="json") for f in followup.review_flags]
        requests = [r.model_dump(mode="json") for r in followup.supplemental_requests]
        limitations.extend(followup.limitations)

    if len({r.snapshot_id for r in records}) != len(records):
        raise ValueError("Source snapshot IDs must be unique")
    buckets: dict[tuple, list[PreparedSourceRecord]] = defaultdict(list)
    for record in records:
        disease, target, drug = scopes[record.snapshot_id]
        buckets[(record.source, record.datasource_id, disease, target,
                 drug.strip().casefold() if drug else None)].append(record)
    groups = []
    for key, rows in buckets.items():
        source, datasource, disease, target, drug_key = key
        ids = [r.snapshot_id for r in rows]
        context_ids = sorted({s for rid in ids for s in context_links[rid]} - set(ids))
        group_id = str(uuid5(NAMESPACE_URL, json.dumps([run_id, *key], ensure_ascii=False)))
        groups.append(EvidenceGroup(
            group_id=group_id, source=source, datasource_id=datasource,
            disease_id=disease, target_id=target, drug_name_key=drug_key,
            drug_name_variants=sorted({scopes[r.snapshot_id][2] for r in rows
                                       if scopes[r.snapshot_id][2] is not None}),
            snapshot_ids=ids, context_snapshot_ids=context_ids, record_count=len(rows),
            field_inventory=_inventory(rows, set().union(*(declared_fields[i] for i in ids))),
            scalar_value_counts=_scalar_counts(rows),
        ))

    query_context = {
        "disease_query": evidence.disease_query, "disease_id": evidence.disease_id,
        "disease_name": evidence.disease_name, "target_query": evidence.target_query,
        "target_id": evidence.target_id, "target_symbol": evidence.target_symbol,
        "overall_association_score": evidence.overall_association_score,
        "datasource_scores": [s.model_dump(mode="json") for s in evidence.datasource_scores],
    }
    limitations.extend([
        "All collected records and null fields are retained; no evidence pruning was performed.",
        "Drug grouping normalizes only whitespace/case of source labels; it does not resolve drug identity or synonyms.",
        "Collection scope, field presence and reference validity do not establish scientific sufficiency or claim support.",
        "claims/review_flags response fields are provisional; the scientific taxonomy is not finalized.",
    ])
    limitations = list(dict.fromkeys(limitations))
    preparation = EvidencePreparation(
        source_run_id=run_id, query_context=query_context, source_collections=collections,
        source_records=records, source_field_claims=claims, collection_review_flags=flags,
        supplemental_requests=requests, groups=groups, model_inputs=[],
        limitations=limitations, research_question=research_question,
    )
    preparation.model_inputs = [
        selection_input(preparation, group, research_question)[0] for group in groups
    ]
    return preparation


def selection_input(preparation: EvidencePreparation, group: EvidenceGroup,
                    research_question: str) -> tuple[PreparedModelInput, dict[str, dict]]:
    for record in preparation.source_records:
        if record.execution_origin or 'execution_calls' in preparation.query_context:
            if record.source_run_id != preparation.source_run_id or (record.execution_origin or {}).get('run_id') != preparation.source_run_id:
                raise ValueError('Cross-run or unscoped execution record cannot enter model input')
    record_map = {r.snapshot_id: r for r in preparation.source_records}
    units = build_catalog(group, record_map, preparation.source_collections, preparation.source_run_id)
    response_schema = ScopedMaterialHandoff.model_json_schema()
    compact=preparation.model_input_contract=='references_v1'
    from evidence_selection import group_source_subjects,group_selection_scope
    source_subjects=group_source_subjects(preparation,group) if compact else {}
    selection_scope=group_selection_scope(preparation,group) if compact else None
    subjects = study_subjects(research_question, units, record_map)
    subject_schema = response_schema['$defs']['ScopedMaterialGap']['properties']['subject_ref']
    subject_schema.clear()
    subject_schema.update({'enum': [None, *subjects], 'default': None})
    candidates = preparation.query_context.get('followup_candidates', [])
    request_schema = response_schema['$defs']['ScopedMaterialGap']['properties']['followup_request_id']
    request_schema.clear()
    request_schema.update({'enum': [None, *[c['request_id'] for c in candidates]], 'default': None})
    coverage = preparation.query_context.get('execution_coverage', [])
    response_schema['$defs']['CoverageReference']['properties']['coverage_id']['enum'] = [c['coverage_id'] for c in coverage]
    if 'ContentAnnotation' in response_schema['$defs']:
        response_schema['$defs']['ContentAnnotation']['properties']['entity_ref']={'enum':[None,*[
            e['entity_ref'] for e in preparation.query_context.get('manifest',{}).get('entities',[]) if not compact or e['entity_type']=='target']],'default':None}
        annotation_properties=response_schema['$defs']['ContentAnnotation']['properties']
        if compact:annotation_properties['source_subject_ref']={'enum':[None,*source_subjects],'default':None}
        else:annotation_properties.pop('source_subject_ref',None)
    if coverage:
        properties = response_schema['$defs']['CoverageReference']['properties']
        properties['field_id'] = {'enum':[None, *[f['field_id'] for c in coverage for f in c.get('field_options', [])]], 'default':None}
        properties['required_pointer'] = {'enum':[None], 'default':None, 'description':'Select field_id; Python supplies its native-record-relative pointer.'}
    for definition in response_schema["$defs"].values():
        evidence_ids = definition.get("properties", {}).get("evidence_ids")
        if evidence_ids is not None:
            if compact:evidence_ids['items']={'$ref':'#/$defs/EvidenceId'}
            else:evidence_ids["items"]["enum"] = list(units)
    native_ids = [key for key, unit in units.items() if unit['reference_scope'] == 'raw_record']
    annotation_ids = response_schema['$defs']['ContentAnnotation']['properties']['evidence_ids']
    if compact:
        response_schema['$defs']['EvidenceId'] = {'type': 'string', 'enum': list(units)}
        native_set = set(native_ids)
        metadata_ids = [key for key in units if key not in native_set]
        native_id = {'$ref': '#/$defs/EvidenceId'}
        if metadata_ids:
            native_id = {'allOf': [native_id, {'not': {'enum': metadata_ids}}]}
        response_schema['$defs']['NativeEvidenceId'] = {
            **native_id, 'description': 'Content annotations cite native record fields only.'
        }
        annotation_ids['items'] = {'$ref': '#/$defs/NativeEvidenceId'}
    elif native_ids:
        annotation_ids['items']['enum'] = native_ids
    else:
        annotation_ids['items'] = False
    allowed = set(group.snapshot_ids + group.context_snapshot_ids)
    model_facts = deepcopy([c for c in preparation.source_field_claims if all(
        r["snapshot_id"] in allowed for r in c["evidence_refs"])])
    for fact in model_facts:
        for ref in fact["evidence_refs"]:
            path = ref.pop("field_path")
            ref["json_pointer"] = "/" + "/".join(
                part.replace("~", "~0").replace("/", "~1") for part in path.split("."))
    scoped_context={k:v for k,v in preparation.query_context.items() if (k != 'execution_calls' or not coverage) and (not compact or k!='native_target_scopes')}
    if 'relative_expression_rankings' in scoped_context:
        origins={(r.execution_origin.get('call_id'),r.execution_origin.get('json_pointer'))
                 for r in preparation.source_records if r.snapshot_id in group.snapshot_ids+group.context_snapshot_ids and r.execution_origin}
        scoped_rankings=[]
        for ranking in scoped_context['relative_expression_rankings']:
            result=deepcopy(ranking['result']); result['groups']=[
                {**g,'rows':[r for r in g['rows'] if (r['source_ref'],r['pointer']) in origins]}
                for g in result['groups']]
            result['groups']=[g for g in result['groups'] if g['rows']]
            if result['groups']:scoped_rankings.append(dict(step_id=ranking['step_id'],result=result))
        scoped_context['relative_expression_rankings']=scoped_rankings
    data = {
        "query_context": scoped_context,
        "execution_coverage": coverage,
        "source_collections": preparation.source_collections,
        "group": group.model_dump(mode="json"),
        "records": [record_map[i].model_dump(mode="json")
                    for i in [*group.snapshot_ids, *group.context_snapshot_ids]],
        "source_field_claims": model_facts,
        "collection_review_flags": [f for f in preparation.collection_review_flags if f["snapshot_id"] in allowed],
        "supplemental_requests": [r for r in preparation.supplemental_requests
                                  if r["origin_snapshot_id"] in allowed
                                  or r["result_snapshot_id"] in allowed],
        "limitations": preparation.limitations,
        "catalog": catalog_for_prompt(units, group.snapshot_ids + group.context_snapshot_ids),
        "questions": question_catalog(research_question),
        "subjects": subjects,
        "research_question": research_question,
        "followup_candidates": candidates,
    }
    if compact:
        ids=group.snapshot_ids+group.context_snapshot_ids;indices={sid:i for i,sid in enumerate(ids)}
        data['source_subjects']={ref:{**{k:v for k,v in subject.items() if k!='snapshot_ids'},
            'record_indices':[indices[sid] for sid in subject['snapshot_ids']]} for ref,subject in source_subjects.items()}
        data['selection_scope']=selection_scope
        from filter_assessment import group_assessed_result
        data['assessed_result']=group_assessed_result(preparation,group)
    packet_scope = ""
    if any(record_map[sid].execution_origin for sid in allowed):
        packet_scope = (
            "\nSelect observations from the current group. Use execution_coverage, execution_calls when present, "
            "and source_collections to distinguish group visibility from run-wide availability. "
            "Flag only a specific requested missing native datum supported by that scope; otherwise use review_flags: []. "
            "For a gap, use matching coverage_refs (native subject, source, material_kind) when available. "
            "For an existing native field choose field_id from that coverage entry's field_options; Python supplies its pointer. "
            "Leave required_pointer null; never write or transform whole-call paths into record-relative paths. "
            "A field absent from the inventory cannot be proven missing just by inventing a pointer: retain the specific material need for review. "
            "Null, false, zero, empty arrays and absent fields are different source states. A collected row does not prove all details exist. "
            "Distinguish collected elsewhere, partial/truncated, empty_result (queried, no rows), not_collected and source_error. "
            "Truncation limits the collection. Choose a matching followup_request_id only for a specific gap: "
            "available reuses collected context; ready may retrieve in one supplemental round; attempted is not retried. "
            "Use null for an unmatched locator; retain a real gap for review without guessing IDs. "
            "Follow-up retrieves source material and cannot determine efficacy or causality. "
        )
    compact_prompt=''
    if compact:
        from evidence_packet import pack_input,INSTRUCTIONS
        data=pack_input(data);compact_prompt=INSTRUCTIONS
        compact_prompt+=('Use source_subject_ref for source-native targets and entity_ref only for named Planner targets. '
                         'Python assessed_result comes from executed source inputs. If allow_empty_response=true, '
                         'claims=[] and review_flags=[] are valid for a bounded zero or excluded background group. '
                         'Zero matches alone do not require follow-up; retain specific missing fields and source failures. ')
    json_options={'ensure_ascii':False,**({'separators':(',',':')} if compact else {})}
    messages = [
        {"role": "system", "content": PROMPT + packet_scope + compact_prompt + "\nResponse schema:\n"
         + json.dumps(response_schema, **json_options)},
        {"role": "user", "content": json.dumps(data, **json_options)},
    ]
    return PreparedModelInput(
        group_id=group.group_id, messages=messages, response_schema=response_schema,
        serialized_messages_bytes=len(json.dumps(messages, ensure_ascii=False).encode("utf-8")),
        input_contract=preparation.model_input_contract,
    ), units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="Saved Agent1RunOutput JSON")
    parser.add_argument("--followup", type=Path, help="Already-collected EvidenceFollowupCase JSON")
    parser.add_argument("--output", type=Path, required=True, help="Prepared model-input JSON")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Research question for evidence selection")
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.evidence, args.followup) if p}:
        parser.error("Output must differ from the saved input files")
    try:
        run = Agent1RunOutput.model_validate_json(args.evidence.read_text(encoding="utf-8"))
        followup = (EvidenceFollowupCase.model_validate_json(args.followup.read_text(encoding="utf-8"))
                    if args.followup else None)
        prepared = prepare_evidence(run, followup, research_question=args.question)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prepared.model_dump_json(indent=2), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Evidence preparation failed: {exc}\n")
    print(json.dumps({
        "stage": prepared.stage, "model_invoked": prepared.model_invoked,
        "source_records": len(prepared.source_records), "groups": len(prepared.groups),
        "model_inputs": len(prepared.model_inputs),
        "max_serialized_messages_bytes": max(
            (i.serialized_messages_bytes for i in prepared.model_inputs), default=0),
        "output": str(args.output.resolve()),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
