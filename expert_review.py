"""Deterministic, human-readable views over raw Agent 1 evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from schemas import (
    DataSourceReviewView,
    DiseaseTargetEvidence,
    DrugEvidenceGroup,
    ExpertReviewView,
    FieldProfile,
)


FIELD_DESCRIPTIONS = {
    "id": "Open Targets 對此筆證據配置的唯一識別碼。",
    "datasourceId": "提供此筆證據的 Open Targets 資料來源。",
    "datatypeId": "證據所屬的大類，例如 known drug。",
    "score": "此筆證據的 Open Targets 分數；不是療效或成功率。",
    "drugFromSource": "原始來源使用的藥物或治療名稱。",
    "clinicalStage": "此臨床報告對應的開發階段或核准狀態。",
    "clinicalReportId": "臨床試驗、監管文件或其他臨床報告識別碼。",
    "studyStartDate": "來源記錄的研究開始日期。",
    "evidenceDate": "此筆臨床證據的日期。",
    "diseaseFromSource": "原始來源使用的疾病名稱。",
    "diseaseFromSourceMappedId": "原始疾病名稱正規化後的 ontology ID。",
    "targetFromSource": "原始來源使用的 Target 名稱。",
    "targetFromSourceId": "原始來源或正規化後的 Target ID。",
    "directionOnTarget": "作用於 Target 的方向，例如 LoF 或 GoF。",
    "directionOnTrait": "對疾病表型的方向，例如 protect 或 risk。",
    "trialWhyStopped": "試驗提前停止時記錄的原始原因文字。",
    "trialStopReasonCategories": "Open Targets 對停止原因的分類。",
    "literature": "支持此筆證據的文獻識別碼，通常為 PMID。",
    "publicationDate": "相關文獻或來源的發表日期。",
}

DATASOURCE_DESCRIPTIONS = {
    "clinical_precedence": (
        "藥物作用於指定 Target，且曾針對指定疾病進入臨床開發或核准的紀錄。"
        "其中可同時包含成功、失敗、停止及結果未明的資料。"
    )
}

STAGE_RANK = {
    "UNKNOWN": 0,
    "PHASE_1": 1,
    "PHASE_1_2": 2,
    "PHASE_2": 3,
    "PHASE_2_3": 4,
    "PHASE_3": 5,
    "PHASE_4": 6,
    "APPROVAL": 7,
}


def _present(value: Any) -> bool:
    return value not in (None, "", [], {})


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _compact_example(value: Any) -> Any:
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, list):
        return value[:10]
    if isinstance(value, dict):
        return dict(list(value.items())[:10])
    return value


def _field_profiles(records: list[dict[str, Any]]) -> list[FieldProfile]:
    names = sorted({key for row in records for key in row})
    result = []
    for name in names:
        values = [row.get(name) for row in records if _present(row.get(name))]
        if not values:
            continue
        examples = []
        seen = set()
        for value in values:
            marker = repr(value)
            if marker in seen:
                continue
            seen.add(marker)
            examples.append(_compact_example(value))
            if len(examples) == 3:
                break
        result.append(
            FieldProfile(
                field_name=name,
                plain_language_description=FIELD_DESCRIPTIONS.get(
                    name, "Open Targets 原始 evidence 欄位；用途需由領域專家確認。"
                ),
                populated_record_count=len(values),
                populated_percentage=round(len(values) / len(records) * 100, 2),
                value_type=_value_type(values[0]),
                example_values=examples,
            )
        )
    return sorted(
        result,
        key=lambda item: (-item.populated_record_count, item.field_name),
    )


def _counter(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(v) for v in values if _present(v)).items()))


def _drug_groups(records: list[dict[str, Any]]) -> list[DrugEvidenceGroup]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        name = str(row.get("drugFromSource") or "[drug name unavailable]").strip()
        grouped[name.casefold()].append(row)

    output = []
    for canonical_key, rows in grouped.items():
        variants = sorted({str(row.get("drugFromSource")) for row in rows if row.get("drugFromSource")})
        display_name = Counter(
            str(row.get("drugFromSource"))
            for row in rows
            if row.get("drugFromSource")
        ).most_common(1)
        stages = [row.get("clinicalStage") for row in rows]
        valid_stages = [str(stage) for stage in stages if _present(stage)]
        highest = max(valid_stages, key=lambda x: STAGE_RANK.get(x, -1), default=None)
        reports = sorted({str(row["clinicalReportId"]) for row in rows if _present(row.get("clinicalReportId"))})
        literature = sorted(
            {
                str(item)
                for row in rows
                for item in (row.get("literature") or [])
                if _present(item)
            }
        )
        stop_rows = [row for row in rows if _present(row.get("trialWhyStopped")) or _present(row.get("trialStopReasonCategories"))]
        categories = [
            str(category)
            for row in stop_rows
            for category in (row.get("trialStopReasonCategories") or [])
        ]
        reasons = []
        for row in stop_rows:
            reason = row.get("trialWhyStopped")
            if reason and reason not in reasons:
                reasons.append(str(reason))
            if len(reasons) == 5:
                break
        output.append(
            DrugEvidenceGroup(
                canonical_drug_name=(display_name[0][0] if display_name else canonical_key),
                source_name_variants=variants,
                evidence_record_count=len(rows),
                clinical_stage_counts=_counter(stages),
                highest_clinical_stage=highest,
                evidence_score_counts=_counter([row.get("score") for row in rows]),
                report_ids=reports,
                literature_ids=literature,
                direction_on_target_counts=_counter([row.get("directionOnTarget") for row in rows]),
                direction_on_trait_counts=_counter([row.get("directionOnTrait") for row in rows]),
                stopped_trial_record_count=len(stop_rows),
                negative_stop_record_count=sum("Negative" in (row.get("trialStopReasonCategories") or []) for row in stop_rows),
                safety_stop_record_count=sum("Safety_Sideeffects" in (row.get("trialStopReasonCategories") or []) for row in stop_rows),
                stop_reason_categories=_counter(categories),
                stop_reason_examples=reasons,
            )
        )
    return sorted(output, key=lambda item: (-item.evidence_record_count, item.canonical_drug_name.casefold()))


def build_expert_review_view(evidence: DiseaseTargetEvidence) -> ExpertReviewView:
    reviews = []
    for source in evidence.datasource_evidence:
        profiles = _field_profiles(source.records)
        reviews.append(
            DataSourceReviewView(
                datasource_id=source.datasource_id,
                datasource_explanation=DATASOURCE_DESCRIPTIONS.get(
                    source.datasource_id,
                    "此資料來源的醫藥意義尚待領域專家確認。",
                ),
                aggregated_score=source.aggregated_score,
                score_interpretation_warning=(
                    "此數值是 Open Targets 對多筆證據的聚合分數，不是療效、"
                    "成功率、因果機率或本專案信心分數。"
                ),
                total_records=source.total_evidence_count,
                fetched_records=source.fetched_evidence_count,
                complete=not source.truncated,
                raw_schema_field_count=len(source.evidence_fields),
                populated_field_count=len(profiles),
                field_profiles=profiles,
                drug_groups=_drug_groups(source.records),
                review_questions=[
                    "哪些欄位應成為資料庫固定欄位，哪些只需保留在 raw JSON？",
                    "同一藥物的多筆報告應以藥物、臨床試驗、適應症或文件為主鍵？",
                    "成功、失敗、停止、安全性與結果不明應採用哪些正式分類？",
                    "哪些臨床階段與停止原因需要人工複核？",
                    "哪些來源 ID 應繼續連到 ClinicalTrials.gov、Europe PMC、ChEMBL 或監管文件？",
                ],
            )
        )
    return ExpertReviewView(
        disease=evidence.disease_name,
        disease_id=evidence.disease_id,
        target=evidence.target_symbol,
        target_id=evidence.target_id,
        datasource_reviews=reviews,
    )
