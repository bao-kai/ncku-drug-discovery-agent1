"""Versioned synthetic demonstrations, not domain-expert approved answers."""

import json

from schemas import QueryManifestV2

EXAMPLE_VERSION = "synthetic_v1"


def _step(number, operation, description, dependencies, outputs, sources=None, **extra):
    return dict(step_id=f"s{number}", operation=operation,
                description=description, depends_on=[f"s{x}" for x in dependencies],
                outputs=outputs, sources=sources or [], **extra)


def _plan(question, entities, steps, primary, safeguards):
    return dict(schema_version="2.0", original_question=question,
                research_objective=question, entities=[dict(entity_ref=ref,
                    entity_type=kind, mention=mention, resolution_status="unresolved")
                    for ref, kind, mention in entities], plan_steps=steps,
                evidence_requirements=["保留每個結果的來源、查詢日期與資料缺口"],
                completion_criteria=["完成問題要求的查詢與分析；未解析或缺資料時明列限制，不捏造結論"],
                global_safeguards=["provenance_pagination_truncation", "missing_not_negative", *safeguards],
                characteristics={"primary": primary, "secondary": []},
                created_before_retrieval=True, generated_at="2026-09-04T00:00:00Z")


def examples():
    # Fictional placeholders intentionally assert no biomedical facts.
    pair = _plan(
        "請規劃查詢疾病甲與標的乙的精確關聯，列出各資料來源分數並依分數排序；不要用文獻篇數代替分數。",
        [("d1", "disease", "疾病甲"), ("t1", "target", "標的乙")],
        [
            _step(1, "resolve_disease_target", "解析兩個名稱；有歧義先釐清，未解析成功不進行後續查詢。", [], ["resolved_pair_ids"], ["open_targets"], entity_refs=["d1", "t1"]),
            _step(2, "retrieve_exact_pair", "使用 s1 的疾病與標的 ID 查精確配對，不以疾病標的前幾名清單代替。", [1], ["pair_association"], ["open_targets"], entity_refs=["d1", "t1"]),
            _step(3, "retrieve_datasource_scores", "取得 s2 精確配對的各來源關聯分數，保留缺失狀態。", [2], ["datasource_scores"], ["open_targets"], entity_refs=["d1", "t1"]),
            _step(4, "rank_datasource_contributions", "對 s3 已取得的來源分數排序，不以紀錄數補分數。", [3], ["ranked_datasources"], ranking={"primary_metric": "datasource_score", "direction": "descending", "secondary_metrics": []}, aggregation={"datasource_contribution_comparison": "preserve_source_scores_and_missingness"}),
        ], "evidence_directed", ["association_score_not_probability", "datasource_score_not_record_count"])
    shared = _plan(
        "請規劃找出疾病丙與疾病丁共同的標的，只保留兩邊關聯分數都大於 0.4 的標的，最後依疾病丙的分數排序；不評估療效。",
        [("d1", "disease", "疾病丙"), ("d2", "disease", "疾病丁")],
        [
            _step(1, "resolve_diseases", "分別解析兩個疾病名稱，不合併疾病範圍；有歧義先釐清。", [], ["resolved_diseases"], ["open_targets"], entity_refs=["d1", "d2"]),
            _step(2, "discover_targets_per_disease", "以 s1 的疾病 ID 分別取得標的與分數，保留各自分頁及截斷狀態。", [1], ["targets_by_disease"], ["open_targets"], entity_refs=["d1", "d2"]),
            _step(3, "intersect_targets", "將 s2 兩個集合以標的 ID 取交集，保留各疾病獨立分數。", [2], ["shared_targets"], aggregation={"target_set_intersection": "by_target_id", "report_scores_per_disease": True}),
            _step(4, "filter_targets", "篩選 s3 結果，兩邊都須大於使用者指定門檻；缺分數不視為零。", [3], ["filtered_shared_targets"], filters={"association_score_by_disease": {"d1": {"gt": 0.4}, "d2": {"gt": 0.4}}, "missing_score_policy": "report_separately"}),
            _step(5, "rank_targets", "依 s4 中疾病丙的分數排序，不產生未要求的綜合分數。", [4], ["ranked_shared_targets"], ranking={"primary_metric": "d1_overall_association_score", "direction": "descending", "secondary_metrics": []}),
        ], "multi_hop", ["association_score_not_probability", "association_does_not_imply_efficacy", "score_not_comparable_without_context"])
    return [
        {"id": "synthetic_pair_sources", "rationale": "先解析身分，再精確查詢；來源排序必須依賴已取得的來源分數。", "plan": pair},
        {"id": "synthetic_shared_filtered", "rationale": "兩個疾病各自查詢後才取交集；篩選在排序之前。問題不問治療支持，因此不強加藥物查詢。", "plan": shared},
    ]


def build_few_shot_prompt(mode="off"):
    if mode == "off":
        return ""
    if mode != EXAMPLE_VERSION:
        raise ValueError("PLANNER_FEW_SHOT_MODE must be off or synthetic_v1")
    items = examples()
    for item in items:
        QueryManifestV2.model_validate(item["plan"])
    return (
        "\n\nPlanning demonstrations (synthetic placeholders, no retrieved facts). "
        "Learn the relationship between the question, sources, dependencies and analysis; "
        "do not copy entities, timestamps, thresholds, step counts or irrelevant operations. "
        "These are examples, not a mandatory workflow or the current question's answer.\n"
        + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    )
