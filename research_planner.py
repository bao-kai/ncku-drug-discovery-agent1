"""LLM Research Planner inside Agent 1; this module has no retrieval tools."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from langchain.agents import create_agent

from model_factory import get_chat_model, get_ollama_settings
from query_opentargets import discover_disease_targets, resolve_entity
from schemas import Agent1ResearchPlan, VerifiedEntity


REGISTERED_TOOLS = {
    "open_targets": "open_targets_search",
    "europe_pmc": "europe_pmc_search",
    "clinicaltrials_gov": "clinicaltrials_gov_search",
    "chembl": "chembl_search",
}

PLANNER_PROMPT = """
You are the Research Planner inside Agent 1. You have no tools. Produce only a
structured research plan from the verified entities supplied by the program.
The plan must include query_manifest, which is the immutable pre-retrieval
contract. Classify the question as one of: single_hop, multi_hop,
evidence_directed, filter_and_rank, comparison_and_aggregation, or
safety_and_negative_knowledge. For the currently supported disease-only input,
use single_hop and query_path disease→target. For a specified disease and target,
use evidence_directed and query_path disease→target→evidence.
For a specified target use target_expansion_strategy='specified_target'. For
disease-only planning choose either 'select_then_expand' or 'expand_all_top10'
according to the research objective and explain the choice. In the current
implementation all verified top targets remain in the manifest and subplans;
select_then_expand means the future execution stage may prioritize a subset.
All four registered data sources are available, but select only the source or
sources needed for the question and explain why. Every target subplan must use
the selected sources and their matching registered tool names. Never invent or alter IDs. Keep verified terms
separate from LLM-proposed unvalidated search terms. Plan queries only: do not
claim that evidence exists, judge evidence importance, or conclude association.
Open Targets integrates association evidence; never describe its associations as
already validated or proven. ClinicalTrials.gov planning must ask whether trials,
statuses, interventions, and posted results exist; never assume results exist.
ChEMBL planning must focus on target mapping, compounds, assays, bioactivity,
potency, and mechanisms; do not use ChEMBL alone to claim development progress.
Every target must list at least one anticipated data gap. Completion criteria
must require provenance, retrieval time, success/error state, pagination, and
truncation reporting for every source.
Collection limits are not yet configured, so every query must use
collection_policy_status='not_configured'. Allow at most one supplemental round.
Write objectives, rationales, and questions in clear Traditional Chinese.
"""


class ResearchPlannerError(RuntimeError):
    pass


def _validate_against_verified_context(plan: Agent1ResearchPlan, context: dict) -> None:
    """Reject any LLM change to IDs or claims that unverified terms are verified."""
    if plan.mode != context["mode"]:
        raise ValueError("plan mode differs from the deterministic input mode")
    disease = context["disease"]
    if (
        plan.disease.entity_id != disease["entity_id"]
        or plan.disease.canonical_name != disease["canonical_name"]
    ):
        raise ValueError("LLM changed the verified disease identity")
    plan.disease = VerifiedEntity.model_validate(disease)
    manifest = plan.query_manifest
    if manifest.original_question != plan.original_question:
        raise ValueError("query_manifest original_question differs from the plan")
    if manifest.disease.entity_id != disease["entity_id"]:
        raise ValueError("LLM changed the manifest disease identity")
    manifest.disease = VerifiedEntity.model_validate(disease)
    expected_targets = {target["entity_id"]: target for target in context["targets"]}
    actual_ids = {subplan.target.entity_id for subplan in plan.target_subplans}
    if actual_ids != set(expected_targets):
        raise ValueError("LLM changed, omitted, or added verified targets")
    manifest_ids = {target.entity_id for target in manifest.targets}
    if manifest_ids != set(expected_targets):
        raise ValueError("LLM changed, omitted, or added manifest targets")
    manifest.targets = [
        VerifiedEntity.model_validate(expected_targets[target.entity_id])
        for target in manifest.targets
    ]
    if context["mode"] == "disease_target":
        if manifest.query_type != "evidence_directed":
            raise ValueError("disease-target input requires evidence_directed query type")
        if manifest.target_expansion_strategy != "specified_target":
            raise ValueError("specified disease-target input requires specified_target")
        if manifest.query_path != ["disease", "target", "evidence"]:
            raise ValueError("disease-target input requires disease-target-evidence path")
    else:
        if manifest.query_type != "single_hop":
            raise ValueError("disease-only input requires single_hop query type")
        if manifest.target_expansion_strategy == "specified_target":
            raise ValueError("disease-only input cannot use specified_target")
        if manifest.query_path != ["disease", "target"]:
            raise ValueError("disease-only input requires disease-target path")
    for subplan in plan.target_subplans:
        expected = expected_targets[subplan.target.entity_id]
        if subplan.target.canonical_name != expected["canonical_name"]:
            raise ValueError("LLM changed a verified target name")
        subplan.target = VerifiedEntity.model_validate(expected)
        allowed_verified = {
            value.casefold()
            for value in (
                expected["query"],
                expected["canonical_name"],
                expected.get("symbol"),
                *expected.get("verified_synonyms", []),
                disease["query"],
                disease["canonical_name"],
                *disease.get("verified_synonyms", []),
            )
            if value
        }
        for query in subplan.queries:
            invalid = [
                term
                for term in query.verified_search_terms
                if term.casefold() not in allowed_verified
            ]
            if invalid:
                raise ValueError(
                    f"unverified terms were labeled verified: {invalid}; move them "
                    "to unvalidated_search_terms"
                )


def _verified_context(
    disease: str,
    target: str | None,
    top_n: int,
    disease_id: str | None = None,
    target_id: str | None = None,
) -> dict:
    if target:
        disease_hit = resolve_entity(disease, "disease", disease_id)
        target_hit = resolve_entity(target, "target", target_id)
        return {
            "mode": "disease_target",
            "disease": VerifiedEntity(
                query=disease,
                entity_id=disease_hit["id"],
                canonical_name=disease_hit["name"],
            ).model_dump(mode="json"),
            "targets": [
                VerifiedEntity(
                    query=target,
                    entity_id=target_hit["id"],
                    canonical_name=target_hit["name"],
                    symbol=target_hit.get("symbol") or target,
                ).model_dump(mode="json")
            ],
        }
    disease_hit = resolve_entity(disease, "disease", disease_id)
    discovery = discover_disease_targets(disease, top_n, disease_hit["id"])
    return {
        "mode": "disease_only",
        "disease": VerifiedEntity(
            query=disease,
            entity_id=discovery["disease_id"],
            canonical_name=discovery["disease_name"],
        ).model_dump(mode="json"),
        "targets": [
            VerifiedEntity(
                query=item["target_symbol"],
                entity_id=item["target_id"],
                canonical_name=item["target_name"],
                symbol=item["target_symbol"],
            ).model_dump(mode="json")
            for item in discovery["targets"]
        ],
    }


class Agent1ResearchPlanner:
    """Generate and validate a plan, with at most two LLM repair attempts."""

    def __init__(self) -> None:
        self._agent = create_agent(
            model=get_chat_model(),
            tools=[],
            system_prompt=PLANNER_PROMPT,
            response_format=Agent1ResearchPlan,
            name="agent1-research-planner",
        )

    def run(
        self,
        disease: str,
        target: str | None = None,
        top_n: int = 10,
        disease_id: str | None = None,
        target_id: str | None = None,
    ) -> Agent1ResearchPlan:
        if target_id and not target:
            raise ResearchPlannerError("--target-id requires --target")
        context = _verified_context(disease, target, top_n, disease_id, target_id)
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        model_name = (
            get_ollama_settings()[0]
            if provider == "ollama"
            else os.getenv("ANTHROPIC_MODEL", "unknown")
        )
        original_question = (
                f"研究 {disease} 與 {target} 的證據" if target else f"探索 {disease} 的前 {top_n} 個 targets"
            )
        request = {
            "original_question": original_question,
            "verified_context": context,
            "registered_tools": REGISTERED_TOOLS,
            "query_manifest_requirements": {
                "created_before_retrieval": True,
                "current_query_type": "evidence_directed" if target else "single_hop",
                "current_query_path": ["disease", "target", "evidence"] if target else ["disease", "target"],
                "allowed_target_strategies": ["specified_target"] if target else ["select_then_expand", "expand_all_top10"],
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_provider": provider,
            "model_name": model_name,
        }
        errors: list[str] = []
        for repair_count in range(3):
            prompt = json.dumps(request, ensure_ascii=False)
            if errors:
                prompt += "\n前一份計畫未通過驗證，請修正：\n" + errors[-1]
            try:
                result = self._agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
                plan = result.get("structured_response")
                if plan is None:
                    raise ValueError("LLM returned no structured_response")
                plan = Agent1ResearchPlan.model_validate(plan.model_dump())
                _validate_against_verified_context(plan, context)
                plan.generated_at = datetime.now(timezone.utc)
                plan.model_provider = provider
                plan.model_name = model_name
                plan.validation_repair_count = repair_count
                return plan
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        raise ResearchPlannerError(
            "Research Plan validation failed after 2 repair attempts: " + errors[-1]
        )
