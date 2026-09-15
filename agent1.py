"""Agent 1: deterministic disease-target evidence collection."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from query_opentargets import discover_disease_targets, get_disease_target_evidence
from schemas import (
    Agent1DiscoveryRunOutput,
    Agent1RunOutput,
    DiseaseDiscoveryInput,
    DiseaseTargetEvidence,
    DiseaseTargetInput,
    ExpertReviewView,
    SynthesizerInterpretation,
    TargetDiscoveryResult,
)


class BioinformaticsEvidenceAgent:
    """Collect and normalize Open Targets evidence for one disease-target pair."""

    def run(
        self,
        request: DiseaseTargetInput,
        *,
        max_evidence_records: int = 500,
    ) -> DiseaseTargetEvidence:
        raw = get_disease_target_evidence(
            request.disease,
            request.target,
            max_evidence_records=max_evidence_records,
        )
        return DiseaseTargetEvidence(**raw)

    def discover_targets(
        self, request: DiseaseDiscoveryInput
    ) -> TargetDiscoveryResult:
        raw = discover_disease_targets(request.disease, request.top_n)
        return TargetDiscoveryResult(**raw)


def package_agent1_output(
    evidence: DiseaseTargetEvidence,
    *,
    expert_review_view: ExpertReviewView | None = None,
    model_interpretation: SynthesizerInterpretation | None = None,
) -> Agent1RunOutput:
    """Attach strict provenance metadata to evidence produced by Agent 1."""
    return Agent1RunOutput(
        producer="agent1",
        run_id=uuid4(),
        generated_at=datetime.now(timezone.utc),
        agent_1_evidence=evidence,
        expert_review_view=expert_review_view,
        model_interpretation=model_interpretation,
    )


def package_discovery_output(
    result: TargetDiscoveryResult,
) -> Agent1DiscoveryRunOutput:
    return Agent1DiscoveryRunOutput(
        producer="agent1",
        run_id=uuid4(),
        generated_at=datetime.now(timezone.utc),
        target_discovery=result,
    )
