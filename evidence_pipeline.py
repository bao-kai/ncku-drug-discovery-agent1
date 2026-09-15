"""Orchestration for the separate Agent 1 and Agent 2 implementations."""

from __future__ import annotations

import json

from agent1 import BioinformaticsEvidenceAgent, package_agent1_output
from agent2 import TargetEvidenceInterpreterAgent
from schemas import (
    DiseaseTargetEvidence,
    DiseaseTargetInput,
    EvidenceInterpretation,
)


def run_evidence_pipeline(
    disease: str, target: str
) -> tuple[DiseaseTargetEvidence, EvidenceInterpretation]:
    request = DiseaseTargetInput(disease=disease, target=target)
    evidence = BioinformaticsEvidenceAgent().run(request)
    interpretation = TargetEvidenceInterpreterAgent().run(evidence)
    return evidence, interpretation


def pipeline_as_json(disease: str, target: str) -> str:
    evidence, interpretation = run_evidence_pipeline(disease, target)
    agent1_output = package_agent1_output(evidence)
    return json.dumps(
        {
            **agent1_output.model_dump(mode="json"),
            "agent_2_interpretation": interpretation.model_dump(mode="json"),
        },
        ensure_ascii=False,
        indent=2,
    )
