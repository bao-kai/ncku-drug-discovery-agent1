"""Agent 2: constrained interpretation of Agent 1 evidence."""

from __future__ import annotations

from langchain.agents import create_agent

from model_factory import get_chat_model
from schemas import DiseaseTargetEvidence, EvidenceInterpretation


INTERPRETER_PROMPT = """
You are the Target Evidence Interpretation Agent in an early-stage biomedical
research system. You receive structured Open Targets evidence produced by a
separate deterministic collection agent.

Rules:
1. Interpret only the supplied evidence. Never invent publications, experiments,
   mechanisms, scores, or clinical claims.
2. Explain what the overall association score and each datasource score support.
2a. When datasource_evidence records are supplied, cite their exact drug,
    clinical-stage, trial/report, mechanism, direction, and reference fields.
    Treat null or absent fields as unavailable and never fill them from memory.
3. Clearly distinguish supporting evidence from weak/missing evidence.
4. Treat Open Targets scores as association evidence, not clinical confidence.
5. Do not calculate a confidence score. The research team has not defined the
   scoring framework, so confidence_score must remain null and
   confidence_score_status must be "pending_research_team_definition".
6. Keep the research-use disclaimer.
"""


class TargetEvidenceInterpreterAgent:
    """Interpret Agent 1 evidence without tools or invented scoring."""

    def __init__(self) -> None:
        self._agent = create_agent(
            model=get_chat_model(),
            tools=[],
            system_prompt=INTERPRETER_PROMPT,
            response_format=EvidenceInterpretation,
            name="target-evidence-interpreter",
        )

    def run(self, evidence: DiseaseTargetEvidence) -> EvidenceInterpretation:
        result = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Interpret this evidence JSON:\n"
                            + evidence.model_dump_json(indent=2)
                        ),
                    }
                ]
            }
        )
        structured = result.get("structured_response")
        if structured is None:
            raise RuntimeError("The interpretation agent returned no structured response.")
        return structured
