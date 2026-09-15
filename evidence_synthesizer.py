"""Agent 1 internal sub-agent for evidence inventory interpretation."""

from __future__ import annotations

from langchain.agents import create_agent

from model_factory import get_chat_model
from schemas import ExpertReviewView, SynthesizerInterpretation


SYNTHESIZER_PROMPT = """
You are the Evidence Synthesizer, an internal sub-agent of Agent 1.
You have no tools and may only interpret the supplied deterministic expert review JSON.

Separate database facts from interpretation. Never invent trial outcomes, mechanisms,
publications, efficacy, safety, scores, or missing fields. Evidence record counts indicate
how many records exist, not how many positive studies exist. Open Targets scores are not
success probabilities. Your purpose is to help biomedical domain experts understand what
the collected dataset contains and decide how it should later be classified in a database.
Use cautious Traditional Chinese. Flag ambiguity and formulate questions instead of making
unsupported medical conclusions.
"""


class EvidenceSynthesizerSubAgent:
    def __init__(self) -> None:
        self._agent = create_agent(
            model=get_chat_model(),
            tools=[],
            system_prompt=SYNTHESIZER_PROMPT,
            response_format=SynthesizerInterpretation,
            name="agent1-evidence-synthesizer",
        )

    def run(self, review: ExpertReviewView) -> SynthesizerInterpretation:
        # Keep the model input bounded. Complete IDs and raw records remain in
        # the deterministic output for experts and are not needed for a first
        # inventory interpretation.
        payload = review.model_dump(mode="json")
        for source in payload.get("datasource_reviews", []):
            for field in source.get("field_profiles", []):
                field["example_values"] = field.get("example_values", [])[:2]
            for drug in source.get("drug_groups", []):
                drug["report_ids"] = drug.get("report_ids", [])[:5]
                drug["literature_ids"] = drug.get("literature_ids", [])[:5]
                drug["stop_reason_examples"] = drug.get("stop_reason_examples", [])[:2]
        result = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "整理以下專家審查資料目錄：\n"
                        + __import__("json").dumps(payload, ensure_ascii=False),
                    }
                ]
            }
        )
        structured = result.get("structured_response")
        if structured is None:
            raise RuntimeError("Evidence Synthesizer returned no structured response.")
        return structured
