"""Explicit human research scope for one question/run; not source synonym proof."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class HumanResearchClarification(BaseModel):
    """An explicit interpretation of one original question, bound to its run."""
    original_question: str = Field(min_length=1)
    confirmation_question: str = Field(min_length=1)
    user_response: str = Field(min_length=1)
    run_id: str | None = None

    def for_run(self, question, run_id):
        if question != self.original_question:
            raise ValueError('Research clarification belongs to a different question')
        if self.run_id is not None and self.run_id != run_id:
            raise ValueError('Research clarification belongs to a different run')
        return self.model_copy(deep=True,update={'run_id':run_id})

    def planner_context(self):
        return ('Run-scoped explicit HUMAN research clarification. Preserve original_question and '
                'all its other requirements. Apply only this supplied clarification for this run; '
                'it is not source evidence or an entity identity confirmation.\n'+self.model_dump_json(indent=2))


class HumanConfirmedEntity(BaseModel):
    entity_type: Literal['disease', 'target']
    mentions: list[str] = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)


class HumanConfirmedEntityContext(BaseModel):
    original_question: str = Field(min_length=1)
    confirmation_question: str = Field(min_length=1)
    user_response: str = Field(min_length=1)
    approval_basis: Literal['explicit_user_scope_confirmation']
    entities: list[HumanConfirmedEntity] = Field(min_length=1)
    run_id: str | None = None

    @model_validator(mode='after')
    def validate_mentions(self):
        seen = {}
        for entity in self.entities:
            for mention in entity.mentions:
                if not mention.strip() or mention not in self.original_question:
                    raise ValueError('Confirmed mention must occur literally in original_question')
                identity = (entity.entity_type, entity.entity_id, entity.source_name)
                if mention in seen and seen[mention] != identity:
                    raise ValueError('Contradictory human scope confirmations for the same mention')
                seen[mention] = identity
        return self

    def for_run(self, question, run_id):
        if question != self.original_question:
            raise ValueError('Human confirmation belongs to a different question')
        if self.run_id is not None and self.run_id != run_id:
            raise ValueError('Human confirmation belongs to a different run')
        return self.model_copy(deep=True, update={'run_id': run_id})

    def binding_for(self, entity):
        for confirmed in self.entities:
            if entity.mention in confirmed.mentions:
                if entity.entity_type != confirmed.entity_type:
                    raise ValueError('Planned entity type contradicts human confirmation')
                return confirmed
        return None

    def validate_plan(self, plan):
        if plan.original_question != self.original_question or not self.run_id:
            raise ValueError('Human confirmation requires its exact question and bound run')
        for entity in plan.entities:
            confirmed = self.binding_for(entity)
            if confirmed is None:
                continue
            if entity.entity_id is not None and entity.entity_id != confirmed.entity_id:
                raise ValueError('Planned entity ID contradicts human confirmation')
            if entity.canonical_name is not None and entity.canonical_name.casefold() != confirmed.source_name.casefold():
                raise ValueError('Planned canonical name contradicts human confirmation')

    def apply_bindings(self, plan):
        self.validate_plan(plan)
        bound = []
        for entity in plan.entities:
            confirmed = self.binding_for(entity)
            if confirmed is not None:
                entity.entity_id = confirmed.entity_id
                entity.canonical_name = None
                entity.resolution_status = 'unresolved'  # Native source check is still required.
                bound.append(entity.entity_ref)
        return bound

    def planner_context(self):
        return ('Run-scoped HUMAN research-scope confirmation follows. Preserve original_question. '
                'This explicitly confirms the listed literal mentions refer to the same listed research scope '
                'for this question/run only; it is not source evidence of translation/synonymy. '
                'Plan operations under this confirmed scope; if separate entities retain these literal mentions, '
                'they receive the same confirmed binding in Python. Do not add an unconfirmed independent '
                'translation lookup for that already confirmed scope. Keep resolution operations to verify '
                'the native ID and source_name. Leave model-generated entity_id/canonical_name null; Python '
                'binds the confirmation and the source must still verify identity. Other entities are unconfirmed.\n'
                + self.model_dump_json(indent=2))
