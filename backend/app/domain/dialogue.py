from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DialogueIntent(StrEnum):
    QUESTION = "question"
    COMPARISON = "comparison"
    MODIFY = "modify_trip"
    PLAN = "plan_trip"
    SEARCH_ALTERNATIVES = "search_alternatives"
    SELECT_OFFER = "select_offer"
    CORRECTION = "correction"
    NEW_TRIP = "new_trip"
    CLARIFY = "clarify"


class ChangeOperation(StrEnum):
    SET = "set"
    CLEAR = "clear"
    ADD = "add"
    REMOVE = "remove"


class StateChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    operation: ChangeOperation
    value: Any = None


class SearchCriteriaDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["current", "other_destinations", "any", "hotels", "flights", "dates"] | None = None
    destination_query: str | None = None
    price_preference: Literal["cheaper", "more_expensive", "comfort", "any"] | None = None
    flexible_dates: bool = False
    missing_fields: list[str] = Field(default_factory=list)


class DialogueDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: DialogueIntent
    topic: str | None = None
    mode: Literal["min", "max"] | None = None
    changes: list[StateChange] = Field(default_factory=list)
    search: SearchCriteriaDecision | None = None
    references: list[str] = Field(default_factory=list)
    resume_pending: bool = False
    clarification: str | None = None
    answer: str | None = None
    narrative: str | None = None

    @model_validator(mode="after")
    def validate_intent_changes(self):
        if self.intent in (DialogueIntent.QUESTION, DialogueIntent.COMPARISON, DialogueIntent.CLARIFY) and self.changes:
            raise ValueError(f"{self.intent} must not change trip state")
        if self.intent in (DialogueIntent.MODIFY, DialogueIntent.SELECT_OFFER, DialogueIntent.CORRECTION) and not self.changes:
            raise ValueError(f"{self.intent} requires at least one state change")
        if self.intent == DialogueIntent.CLARIFY and not (self.clarification or self.answer):
            raise ValueError("clarify requires a clarification")
        return self
