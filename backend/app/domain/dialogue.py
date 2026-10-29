from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DialogueIntent(StrEnum):
    QUESTION = "question"
    COMPARISON = "comparison"
    MODIFY = "modify_trip"
    PLAN = "plan_trip"
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


class DialogueDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: DialogueIntent
    topic: str | None = None
    mode: Literal["min", "max"] | None = None
    changes: list[StateChange] = Field(default_factory=list)
    clarification: str | None = None
    answer: str | None = None
    narrative: str | None = None

    @model_validator(mode="after")
    def validate_intent_changes(self):
        if self.intent in (DialogueIntent.QUESTION, DialogueIntent.COMPARISON, DialogueIntent.CLARIFY) and self.changes:
            raise ValueError(f"{self.intent} must not change trip state")
        if self.intent == DialogueIntent.MODIFY and not self.changes:
            raise ValueError("modify_trip requires at least one state change")
        if self.intent == DialogueIntent.CLARIFY and not (self.clarification or self.answer):
            raise ValueError("clarify requires a clarification")
        return self
