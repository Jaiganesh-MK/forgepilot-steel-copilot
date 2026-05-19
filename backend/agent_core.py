from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PlantContext:
    index: int
    current: dict[str, Any]
    history: pd.DataFrame
    playbook: list[dict[str, Any]]
    similar_cases: list[dict[str, Any]]


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    message_type: str
    content: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentSignal:
    agent_name: str
    decision_area: str
    severity: str
    confidence: float
    message: str
    evidence: list[str] = field(default_factory=list)
    proposed_actions: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRecommendation:
    id: str
    timestamp: str
    agent_name: str
    decision_area: str
    action_summary: str
    action_deltas: dict[str, Any]
    confidence: float
    risk_level: str
    reasoning: str
    evidence: list[str]
    prerequisites: list[str]
    automation_mode: str
    approval_required: bool
    safety_notes: list[str]
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class BaseAgent:
    name = "BaseAgent"
    decision_area = "General"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        raise NotImplementedError
