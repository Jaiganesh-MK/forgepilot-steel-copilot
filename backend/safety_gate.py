from __future__ import annotations

from typing import Any

from . import config
from .agent_core import AgentSignal, clamp, safe_float


def _is_nonzero(value: Any) -> bool:
    try:
        return abs(float(value)) > 1e-9
    except Exception:
        return bool(value)


def check_action_limits(actions: dict[str, Any]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    for key, value in actions.items():
        if key in config.ACTION_LIMITS and _is_nonzero(value):
            low, high = config.ACTION_LIMITS[key]
            numeric = safe_float(value)
            if numeric < low or numeric > high:
                ok = False
                notes.append(f"{key}={numeric} is outside one-step advisory envelope [{low}, {high}].")
            else:
                notes.append(f"{key}={numeric} is within one-step advisory envelope [{low}, {high}].")
        elif key in {"burden_distribution_change", "tapping_priority"} and value:
            notes.append(f"{key} requires procedural/operator confirmation.")
        elif key == "monitoring_action" and value:
            notes.append("Monitoring/logging action is eligible for automatic workflow execution.")
    return ok, notes


def contains_setpoint_change(actions: dict[str, Any]) -> bool:
    return any(key in actions and _is_nonzero(actions[key]) for key in config.SETPOINT_ACTION_KEYS)


def assign_automation(signal: AgentSignal, plant_level: str) -> tuple[str, bool, list[str]]:
    actions = signal.proposed_actions or {}
    in_limits, notes = check_action_limits(actions)
    has_setpoint = contains_setpoint_change(actions)
    if not actions:
        return "NO_ACTION_MONITOR_ONLY", False, ["No setpoint or workflow action proposed."]
    if has_setpoint:
        if not in_limits:
            notes.append("Blocked from direct action because one or more proposals exceed configured advisory envelope.")
            return "HUMAN_APPROVAL_REQUIRED_LIMIT_CHECK", True, notes
        if not config.ALLOW_DIRECT_SETPOINT_ACTIONS:
            notes.append("Default POC posture: process setpoint changes are advisory only and require operator approval.")
            return "HUMAN_APPROVAL_REQUIRED", True, notes
        if plant_level in {"critical", "high"}:
            notes.append("High plant risk prevents direct setpoint action even when direct-action mode is enabled.")
            return "HUMAN_APPROVAL_REQUIRED_HIGH_RISK", True, notes
        if signal.confidence >= config.DIRECT_ACTION_CONFIDENCE_THRESHOLD:
            notes.append("Direct-action candidate because confidence exceeds threshold and proposed delta is within limits. Not executed in POC.")
            return "DIRECT_ACTION_CANDIDATE_NOT_EXECUTED_IN_POC", False, notes
        notes.append("Confidence is below direct-action threshold; operator approval is required.")
        return "HUMAN_APPROVAL_REQUIRED_LOW_CONFIDENCE", True, notes
    if "monitoring_action" in actions and signal.confidence >= 0.65:
        notes.append("Auto-executable as non-control workflow: log, notify, validate, or increase monitoring.")
        return "AUTO_EXECUTE_INFORMATIONAL_WORKFLOW", False, notes
    return "OPERATOR_REVIEW", True, notes


def plant_risk_score(row: dict[str, Any]) -> float:
    thermal = safe_float(row.get("thermal_risk_score"))
    perm = safe_float(row.get("permeability_risk_score"))
    quality = safe_float(row.get("quality_risk_score"))
    return round(clamp(0.4 * thermal + 0.4 * perm + 0.2 * quality, 0, 100), 1)


def plant_risk_level(row: dict[str, Any]) -> str:
    score = plant_risk_score(row)
    if score >= 70:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 30:
        return "medium"
    return "low"
