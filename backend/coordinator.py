from __future__ import annotations

import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

from . import config
from .agent_core import AgentMessage, AgentRecommendation, AgentSignal, PlantContext, clamp, safe_float
from .agents import (
    BlastTemperatureAgent,
    BurdenDistributionAgent,
    CokeRateAgent,
    FuelRateAgent,
    OxygenEnrichmentAgent,
    PCIAgent,
    PermeabilityAgent,
    QualityAgent,
    TappingAgent,
    ThermalStateAgent,
    TopPressureAgent,
    WindVolumeAgent,
)
from .llm import ReasoningSynthesizer
from .safety_gate import assign_automation, plant_risk_level


class AgentCoordinator:
    name = "CoordinatorAgent"

    def __init__(self) -> None:
        self.agents = [
            ThermalStateAgent(),
            PermeabilityAgent(),
            WindVolumeAgent(),
            PCIAgent(),
            CokeRateAgent(),
            FuelRateAgent(),
            OxygenEnrichmentAgent(),
            BlastTemperatureAgent(),
            TopPressureAgent(),
            BurdenDistributionAgent(),
            TappingAgent(),
            QualityAgent(),
        ]
        self.synthesizer = ReasoningSynthesizer()
        self._last_llm_agent_requested: list[str] = []
        self._last_llm_agent_reviewed: list[str] = []

    def run(
        self,
        context: PlantContext,
        include_llm_summary: bool = True,
        include_llm_agents: bool = False,
        llm_agent_max_agents: int | None = None,
        llm_agent_timeout_seconds: float | None = None,
        llm_agent_max_workers: int | None = None,
    ) -> dict[str, Any]:
        deterministic_signals = [agent.evaluate(context) for agent in self.agents]
        llm_agent_requested = bool(include_llm_agents or config.USE_LLM_AGENTS)
        signals = (
            self._review_signals_with_llm(
                deterministic_signals,
                context,
                max_agents=llm_agent_max_agents,
                timeout_seconds=llm_agent_timeout_seconds,
                max_workers=llm_agent_max_workers,
            )
            if llm_agent_requested
            else deterministic_signals
        )
        self._last_llm_agent_requested = [s.agent_name for s in signals if (s.metadata or {}).get("llm_review_requested")]
        self._last_llm_agent_reviewed = [s.agent_name for s in signals if (s.metadata or {}).get("llm_used")]

        risk_level = plant_risk_level(context.current)
        messages = self._build_messages(signals)
        recommendations = self._build_recommendations(signals, context, risk_level)
        coordinated = self._build_coordinated_recommendation(signals, context, risk_level)
        if coordinated:
            recommendations.insert(0, coordinated)
        recommendations = sorted(
            recommendations,
            key=lambda r: (self._risk_sort_value(r.risk_level), r.confidence, 1 if r.agent_name == self.name else 0),
            reverse=True,
        )
        payload = {
            "dataset_index": context.index,
            "timestamp": context.current.get("timestamp"),
            "state": self._state_summary(context),
            "signals": [signal.to_dict() for signal in signals],
            "agent_messages": [message.to_dict() for message in messages],
            "recommendations": [rec.to_dict() for rec in recommendations],
            "similar_cases": context.similar_cases,
            "playbook_matches": self._playbook_matches(context, signals),
            "llm_agent_reviews": self._llm_agent_reviews(signals),
            "architecture_status": self._architecture_status(llm_agent_requested=llm_agent_requested),
        }
        payload["executive_summary"] = self.synthesizer.synthesize(payload) if include_llm_summary else self.synthesizer.deterministic_summary(payload)
        payload["summary_source"] = "openrouter_or_fallback" if include_llm_summary else "deterministic"
        payload["architecture_status"] = self._architecture_status(llm_agent_requested=llm_agent_requested)
        return payload

    def llm_status(self) -> dict[str, Any]:
        return self.synthesizer.status()

    def _review_signals_with_llm(
        self,
        signals: list[AgentSignal],
        context: PlantContext,
        max_agents: int | None = None,
        timeout_seconds: float | None = None,
        max_workers: int | None = None,
    ) -> list[AgentSignal]:
        if not self.synthesizer.enabled:
            return signals

        selected = self._select_signals_for_llm_review(signals, context, max_agents=max_agents)
        if not selected:
            return signals

        selected = [self._mark_llm_agent_selected(signal) for signal in selected]
        by_agent = {signal.agent_name: signal for signal in signals}
        for signal in selected:
            by_agent[signal.agent_name] = signal
        budget_seconds = max(2.0, float(timeout_seconds or config.LLM_AGENT_TOTAL_TIMEOUT_SECONDS))

        if config.LLM_AGENT_BATCH_REVIEWS:
            # Efficient free-model path: one OpenRouter request reviews all selected agents.
            # This avoids N parallel requests, which quickly hits free-model quotas and creates timeouts.
            reviewed = self.synthesizer.review_agent_signals_batch(selected, context, timeout_seconds=budget_seconds)
            for signal in reviewed:
                by_agent[signal.agent_name] = signal
            return [by_agent[signal.agent_name] for signal in signals]

        worker_count = min(max(1, int(max_workers or config.LLM_AGENT_MAX_WORKERS)), len(selected))
        executor = ThreadPoolExecutor(max_workers=worker_count)
        futures = {executor.submit(self.synthesizer.review_agent_signal, signal, context): signal.agent_name for signal in selected}
        done, pending = wait(futures, timeout=budget_seconds)

        for future in done:
            agent_name = futures[future]
            try:
                by_agent[agent_name] = future.result()
            except Exception as exc:
                by_agent[agent_name] = self._mark_llm_agent_fallback(by_agent[agent_name], str(exc))

        for future in pending:
            agent_name = futures[future]
            future.cancel()
            by_agent[agent_name] = self._mark_llm_agent_fallback(
                by_agent[agent_name],
                f"LLM review exceeded specialist-agent budget of {budget_seconds:.1f}s; deterministic scaffold retained.",
            )

        executor.shutdown(wait=False, cancel_futures=True)
        return [by_agent[signal.agent_name] for signal in signals]

    @staticmethod
    def _mark_llm_agent_selected(signal: AgentSignal) -> AgentSignal:
        metadata = dict(signal.metadata or {})
        threshold = float(config.LLM_AGENT_CONFIDENCE_THRESHOLD)
        if signal.confidence < threshold:
            selection_reason = f"deterministic_confidence_{signal.confidence:.2f}_below_{threshold:.2f}"
        elif signal.proposed_actions:
            selection_reason = "active_recommendation"
        else:
            selection_reason = "risk_or_mode_selection"
        metadata.update(
            {
                "llm_review_requested": True,
                "llm_selection_reason": selection_reason,
                "llm_confidence_threshold": threshold,
            }
        )
        return AgentSignal(
            signal.agent_name,
            signal.decision_area,
            signal.severity,
            signal.confidence,
            signal.message,
            signal.evidence,
            signal.proposed_actions,
            signal.dependencies,
            signal.prerequisites,
            signal.risk_tags,
            metadata,
        )

    @staticmethod
    def _mark_llm_agent_fallback(signal: AgentSignal, error: str) -> AgentSignal:
        metadata = dict(signal.metadata or {})
        metadata.update({"llm_used": False, "llm_error": error, "decision_basis": "deterministic_rules_fallback"})
        return AgentSignal(
            signal.agent_name,
            signal.decision_area,
            signal.severity,
            signal.confidence,
            signal.message,
            signal.evidence,
            signal.proposed_actions,
            signal.dependencies,
            signal.prerequisites,
            signal.risk_tags,
            metadata,
        )

    def _select_signals_for_llm_review(self, signals: list[AgentSignal], context: PlantContext, max_agents: int | None = None) -> list[AgentSignal]:
        mode = config.LLM_AGENT_MODE
        min_severity_score = self._risk_sort_value(config.LLM_AGENT_MIN_SEVERITY)
        plant_score = safe_float(context.current.get("plant_risk_score"))
        event_label = str(context.current.get("event_label", "")).lower()
        threshold = float(config.LLM_AGENT_CONFIDENCE_THRESHOLD)
        low_confidence_selected: list[AgentSignal] = []
        regular_selected: list[AgentSignal] = []

        for signal in signals:
            severity_score = self._risk_sort_value(signal.severity)
            has_actions = bool(signal.proposed_actions)
            low_confidence = bool(config.LLM_AGENT_REVIEW_LOW_CONFIDENCE and signal.confidence < threshold)

            # Confidence-gate override: any specialist agent below the deterministic
            # confidence threshold is eligible for OpenRouter review even if the
            # plant state is normal / low risk and even if the agent has no action.
            if low_confidence:
                low_confidence_selected.append(signal)
                continue

            if mode == "all":
                include = True
            elif mode == "high_risk_only":
                include = severity_score >= self._risk_sort_value("high") or plant_score >= 50
            else:
                include = has_actions or severity_score >= min_severity_score

            if event_label == "normal" and plant_score < 30 and not config.LLM_AGENT_CALL_ON_NORMAL_LOW_RISK and not has_actions:
                include = False
            if include:
                regular_selected.append(signal)

        def sort_key(signal: AgentSignal) -> tuple[int, int, float]:
            return (self._risk_sort_value(signal.severity), 1 if signal.proposed_actions else 0, 1.0 - float(signal.confidence))

        low_confidence_selected = sorted(low_confidence_selected, key=sort_key, reverse=True)
        regular_selected = sorted(regular_selected, key=sort_key, reverse=True)
        review_limit = max(1, int(max_agents or config.LLM_AGENT_MAX_AGENTS))

        if config.LLM_AGENT_LOW_CONFIDENCE_OVERRIDES_LIMIT:
            # Return all sub-threshold agents. The max_agents setting limits only
            # extra active/high-risk agents above the threshold. This implements
            # the desired behavior: confidence < threshold -> call LLM every time
            # specialist review is requested.
            remaining_regular_slots = max(0, review_limit - len(low_confidence_selected))
            return low_confidence_selected + regular_selected[:remaining_regular_slots]

        selected = low_confidence_selected + regular_selected
        return selected[:review_limit]

    def _architecture_status(self, llm_agent_requested: bool = False) -> dict[str, Any]:
        status = self.synthesizer.status()
        if config.OPENROUTER_INCLUDE_KEY_HEALTH_CHECK:
            try:
                status["openrouter_key_health"] = self.synthesizer.key_health()
            except Exception as exc:
                status["openrouter_key_health"] = {"ok": False, "error": str(exc)}
        status.update(
            {
                "direct_setpoint_action_enabled": config.ALLOW_DIRECT_SETPOINT_ACTIONS,
                "direct_action_confidence_threshold": config.DIRECT_ACTION_CONFIDENCE_THRESHOLD,
                "default_safety_posture": "Setpoint changes require operator approval unless ALLOW_DIRECT_SETPOINT_ACTIONS=true.",
                "llm_specialist_agents_requested": llm_agent_requested,
                "llm_specialist_agents_active": bool(llm_agent_requested and self.synthesizer.enabled),
                "llm_specialist_agents_requested_list": self._last_llm_agent_requested,
                "llm_specialist_agents_reviewed": self._last_llm_agent_reviewed,
                "llm_agent_total_timeout_seconds": config.LLM_AGENT_TOTAL_TIMEOUT_SECONDS,
                "llm_agent_batch_reviews": config.LLM_AGENT_BATCH_REVIEWS,
                "llm_agent_batch_size": config.LLM_AGENT_BATCH_SIZE,
                "llm_agent_confidence_threshold": config.LLM_AGENT_CONFIDENCE_THRESHOLD,
                "llm_agent_review_low_confidence": config.LLM_AGENT_REVIEW_LOW_CONFIDENCE,
                "llm_agent_low_confidence_overrides_limit": config.LLM_AGENT_LOW_CONFIDENCE_OVERRIDES_LIMIT,
                "llm_agent_note": "Specialist agents run deterministic evidence first. When specialist LLM review is requested, selected sub-threshold agents are reviewed through a batched OpenRouter JSON request before coordination.",
            }
        )
        return status

    @staticmethod
    def _risk_sort_value(level: str) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(str(level).lower(), 0)

    @staticmethod
    def _state_summary(context: PlantContext) -> dict[str, Any]:
        row = context.current
        keys = [
            "timestamp", "furnace_id", "shift", "crew", "operating_mode", "event_label", "sensor_quality_flag",
            "wind_volume_nm3_min", "hot_blast_temp_c", "oxygen_enrichment_pct", "top_pressure_kpa",
            "pci_rate_kg_thm", "coke_rate_kg_thm", "total_fuel_rate_kg_thm", "gas_utilization_pct",
            "pressure_drop_kpa", "permeability_index", "raceway_adiabatic_flame_temp_c", "hearth_liquid_level_index",
            "hot_metal_temp_c", "hot_metal_si_pct", "production_tph", "thermal_state_index", "thermal_risk_score",
            "permeability_risk_score", "quality_risk_score", "plant_risk_score", "plant_risk_level", "health_score",
            "operating_summary",
        ]
        return {key: row.get(key) for key in keys if key in row}

    def _build_messages(self, signals: list[AgentSignal]) -> list[AgentMessage]:
        messages: list[AgentMessage] = []
        active = [s for s in signals if s.proposed_actions or s.severity in {"medium", "high", "critical"}]
        for signal in active:
            for dependency in signal.dependencies:
                messages.append(AgentMessage(signal.agent_name, dependency, "coordination_request", f"{signal.decision_area}: {signal.message}", signal.confidence))
            metadata = signal.metadata or {}
            if metadata.get("llm_used"):
                messages.append(
                    AgentMessage(
                        signal.agent_name,
                        "OpenRouterLLM",
                        "specialist_review_request",
                        f"{signal.agent_name} sent evidence/actions to OpenRouter for domain review.",
                        signal.confidence,
                    )
                )
                messages.append(
                    AgentMessage(
                        "OpenRouterLLM",
                        signal.agent_name,
                        "specialist_review_response",
                        f"LLM specialist review returned via {metadata.get('llm_model')}; signal basis is hybrid.",
                        signal.confidence,
                    )
                )
            elif metadata.get("llm_error"):
                messages.append(
                    AgentMessage(
                        "OpenRouterLLM",
                        signal.agent_name,
                        "specialist_review_fallback",
                        f"LLM review unavailable; {signal.agent_name} retained deterministic fallback.",
                        signal.confidence,
                    )
                )
        by_name = {s.agent_name: s for s in signals}
        thermal = by_name.get("ThermalStateAgent")
        perm = by_name.get("PermeabilityAgent")
        if thermal and perm and thermal.severity in {"high", "critical"} and perm.severity in {"high", "critical"}:
            messages.append(
                AgentMessage(
                    self.name,
                    "AllSetpointAgents",
                    "constraint_resolution",
                    "Thermal and permeability stress are both active. Prioritize stability: avoid aggressive wind increases, coordinate PCI reduction with coke/oxygen/temperature support.",
                    min(thermal.confidence, perm.confidence),
                )
            )
        quality = by_name.get("QualityAgent")
        if quality and quality.severity in {"medium", "high", "critical"}:
            messages.append(
                AgentMessage(
                    "QualityAgent",
                    "SafetyGateAgent",
                    "validation_request",
                    "Quality excursion or data-quality flag requires lab/sensor validation before setpoint action.",
                    quality.confidence,
                )
            )
        return messages

    def _build_recommendations(self, signals: list[AgentSignal], context: PlantContext, risk_level: str) -> list[AgentRecommendation]:
        recommendations: list[AgentRecommendation] = []
        for signal in signals:
            if not signal.proposed_actions:
                continue
            automation_mode, approval_required, safety_notes = assign_automation(signal, risk_level)
            if (signal.metadata or {}).get("llm_used"):
                safety_notes.insert(0, "This recommendation was refined by an OpenRouter specialist-agent review, then checked by deterministic safety logic.")
            elif (signal.metadata or {}).get("llm_error"):
                safety_notes.insert(0, "LLM specialist review was unavailable; deterministic fallback logic is shown.")
            recommendations.append(
                AgentRecommendation(
                    self._make_id(context.index, signal.agent_name, signal.proposed_actions),
                    str(context.current.get("timestamp")),
                    signal.agent_name,
                    signal.decision_area,
                    self._action_summary(signal.proposed_actions),
                    signal.proposed_actions,
                    round(float(signal.confidence), 3),
                    signal.severity,
                    signal.message,
                    signal.evidence,
                    signal.prerequisites,
                    automation_mode,
                    approval_required,
                    safety_notes,
                )
            )
        if not recommendations:
            signal = AgentSignal(
                self.name,
                "Overall operation",
                "low",
                0.76,
                "All core agents recommend maintaining current operation and continuing standard monitoring.",
                ["No active agent proposed a process setpoint or workflow change."],
                {"monitoring_action": "Continue normal surveillance and shift logging"},
                [],
                [],
                ["maintain"],
            )
            automation_mode, approval_required, safety_notes = assign_automation(signal, risk_level)
            recommendations.append(
                AgentRecommendation(
                    self._make_id(context.index, signal.agent_name, signal.proposed_actions),
                    str(context.current.get("timestamp")),
                    signal.agent_name,
                    signal.decision_area,
                    self._action_summary(signal.proposed_actions),
                    signal.proposed_actions,
                    signal.confidence,
                    "low",
                    signal.message,
                    signal.evidence,
                    signal.prerequisites,
                    automation_mode,
                    approval_required,
                    safety_notes,
                )
            )
        return recommendations

    def _build_coordinated_recommendation(self, signals: list[AgentSignal], context: PlantContext, risk_level: str) -> AgentRecommendation | None:
        active = [s for s in signals if s.proposed_actions and s.severity in {"medium", "high", "critical"}]
        if len(active) < 2:
            return None
        actions, action_sources = self._aggregate_actions(active)
        if not actions:
            return None
        confidence = round(max(0.55, min(s.confidence for s in active) * 0.94), 3)
        severity = self._dominant_severity(active)
        signal = AgentSignal(
            self.name,
            "Coordinated operating response",
            severity,
            confidence,
            self._coordinated_reasoning(active, action_sources),
            self._coordinated_evidence(active),
            actions,
            [s.agent_name for s in active],
            self._merge_prerequisites(active),
            sorted({tag for s in active for tag in s.risk_tags}),
            {"decision_basis": "coordinator_aggregated_hybrid_signals" if any((s.metadata or {}).get("llm_used") for s in active) else "coordinator_aggregated_deterministic_signals"},
        )
        automation_mode, approval_required, safety_notes = assign_automation(signal, risk_level)
        safety_notes.insert(0, "Coordinated recommendation consolidates multiple agent proposals; apply as a package or explicitly reject/modify.")
        return AgentRecommendation(
            self._make_id(context.index, self.name, actions),
            str(context.current.get("timestamp")),
            signal.agent_name,
            signal.decision_area,
            self._action_summary(actions),
            actions,
            confidence,
            severity,
            signal.message,
            signal.evidence,
            signal.prerequisites,
            automation_mode,
            approval_required,
            safety_notes,
        )

    @staticmethod
    def _dominant_severity(signals: list[AgentSignal]) -> str:
        order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return max(signals, key=lambda s: order.get(s.severity, 0)).severity

    def _aggregate_actions(self, signals: list[AgentSignal]) -> tuple[dict[str, Any], dict[str, list[str]]]:
        values: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
        texts: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        sources: dict[str, list[str]] = defaultdict(list)
        severity_weight = {"critical": 1.25, "high": 1.0, "medium": 0.75, "low": 0.4}
        for signal in signals:
            for key, value in signal.proposed_actions.items():
                sources[key].append(signal.agent_name)
                if key in config.ACTION_LIMITS:
                    values[key].append((safe_float(value), signal.agent_name, signal.severity))
                else:
                    texts[key].append((str(value), signal.agent_name, signal.severity))
        actions: dict[str, Any] = {}
        for key, entries in values.items():
            positives = [v for v, _, _ in entries if v > 0]
            negatives = [v for v, _, _ in entries if v < 0]
            if positives and negatives:
                permeability_entries = [v for v, agent, _ in entries if agent in {"PermeabilityAgent", "WindVolumeAgent"}]
                if permeability_entries:
                    chosen = sum(permeability_entries) / len(permeability_entries)
                else:
                    chosen = sum(v * severity_weight.get(sev, 1.0) for v, _, sev in entries) / max(sum(severity_weight.get(sev, 1.0) for _, _, sev in entries), 1.0)
            else:
                chosen = max(entries, key=lambda item: abs(item[0]))[0]
            if abs(chosen) < 1e-9:
                continue
            low, high = config.ACTION_LIMITS[key]
            actions[key] = round(clamp(chosen, low, high), 2)
        order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        for key, entries in texts.items():
            selected = max(entries, key=lambda item: order.get(item[2], 0))
            actions[key] = selected[0]
        return actions, dict(sources)

    @staticmethod
    def _merge_prerequisites(signals: list[AgentSignal]) -> list[str]:
        merged: list[str] = []
        for signal in signals:
            for item in signal.prerequisites:
                if item not in merged:
                    merged.append(item)
        return merged[:8]

    @staticmethod
    def _coordinated_evidence(signals: list[AgentSignal]) -> list[str]:
        evidence: list[str] = []
        for signal in signals:
            for item in signal.evidence[:3]:
                entry = f"{signal.agent_name}: {item}"
                if entry not in evidence:
                    evidence.append(entry)
        return evidence[:12]

    @staticmethod
    def _coordinated_reasoning(signals: list[AgentSignal], action_sources: dict[str, list[str]]) -> str:
        active_names = ", ".join(s.agent_name for s in signals)
        source_fragments = [f"{key} from {', '.join(sorted(set(agents)))}" for key, agents in action_sources.items()]
        hybrid_note = " Some source signals were reviewed by OpenRouter specialist agents." if any((s.metadata or {}).get("llm_used") for s in signals) else ""
        return f"Integrated response based on active agents: {active_names}. The package resolves overlapping proposals conservatively: {'; '.join(source_fragments)}.{hybrid_note} Apply only after plant-standard validation and operator review."

    @staticmethod
    def _action_summary(actions: dict[str, Any]) -> str:
        if not actions:
            return "Maintain current operation."
        labels = {
            "wind_volume_delta_nm3_min": "wind volume",
            "pci_delta_kg_thm": "PCI",
            "coke_rate_delta_kg_thm": "coke rate",
            "oxygen_enrichment_delta_pct": "oxygen enrichment",
            "blast_temp_delta_c": "hot blast temperature",
            "top_pressure_delta_kpa": "top pressure",
            "burden_distribution_change": "burden distribution",
            "tapping_priority": "tapping priority",
            "monitoring_action": "workflow",
        }
        units = {
            "wind_volume_delta_nm3_min": "Nm3/min",
            "pci_delta_kg_thm": "kg/thm",
            "coke_rate_delta_kg_thm": "kg/thm",
            "oxygen_enrichment_delta_pct": "pct-pt",
            "blast_temp_delta_c": "C",
            "top_pressure_delta_kpa": "kPa",
        }
        parts: list[str] = []
        for key, value in actions.items():
            label = labels.get(key, key)
            if key in units:
                sign = "+" if safe_float(value) > 0 else ""
                parts.append(f"{label} {sign}{value} {units[key]}")
            else:
                parts.append(f"{label}: {value}")
        return "; ".join(parts)

    @staticmethod
    def _make_id(index: int, agent_name: str, actions: dict[str, Any]) -> str:
        raw = f"{index}|{agent_name}|{actions}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"REC-{index}-{agent_name.replace('Agent', '').upper()}-{digest}"

    @staticmethod
    def _playbook_matches(context: PlantContext, signals: list[AgentSignal]) -> list[dict[str, Any]]:
        tags = " ".join(tag for signal in signals for tag in signal.risk_tags).lower()
        rows = []
        for row in context.playbook:
            situation = str(row.get("Situation", "")).lower()
            score = 0
            if "cold" in tags and "cold" in situation:
                score += 2
            if "hot" in tags and "hot" in situation:
                score += 2
            if "permeability" in tags and "permeability" in situation:
                score += 2
            if "quality" in tags and "quality" in situation:
                score += 2
            if score > 0:
                item = dict(row)
                item["match_score"] = score
                rows.append(item)
        return sorted(rows, key=lambda item: item["match_score"], reverse=True)[:3]

    @staticmethod
    def _llm_agent_reviews(signals: list[AgentSignal]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for signal in signals:
            metadata = signal.metadata or {}
            if metadata.get("llm_review_requested") or metadata.get("llm_used") or metadata.get("llm_error"):
                rows.append(
                    {
                        "agent_name": signal.agent_name,
                        "decision_area": signal.decision_area,
                        "deterministic_confidence": signal.confidence,
                        "llm_review_requested": bool(metadata.get("llm_review_requested")),
                        "llm_selection_reason": metadata.get("llm_selection_reason"),
                        "llm_used": bool(metadata.get("llm_used")),
                        "llm_model": metadata.get("llm_model"),
                        "llm_key": metadata.get("llm_key"),
                        "llm_error": metadata.get("llm_error"),
                        "decision_basis": metadata.get("decision_basis"),
                        "reasoning_addendum": metadata.get("llm_reasoning_addendum"),
                    }
                )
        return rows
