from __future__ import annotations

from dataclasses import replace
from typing import Any, TypedDict

from . import config
from .agent_core import AgentSignal, PlantContext, safe_float
from .llm import ReasoningSynthesizer


class GraphState(TypedDict, total=False):
    context: PlantContext
    seed_signals: list[AgentSignal]
    reviewed_signals: list[AgentSignal]
    final_signals: list[AgentSignal]
    graph_trace: list[dict[str, Any]]
    error: str | None


class LangGraphSpecialistOrchestrator:
    """LangGraph-based LLM-first specialist reasoning workflow.

    This is deliberately narrow: it replaces only the specialist LLM reasoning
    layer. The existing deterministic rules remain available for fallback and
    downstream safety validation/coordinator aggregation.
    """

    def __init__(self, synthesizer: ReasoningSynthesizer, agents: list[Any]) -> None:
        self.synthesizer = synthesizer
        self.agents = agents
        self._compiled_graph = None
        self.available = False
        self.import_error: str | None = None
        try:
            from langgraph.graph import END, START, StateGraph  # type: ignore

            graph = StateGraph(GraphState)
            graph.add_node("prepare_shared_context", self._prepare_shared_context)
            graph.add_node("select_specialists", self._select_specialists)
            graph.add_node("llm_specialist_reasoning", self._llm_specialist_reasoning)
            graph.add_node("validate_and_finalize", self._validate_and_finalize)
            graph.add_edge(START, "prepare_shared_context")
            graph.add_edge("prepare_shared_context", "select_specialists")
            graph.add_edge("select_specialists", "llm_specialist_reasoning")
            graph.add_edge("llm_specialist_reasoning", "validate_and_finalize")
            graph.add_edge("validate_and_finalize", END)
            self._compiled_graph = graph.compile()
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on deployment env
            self.import_error = str(exc)
            self.available = False

    def run(
        self,
        context: PlantContext,
        deterministic_signals: list[AgentSignal],
        max_agents: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[list[AgentSignal], dict[str, Any]]:
        if not self.available or self._compiled_graph is None:
            return deterministic_signals, {
                "enabled": False,
                "available": False,
                "error": self.import_error or "LangGraph is not available",
                "trace": [],
            }

        fallback_by_agent = {signal.agent_name: signal for signal in deterministic_signals}
        seed_signals = self._make_seed_signals(context, deterministic_signals)
        state: GraphState = {
            "context": context,
            "seed_signals": seed_signals,
            "reviewed_signals": [],
            "final_signals": [],
            "graph_trace": [],
            "error": None,
        }
        try:
            result = self._compiled_graph.invoke(
                state,
                config={"recursion_limit": 20},
            )
            final = list(result.get("final_signals") or [])
            if not final:
                final = deterministic_signals
            # Safety fallback: any LLM-first node that did not complete becomes
            # the corresponding deterministic specialist signal.
            safe_final: list[AgentSignal] = []
            for signal in final:
                meta = dict(signal.metadata or {})
                if meta.get("llm_reasoning_mode") == "langgraph_llm_first" and not meta.get("llm_used"):
                    fallback = fallback_by_agent.get(signal.agent_name)
                    if fallback is not None:
                        fm = dict(fallback.metadata or {})
                        fm.update({
                            "decision_basis": "deterministic_rules_fallback_after_langgraph_failure",
                            "llm_reasoning_mode": "langgraph_llm_first",
                            "llm_error": meta.get("llm_error") or "LangGraph LLM specialist reasoning did not produce usable output.",
                            "langgraph_used": True,
                        })
                        safe_final.append(replace(fallback, metadata=fm))
                        continue
                safe_final.append(signal)
            return safe_final, {
                "enabled": True,
                "available": True,
                "error": result.get("error"),
                "trace": result.get("graph_trace") or [],
            }
        except Exception as exc:
            return deterministic_signals, {
                "enabled": True,
                "available": True,
                "error": str(exc),
                "trace": [{"node": "graph_exception", "status": "failed", "detail": str(exc)}],
            }

    def _prepare_shared_context(self, state: GraphState) -> GraphState:
        trace = list(state.get("graph_trace") or [])
        context = state["context"]
        trace.append({
            "node": "prepare_shared_context",
            "status": "completed",
            "timestamp": context.current.get("timestamp"),
            "event_label": context.current.get("event_label"),
            "plant_risk_score": context.current.get("plant_risk_score"),
        })
        return {**state, "graph_trace": trace}

    def _select_specialists(self, state: GraphState) -> GraphState:
        trace = list(state.get("graph_trace") or [])
        seed_signals = list(state.get("seed_signals") or [])
        # For LangGraph LLM-first mode, review the selected agents supplied by
        # the UI depth option. If max_agents needs to be changed, it is already
        # controlled before this graph is constructed.
        trace.append({
            "node": "select_specialists",
            "status": "completed",
            "selected_agents": [s.agent_name for s in seed_signals],
            "selection_basis": "LLM-first graph mode uses plant state + shared memory + agent playbooks; deterministic signal is not supplied as reasoning input.",
        })
        return {**state, "seed_signals": seed_signals, "graph_trace": trace}

    def _llm_specialist_reasoning(self, state: GraphState) -> GraphState:
        trace = list(state.get("graph_trace") or [])
        seed_signals = list(state.get("seed_signals") or [])
        context = state["context"]
        if not self.synthesizer.enabled:
            trace.append({"node": "llm_specialist_reasoning", "status": "skipped", "detail": "OpenRouter not enabled"})
            return {**state, "reviewed_signals": seed_signals, "graph_trace": trace, "error": "OpenRouter not enabled"}
        try:
            reviewed = self.synthesizer.review_agent_signals_batch(
                seed_signals,
                context,
                timeout_seconds=config.LANGGRAPH_TOTAL_TIMEOUT_SECONDS,
            )
            trace.append({
                "node": "llm_specialist_reasoning",
                "status": "completed",
                "requested_agents": [s.agent_name for s in seed_signals],
                "reviewed_agents": [s.agent_name for s in reviewed if (s.metadata or {}).get("llm_used")],
                "model": self.synthesizer.last_successful_model,
            })
            return {**state, "reviewed_signals": reviewed, "graph_trace": trace}
        except Exception as exc:
            trace.append({"node": "llm_specialist_reasoning", "status": "failed", "detail": str(exc)})
            return {**state, "reviewed_signals": seed_signals, "graph_trace": trace, "error": str(exc)}

    def _validate_and_finalize(self, state: GraphState) -> GraphState:
        trace = list(state.get("graph_trace") or [])
        reviewed = list(state.get("reviewed_signals") or state.get("seed_signals") or [])
        finalized: list[AgentSignal] = []
        for signal in reviewed:
            meta = dict(signal.metadata or {})
            meta["langgraph_used"] = True
            meta["langgraph_node"] = "validate_and_finalize"
            finalized.append(replace(signal, metadata=meta))
        trace.append({
            "node": "validate_and_finalize",
            "status": "completed",
            "final_agents": [s.agent_name for s in finalized],
            "validation": "Actions are still clamped by deterministic safety logic and require operator approval.",
        })
        return {**state, "final_signals": finalized, "graph_trace": trace}

    def _make_seed_signals(self, context: PlantContext, deterministic_signals: list[AgentSignal]) -> list[AgentSignal]:
        fallback_by_agent = {signal.agent_name: signal for signal in deterministic_signals}
        seeds: list[AgentSignal] = []
        plant_score = safe_float(context.current.get("plant_risk_score"))
        event_label = str(context.current.get("event_label", "normal"))
        for agent in self.agents:
            fallback = fallback_by_agent.get(agent.name)
            metadata = {
                "decision_basis": "langgraph_llm_first_pending_safety_validation",
                "llm_reasoning_mode": "langgraph_llm_first",
                "llm_review_requested": True,
                "llm_selection_reason": "langgraph_supervisor_selected_specialist",
                "deterministic_fallback_available": bool(fallback),
                "deterministic_fallback_confidence": fallback.confidence if fallback else None,
                "deterministic_fallback_message": fallback.message if fallback else None,
                "deterministic_scaffold_text": "Not supplied to the LLM in LangGraph LLM-first mode. Deterministic logic is used only as fallback and safety validation.",
                "langgraph_used": True,
            }
            seeds.append(
                AgentSignal(
                    agent.name,
                    agent.decision_area,
                    "medium" if plant_score >= 50 or event_label.lower() != "normal" else "low",
                    0.0,
                    f"LangGraph LLM-first specialist reasoning requested for {agent.decision_area}.",
                    [
                        f"Plant risk score: {plant_score:.1f}",
                        f"Detected scenario: {event_label}",
                        "Specialist must reason from shared memory, plant state, trends, similar cases, and allowed action bounds.",
                    ],
                    {},
                    [],
                    ["Operator approval required for setpoint changes"],
                    ["langgraph_llm_first"],
                    metadata,
                )
            )
        return seeds
