from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import escape
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _load_streamlit_secrets_to_env() -> None:
    """Expose Streamlit Cloud secrets as environment variables before backend config imports."""
    try:
        secrets = st.secrets
    except Exception:
        return

    def set_if_scalar(key: str, value: Any) -> None:
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(str(key), str(value))
        elif isinstance(value, (list, tuple)):
            os.environ.setdefault(str(key), ",".join(str(item) for item in value))

    try:
        for key, value in secrets.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    set_if_scalar(nested_key, nested_value)
            else:
                set_if_scalar(key, value)
    except Exception:
        return


_load_streamlit_secrets_to_env()

from backend.coordinator import AgentCoordinator
from backend.data_store import DataStore

DEFAULT_BACKEND_URL = "embedded-streamlit-runtime"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


FRONTEND_API_TIMEOUT_SECONDS = _env_float("FRONTEND_API_TIMEOUT_SECONDS", 15.0)
FRONTEND_LLM_API_TIMEOUT_SECONDS = _env_float("FRONTEND_LLM_API_TIMEOUT_SECONDS", 25.0)

st.set_page_config(page_title="Blast Furnace Operator Copilot POC", layout="wide", initial_sidebar_state="expanded")

APP_CSS = """
<style>
.block-container {padding-top: 1.1rem;}
.bf-status-grid,
.bf-kpi-grid {
    display: grid;
    gap: 0.85rem;
    margin: 0.6rem 0 1.0rem 0;
}
.bf-status-grid {grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));}
.bf-kpi-grid {grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));}
.bf-kpi-card {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-left: 7px solid #64748b;
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    min-height: 112px;
    box-shadow: 0 1px 7px rgba(15, 23, 42, 0.08);
    overflow-wrap: anywhere;
}
.bf-kpi-card.level-critical {border-left-color: #dc2626; background: #fff7f7;}
.bf-kpi-card.level-high {border-left-color: #ea580c; background: #fff7ed;}
.bf-kpi-card.level-medium {border-left-color: #ca8a04; background: #fffbeb;}
.bf-kpi-card.level-low {border-left-color: #16a34a; background: #f0fdf4;}
.bf-kpi-card.level-normal {border-left-color: #2563eb; background: #eff6ff;}
.bf-kpi-card.level-info {border-left-color: #0891b2; background: #ecfeff;}
.bf-kpi-label {
    color: #334155;
    font-size: 0.84rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    line-height: 1.18;
    margin-bottom: 0.45rem;
}
.bf-kpi-value-row {
    display: flex;
    align-items: baseline;
    gap: 0.35rem;
    flex-wrap: wrap;
}
.bf-kpi-value {
    color: #0f172a;
    font-size: clamp(1.45rem, 2.2vw, 2.3rem);
    font-weight: 800;
    line-height: 1.05;
    white-space: normal;
}
.bf-kpi-unit {
    color: #475569;
    font-size: 0.92rem;
    font-weight: 650;
    line-height: 1.15;
}
.bf-kpi-subtitle {
    color: #475569;
    font-size: 0.84rem;
    line-height: 1.25;
    margin-top: 0.45rem;
}
.bf-section-note {
    background: #f8fafc;
    border: 1px solid #dbe4ef;
    border-radius: 12px;
    padding: 0.75rem 0.95rem;
    color: #334155;
    margin: 0.5rem 0 0.8rem 0;
}
.bf-message-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
    gap: 0.7rem;
    margin-top: 0.5rem;
}
.bf-message-card {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 12px;
    padding: 0.75rem 0.8rem;
    box-shadow: 0 1px 6px rgba(15, 23, 42, 0.07);
}
.bf-message-path {
    font-size: 0.84rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.35rem;
}
.bf-message-type {
    color: #0891b2;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.bf-message-content {
    color: #334155;
    font-size: 0.82rem;
    line-height: 1.28;
    min-height: 2.5rem;
}
.bf-confidence-bar {
    height: 7px;
    background: #e2e8f0;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 0.55rem;
}
.bf-confidence-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #0891b2, #2563eb);
}
</style>
"""

SEVERITY_COLORS = {
    "critical": "#dc2626",
    "high": "#ea580c",
    "medium": "#ca8a04",
    "low": "#16a34a",
    "normal": "#2563eb",
    "info": "#0891b2",
    "unknown": "#64748b",
}

NODE_FILL = {
    "source": "#e0f2fe",
    "context": "#dcfce7",
    "diagnostic": "#fef3c7",
    "lever": "#ede9fe",
    "coord": "#fce7f3",
    "safety": "#fee2e2",
    "operator": "#e0e7ff",
    "feedback": "#f1f5f9",
    "llm": "#ccfbf1",
}

AGENT_SHORT_NAMES = {
    "ThermalStateAgent": "Thermal State",
    "PermeabilityAgent": "Permeability",
    "QualityAgent": "Hot Metal Quality",
    "WindVolumeAgent": "Wind Volume",
    "PCIAgent": "PCI Rate",
    "CokeRateAgent": "Coke Rate",
    "FuelRateAgent": "Fuel Rate",
    "OxygenEnrichmentAgent": "Oxygen Enrichment",
    "BlastTemperatureAgent": "Blast Temperature",
    "TopPressureAgent": "Top Pressure",
    "BurdenDistributionAgent": "Burden Distribution",
    "TappingAgent": "Tapping Priority",
    "CoordinatorAgent": "Coordinator",
}


def render_html_fragment(html: str) -> None:
    """Render HTML as HTML, not as a Markdown code block.

    Streamlit >=1.36 provides st.html, which avoids Markdown parsing. For older
    Streamlit versions, the dedented unsafe_allow_html fallback is retained.
    """
    fragment = dedent(html).strip()
    if not fragment:
        return
    if hasattr(st, "html"):
        st.html(fragment)
    else:
        st.markdown(fragment, unsafe_allow_html=True)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}




def clean_operator_text(value: Any) -> str:
    """Prevent raw LLM JSON/code-fence output from leaking into the operator UI."""
    text = str(value or "").strip()
    if not text:
        return ""
    markers = [
        "OpenRouter returned a non-JSON review",
        "Model note:",
        "```json",
        "{ \"agent_reviews\"",
        "agent_reviews",
    ]
    if any(marker in text for marker in markers):
        prefix = text.split("OpenRouter returned a non-JSON review", 1)[0].strip()
        prefix = prefix.split("Model note:", 1)[0].strip()
        if prefix:
            return prefix + " OpenRouter review was received but could not be mapped cleanly; deterministic scaffold retained."
        return "OpenRouter review was received but could not be mapped cleanly; deterministic scaffold retained."
    return text


def clean_records_for_display(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in records:
        new_row = dict(row)
        for key in ["message", "content", "llm_error", "reasoning_addendum"]:
            if key in new_row:
                new_row[key] = clean_operator_text(new_row.get(key))
        cleaned.append(new_row)
    return cleaned


@dataclass
class EmbeddedRuntime:
    store: DataStore
    coordinator: AgentCoordinator
    executor: ThreadPoolExecutor
    lock: threading.Lock
    jobs: dict[str, dict[str, Any]]


@st.cache_resource(show_spinner=False)
def get_runtime() -> EmbeddedRuntime:
    return EmbeddedRuntime(
        store=DataStore(),
        coordinator=AgentCoordinator(),
        executor=ThreadPoolExecutor(max_workers=2),
        lock=threading.Lock(),
        jobs={},
    )


def _compact_job(job: dict[str, Any]) -> dict[str, Any]:
    return dict(job)


def _run_llm_review_job(
    job_id: str,
    index: int,
    include_llm_summary: bool,
    include_llm_agents: bool,
    llm_agent_max_agents: int | None,
    llm_agent_timeout_seconds: float | None,
    llm_agent_max_workers: int | None,
) -> None:
    runtime = get_runtime()
    with runtime.lock:
        if job_id in runtime.jobs:
            runtime.jobs[job_id].update({"status": "running", "started_at_epoch": time.time()})
    try:
        result = runtime.coordinator.run(
            runtime.store.get_context(index),
            include_llm_summary=include_llm_summary,
            include_llm_agents=include_llm_agents,
            llm_agent_max_agents=llm_agent_max_agents,
            llm_agent_timeout_seconds=llm_agent_timeout_seconds,
            llm_agent_max_workers=llm_agent_max_workers,
        )
        with runtime.lock:
            if job_id in runtime.jobs:
                runtime.jobs[job_id].update(
                    {
                        "status": "completed",
                        "completed_at_epoch": time.time(),
                        "result": result,
                        "error": None,
                    }
                )
    except Exception as exc:
        with runtime.lock:
            if job_id in runtime.jobs:
                runtime.jobs[job_id].update(
                    {
                        "status": "failed",
                        "completed_at_epoch": time.time(),
                        "result": None,
                        "error": str(exc),
                    }
                )


@st.cache_data(ttl=4, show_spinner=False)
def api_get(base_url: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Embedded replacement for the local FastAPI calls.

    The Streamlit Cloud version runs the agent engine in-process, so there is no
    localhost backend. The function keeps the old API shape to minimize UI changes.
    """
    runtime = get_runtime()
    store = runtime.store
    coordinator = runtime.coordinator
    params = params or {}

    def as_bool(name: str, default: bool = False) -> bool:
        return _truthy(params.get(name, default))

    def as_int(name: str, default: int | None = None) -> int | None:
        value = params.get(name, default)
        if value is None:
            return None
        return int(value)

    def as_float(name: str, default: float | None = None) -> float | None:
        value = params.get(name, default)
        if value is None:
            return None
        return float(value)

    if path == "/health":
        llm_status = coordinator.synthesizer.status()
        return {
            "status": "ok",
            "rows_loaded": len(store.df),
            "current_index": store.current_index,
            "llm_provider": llm_status.get("llm_provider"),
            "openrouter_reasoning_enabled": llm_status.get("openrouter_reasoning_enabled"),
            "openrouter_free_models": llm_status.get("openrouter_free_models", []),
        }
    if path == "/api/llm-status":
        status = coordinator.synthesizer.status()
        try:
            status["openrouter_key_health"] = coordinator.synthesizer.key_health()
        except Exception as exc:
            status["openrouter_key_health"] = {"ok": False, "error": str(exc)}
        return status
    if path == "/api/metadata":
        return store.metadata_payload()
    if path == "/api/state":
        return store.get_state(as_int("index"))
    if path == "/api/history":
        idx = store.normalize_index(as_int("index"))
        window = int(params.get("window", 36))
        return {"dataset_index": idx, "window": window, "records": store.get_history(idx, window=window)}
    if path == "/api/recommendations":
        idx = store.normalize_index(as_int("index"))
        return coordinator.run(
            store.get_context(idx),
            include_llm_summary=as_bool("include_llm_summary"),
            include_llm_agents=as_bool("include_llm_agents"),
            llm_agent_max_agents=as_int("llm_agent_max_agents"),
            llm_agent_timeout_seconds=as_float("llm_agent_timeout_seconds"),
            llm_agent_max_workers=as_int("llm_agent_max_workers"),
        )
    if path.startswith("/api/llm-review/"):
        job_id = path.rsplit("/", 1)[-1]
        with runtime.lock:
            job = runtime.jobs.get(job_id)
            if not job:
                raise KeyError(f"Unknown LLM review job_id: {job_id}")
            return _compact_job(job)
    if path == "/api/llm-review":
        with runtime.lock:
            jobs = sorted(runtime.jobs.values(), key=lambda item: float(item.get("created_at_epoch", 0)), reverse=True)
            return {"jobs": [_compact_job(job) for job in jobs[:20]]}
    if path == "/api/agent-network":
        idx = store.normalize_index(as_int("index"))
        payload = coordinator.run(store.get_context(idx), include_llm_summary=False)
        return {"dataset_index": idx, "agent_messages": payload["agent_messages"], "signals": payload["signals"], "architecture_status": payload["architecture_status"]}
    if path == "/api/similar-cases":
        idx = store.normalize_index(as_int("index"))
        limit = int(params.get("limit", 5))
        return {"dataset_index": idx, "cases": store.get_similar_cases(idx, limit=limit)}
    if path == "/api/decision-log":
        return {"records": store.get_decision_log()}
    if path == "/api/playbook":
        return {"records": store.get_playbook_records()}
    if path == "/api/data-dictionary":
        return {"records": store.get_data_dictionary_records()}
    raise ValueError(f"Unsupported embedded API path: {path}")


def api_post(base_url: str, path: str, params: dict[str, Any] | None = None, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = get_runtime()
    store = runtime.store
    coordinator = runtime.coordinator
    params = params or {}
    json_payload = json_payload or {}

    if path == "/api/simulate-step":
        new_index = store.step_index(int(json_payload.get("index", store.current_index)), str(json_payload.get("direction", "next")))
        store.set_current_index(new_index)
        api_get.clear()
        return {"current_index": new_index, "state": store.get_state(new_index)}

    if path == "/api/llm-review/start":
        idx = store.normalize_index(int(params.get("index", store.current_index)))
        job_id = str(uuid.uuid4())
        now = time.time()
        include_llm_summary = _truthy(params.get("include_llm_summary", True))
        include_llm_agents = _truthy(params.get("include_llm_agents", True))
        llm_agent_max_agents = int(params.get("llm_agent_max_agents", 1))
        llm_agent_timeout_seconds = float(params.get("llm_agent_timeout_seconds", 12.0))
        llm_agent_max_workers = int(params.get("llm_agent_max_workers", 1))
        job = {
            "job_id": job_id,
            "status": "queued",
            "dataset_index": idx,
            "include_llm_summary": include_llm_summary,
            "include_llm_agents": include_llm_agents,
            "llm_agent_max_agents": llm_agent_max_agents,
            "llm_agent_timeout_seconds": llm_agent_timeout_seconds,
            "llm_agent_max_workers": llm_agent_max_workers,
            "created_at_epoch": now,
            "started_at_epoch": None,
            "completed_at_epoch": None,
            "result": None,
            "error": None,
        }
        with runtime.lock:
            runtime.jobs[job_id] = job
            if len(runtime.jobs) > 50:
                ordered = sorted(runtime.jobs.items(), key=lambda item: float(item[1].get("created_at_epoch", 0)))
                for old_job_id, _ in ordered[: len(runtime.jobs) - 50]:
                    runtime.jobs.pop(old_job_id, None)
        runtime.executor.submit(
            _run_llm_review_job,
            job_id,
            idx,
            include_llm_summary,
            include_llm_agents,
            llm_agent_max_agents,
            llm_agent_timeout_seconds,
            llm_agent_max_workers,
        )
        api_get.clear()
        return {"job_id": job_id, "status": "queued", "dataset_index": idx, "message": "OpenRouter LLM review job started."}

    if path.startswith("/api/recommendations/") and path.endswith("/feedback"):
        recommendation_id = path.split("/")[3]
        idx = int(params.get("index", store.current_index))
        record = store.append_feedback(
            idx,
            recommendation_id,
            str(json_payload.get("operator_id", "operator")),
            str(json_payload.get("decision", "reviewed")),
            json_payload.get("modified_action"),
            json_payload.get("notes"),
        )
        api_get.clear()
        return {"status": "logged", "record": record}

    raise ValueError(f"Unsupported embedded API POST path: {path}")


def fmt(value: Any, digits: int = 1, default: str = "-") -> str:
    if value is None:
        return default
    try:
        number = float(value)
        if abs(number) >= 100:
            return f"{number:,.0f}"
        return f"{number:,.{digits}f}"
    except Exception:
        return str(value)


def level_from_score(score: Any) -> str:
    try:
        value = float(score)
    except Exception:
        return "info"
    if value >= 85:
        return "critical"
    if value >= 70:
        return "high"
    if value >= 45:
        return "medium"
    return "normal"


def level_from_text(value: Any) -> str:
    text = str(value or "").lower()
    if "critical" in text or "very high" in text:
        return "critical"
    if "high" in text or "abnormal" in text:
        return "high"
    if "medium" in text or "warning" in text or "reduced" in text:
        return "medium"
    if "normal" in text or "good" in text or "stable" in text:
        return "normal"
    return "info"


def render_kpi_grid(cards: list[dict[str, Any]], css_class: str = "bf-kpi-grid") -> None:
    """Render responsive KPI cards.

    The fragments are dedented before being passed to Streamlit. Otherwise,
    Markdown can treat indented HTML as a code block and display it literally.
    """
    html_parts = [f'<div class="{escape(css_class)}">']
    for card in cards:
        level = escape(str(card.get("level") or "info"))
        title = escape(str(card.get("title") or "-"))
        value = escape(str(card.get("value") or "-"))
        unit = escape(str(card.get("unit") or ""))
        subtitle = escape(str(card.get("subtitle") or ""))
        html_parts.append(
            dedent(
                f"""
                <div class="bf-kpi-card level-{level}">
                    <div class="bf-kpi-label">{title}</div>
                    <div class="bf-kpi-value-row">
                        <span class="bf-kpi-value">{value}</span>
                        <span class="bf-kpi-unit">{unit}</span>
                    </div>
                    <div class="bf-kpi-subtitle">{subtitle}</div>
                </div>
                """
            ).strip()
        )
    html_parts.append("</div>")
    render_html_fragment("\n".join(html_parts))


def make_gauge(title: str, value: float, max_value: float = 100.0) -> go.Figure:
    val = float(value or 0)
    if val >= 85:
        bar_color = "#dc2626"
    elif val >= 70:
        bar_color = "#ea580c"
    elif val >= 45:
        bar_color = "#ca8a04"
    else:
        bar_color = "#2563eb"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": title, "font": {"size": 17}},
            number={"font": {"size": 30}},
            gauge={
                "axis": {"range": [0, max_value], "tickwidth": 1},
                "bar": {"color": bar_color},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 45], "color": "#eff6ff"},
                    {"range": [45, 70], "color": "#fef3c7"},
                    {"range": [70, 85], "color": "#ffedd5"},
                    {"range": [85, max_value], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig.update_layout(height=240, margin={"l": 20, "r": 20, "t": 55, "b": 20})
    return fig


def trend_figure(df: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    x = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else df.index
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines+markers", name=col))
    fig.update_layout(title=title, height=320, margin={"l": 20, "r": 20, "t": 45, "b": 20}, legend={"orientation": "h"})
    return fig


def action_table(action_deltas: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"Action parameter": key, "Recommended value/change": value} for key, value in action_deltas.items()])


def _svg_text_lines(text: str, max_chars: int = 20, max_lines: int = 3) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) >= max_lines:
            break
    if len(lines) < max_lines and current:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1].rstrip(".") + "..."
    return lines[:max_lines]


def _node_center(nodes: dict[str, dict[str, Any]], key: str) -> tuple[float, float]:
    node = nodes[key]
    return float(node["x"]) + float(node["w"]) / 2.0, float(node["y"]) + float(node["h"]) / 2.0


def _node_left(nodes: dict[str, dict[str, Any]], key: str) -> tuple[float, float]:
    node = nodes[key]
    return float(node["x"]), float(node["y"]) + float(node["h"]) / 2.0


def _node_right(nodes: dict[str, dict[str, Any]], key: str) -> tuple[float, float]:
    node = nodes[key]
    return float(node["x"]) + float(node["w"]), float(node["y"]) + float(node["h"]) / 2.0


def render_agent_flow_figure(messages: list[dict[str, Any]], signals: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> None:
    signal_by_agent: dict[str, dict[str, Any]] = {str(sig.get("agent_name")): sig for sig in signals if sig.get("agent_name")}
    recommended_agents = {str(rec.get("agent_name")) for rec in recommendations if rec.get("agent_name")}

    nodes: dict[str, dict[str, Any]] = {
        "sources": {"label": "Plant data sources", "sub": "Historian, LIMS, MES, SCADA, shift logs, playbooks", "x": 30, "y": 70, "w": 205, "h": 92, "type": "source"},
        "context": {"label": "Shared plant context", "sub": "Current state, trends, data quality, operating envelope", "x": 280, "y": 70, "w": 230, "h": 92, "type": "context"},
        "ThermalStateAgent": {"label": "Thermal State Agent", "sub": "Hot metal temp, Si, heat trend", "x": 575, "y": 30, "w": 190, "h": 76, "type": "diagnostic"},
        "PermeabilityAgent": {"label": "Permeability Agent", "sub": "Pressure drop, gas flow, slips", "x": 575, "y": 126, "w": 190, "h": 76, "type": "diagnostic"},
        "QualityAgent": {"label": "Quality Agent", "sub": "Si, S, chemistry variability", "x": 575, "y": 222, "w": 190, "h": 76, "type": "diagnostic"},
        "WindVolumeAgent": {"label": "Wind Volume Agent", "sub": "Wind delta advisory", "x": 825, "y": 22, "w": 185, "h": 70, "type": "lever"},
        "PCIAgent": {"label": "PCI Agent", "sub": "Injection-rate advisory", "x": 1035, "y": 22, "w": 185, "h": 70, "type": "lever"},
        "CokeRateAgent": {"label": "Coke Rate Agent", "sub": "Coke-rate advisory", "x": 825, "y": 110, "w": 185, "h": 70, "type": "lever"},
        "FuelRateAgent": {"label": "Fuel Rate Agent", "sub": "Fuel-rate consistency", "x": 1035, "y": 110, "w": 185, "h": 70, "type": "lever"},
        "OxygenEnrichmentAgent": {"label": "Oxygen Enrichment Agent", "sub": "O2 enrichment advisory", "x": 825, "y": 198, "w": 185, "h": 70, "type": "lever"},
        "BlastTemperatureAgent": {"label": "Blast Temperature Agent", "sub": "Hot blast temp advisory", "x": 1035, "y": 198, "w": 185, "h": 70, "type": "lever"},
        "TopPressureAgent": {"label": "Top Pressure Agent", "sub": "Top-pressure advisory", "x": 825, "y": 286, "w": 185, "h": 70, "type": "lever"},
        "BurdenDistributionAgent": {"label": "Burden Distribution Agent", "sub": "Charging pattern advisory", "x": 1035, "y": 286, "w": 185, "h": 70, "type": "lever"},
        "TappingAgent": {"label": "Tapping Agent", "sub": "Tapping priority advisory", "x": 930, "y": 374, "w": 185, "h": 70, "type": "lever"},
        "CoordinatorAgent": {"label": "Coordinator Agent", "sub": "Resolves conflicts, ranks recommendations", "x": 1280, "y": 126, "w": 205, "h": 90, "type": "coord"},
        "SafetyGate": {"label": "Safety and automation gate", "sub": "Envelope check, confidence, approval mode", "x": 1280, "y": 262, "w": 205, "h": 90, "type": "safety"},
        "LLMSummary": {"label": "OpenRouter reasoning summary", "sub": "Free-model narrative only; deterministic agents decide", "x": 1280, "y": 398, "w": 205, "h": 90, "type": "llm"},
        "Operator": {"label": "Operator dashboard", "sub": "Accept, reject, or modify recommendation", "x": 1280, "y": 534, "w": 205, "h": 90, "type": "operator"},
        "Log": {"label": "Decision log and learning loop", "sub": "Feedback saved for later tuning", "x": 930, "y": 610, "w": 255, "h": 80, "type": "feedback"},
    }

    static_edges = [
        ("sources", "context", "#0891b2", "Data ingestion"),
        ("context", "ThermalStateAgent", "#16a34a", "State context"),
        ("context", "PermeabilityAgent", "#16a34a", "State context"),
        ("context", "QualityAgent", "#16a34a", "State context"),
        ("context", "WindVolumeAgent", "#64748b", "Operating state"),
        ("context", "PCIAgent", "#64748b", "Operating state"),
        ("context", "CokeRateAgent", "#64748b", "Operating state"),
        ("context", "FuelRateAgent", "#64748b", "Operating state"),
        ("context", "OxygenEnrichmentAgent", "#64748b", "Operating state"),
        ("context", "BlastTemperatureAgent", "#64748b", "Operating state"),
        ("context", "TopPressureAgent", "#64748b", "Operating state"),
        ("context", "BurdenDistributionAgent", "#64748b", "Operating state"),
        ("context", "TappingAgent", "#64748b", "Operating state"),
        ("WindVolumeAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("PCIAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("CokeRateAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("FuelRateAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("OxygenEnrichmentAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("BlastTemperatureAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("TopPressureAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("BurdenDistributionAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("TappingAgent", "CoordinatorAgent", "#7c3aed", "Candidate action"),
        ("CoordinatorAgent", "SafetyGate", "#be123c", "Coordinated recommendation"),
        ("CoordinatorAgent", "LLMSummary", "#0f766e", "Structured reasoning"),
        ("SafetyGate", "Operator", "#dc2626", "Approval requirement"),
        ("LLMSummary", "Operator", "#0f766e", "Readable explanation"),
        ("Operator", "Log", "#334155", "Human feedback"),
        ("Log", "context", "#334155", "Learning loop"),
    ]

    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for msg in messages:
        src = str(msg.get("from_agent") or "")
        tgt = str(msg.get("to_agent") or "")
        if src not in nodes or tgt not in nodes:
            continue
        key = (src, tgt)
        conf = max(0.05, min(float(msg.get("confidence") or 0.2), 1.0))
        if key not in edge_map or conf > edge_map[key]["confidence"]:
            edge_map[key] = {"confidence": conf, "content": clean_operator_text(msg.get("content")), "message_type": str(msg.get("message_type") or "message")}
        else:
            edge_map[key]["content"] = edge_map[key]["content"]
    active_edges = list(edge_map.items())[:18]

    svg_parts: list[str] = []
    svg_parts.append("""
    <svg viewBox="0 0 1520 730" width="100%" height="730" role="img" aria-label="Agentic blast furnace information flow diagram">
        <defs>
            <marker id="arrowDefault" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
            </marker>
            <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#0891b2" />
            </marker>
            <marker id="arrowGreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#16a34a" />
            </marker>
            <marker id="arrowPurple" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#7c3aed" />
            </marker>
            <marker id="arrowRed" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#dc2626" />
            </marker>
            <marker id="arrowTeal" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#0f766e" />
            </marker>
            <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.16"/>
            </filter>
        </defs>
        <rect x="0" y="0" width="1520" height="730" rx="18" fill="#ffffff" stroke="#cbd5e1" />
        <text x="30" y="30" font-size="20" font-weight="800" fill="#0f172a">Agentic information flow for the selected timestamp</text>
        <text x="30" y="54" font-size="13" fill="#475569">Solid arrows show the POC pipeline. Colored curved arrows show live inter-agent messages for the current furnace state.</text>
    """)

    def marker_for(color: str) -> str:
        if color == "#0891b2":
            return "arrowBlue"
        if color == "#16a34a":
            return "arrowGreen"
        if color == "#7c3aed":
            return "arrowPurple"
        if color in {"#dc2626", "#be123c"}:
            return "arrowRed"
        if color == "#0f766e":
            return "arrowTeal"
        return "arrowDefault"

    for src, tgt, color, label in static_edges:
        if src not in nodes or tgt not in nodes:
            continue
        sx, sy = _node_right(nodes, src)
        tx, ty = _node_left(nodes, tgt)
        if src == "Operator" and tgt == "Log":
            sx, sy = _node_left(nodes, src)
            tx, ty = _node_right(nodes, tgt)
        if src == "Log" and tgt == "context":
            sx, sy = _node_left(nodes, src)
            tx, ty = _node_center(nodes, tgt)
            ty = ty + 40
            path = f"M {sx:.1f} {sy:.1f} C {sx-180:.1f} {sy+70:.1f}, {tx-220:.1f} {ty+60:.1f}, {tx:.1f} {ty:.1f}"
        else:
            bend = max(55, min(abs(tx - sx) / 2, 150))
            path = f"M {sx:.1f} {sy:.1f} C {sx+bend:.1f} {sy:.1f}, {tx-bend:.1f} {ty:.1f}, {tx:.1f} {ty:.1f}"
        svg_parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.0" stroke-opacity="0.45" marker-end="url(#{marker_for(color)})"><title>{escape(label)}</title></path>')

    for (src, tgt), edge in active_edges:
        sx, sy = _node_right(nodes, src)
        tx, ty = _node_left(nodes, tgt)
        if sx > tx:
            sx, sy = _node_left(nodes, src)
            tx, ty = _node_right(nodes, tgt)
        conf = float(edge["confidence"])
        stroke_width = 2.5 + conf * 3.5
        color = "#dc2626" if conf >= 0.9 else "#ea580c" if conf >= 0.75 else "#2563eb"
        bend = max(70, min(abs(tx - sx) / 2.0, 190))
        path = f"M {sx:.1f} {sy:.1f} C {sx+bend:.1f} {sy-30:.1f}, {tx-bend:.1f} {ty+30:.1f}, {tx:.1f} {ty:.1f}"
        tooltip = f"{AGENT_SHORT_NAMES.get(src, src)} to {AGENT_SHORT_NAMES.get(tgt, tgt)} | confidence {conf:.2f} | {edge.get('content', '')}"
        svg_parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke_width:.1f}" stroke-opacity="0.78" marker-end="url(#{marker_for(color)})"><title>{escape(tooltip)}</title></path>')

    for key, node in nodes.items():
        node_type = str(node.get("type", "context"))
        fill = NODE_FILL.get(node_type, "#f8fafc")
        label = str(node.get("label", key))
        sub = str(node.get("sub", ""))
        sig = signal_by_agent.get(key)
        severity = str(sig.get("severity") if sig else "normal").lower()
        stroke = SEVERITY_COLORS.get(severity, "#64748b")
        stroke_width = 3 if sig or key in recommended_agents else 1.2
        if key in {"CoordinatorAgent", "SafetyGate", "Operator"}:
            stroke_width = max(stroke_width, 2.2)
        x = float(node["x"])
        y = float(node["y"])
        w = float(node["w"])
        h = float(node["h"])
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" filter="url(#shadow)" />')
        for idx, line in enumerate(_svg_text_lines(label, max_chars=23, max_lines=2)):
            svg_parts.append(f'<text x="{x+12}" y="{y+24+idx*16}" font-size="14" font-weight="800" fill="#0f172a">{escape(line)}</text>')
        sub_y = y + 51
        for idx, line in enumerate(_svg_text_lines(sub, max_chars=30, max_lines=2)):
            svg_parts.append(f'<text x="{x+12}" y="{sub_y+idx*14}" font-size="11.5" fill="#475569">{escape(line)}</text>')
        if sig:
            conf = float(sig.get("confidence") or 0)
            sev_color = SEVERITY_COLORS.get(severity, "#64748b")
            svg_parts.append(f'<circle cx="{x+w-18}" cy="{y+17}" r="7" fill="{sev_color}" />')
            svg_parts.append(f'<text x="{x+12}" y="{y+h-10}" font-size="11" font-weight="800" fill="{sev_color}">{escape(severity.upper())} | confidence {conf:.2f}</text>')
        elif key in recommended_agents:
            svg_parts.append(f'<text x="{x+12}" y="{y+h-10}" font-size="11" font-weight="800" fill="#7c3aed">RECOMMENDATION OWNER</text>')

    svg_parts.append("""
        <rect x="30" y="640" width="770" height="52" rx="11" fill="#f8fafc" stroke="#cbd5e1" />
        <circle cx="54" cy="657" r="7" fill="#dc2626" /><text x="70" y="662" font-size="12" fill="#334155">Critical/high-confidence live message</text>
        <circle cx="300" cy="657" r="7" fill="#ea580c" /><text x="316" y="662" font-size="12" fill="#334155">High-severity coordination</text>
        <circle cx="510" cy="657" r="7" fill="#2563eb" /><text x="526" y="662" font-size="12" fill="#334155">Normal coordination message</text>
        <text x="54" y="681" font-size="12" fill="#475569">Node border color indicates the active agent severity. Thick borders indicate active signals or recommendation ownership.</text>
    </svg>
    """)

    diagram_html = """
    <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:18px; padding:8px; overflow-x:auto;">
    """ + "\n".join(svg_parts) + "</div>"
    components.html(diagram_html, height=780, scrolling=True)


def render_message_cards(messages: list[dict[str, Any]], limit: int = 12) -> None:
    """Render inter-agent message cards without Markdown code-block indentation."""
    if not messages:
        st.write("No inter-agent messages for the current timestamp.")
        return
    cards = ['<div class="bf-message-grid">']
    for msg in messages[:limit]:
        src = AGENT_SHORT_NAMES.get(str(msg.get("from_agent")), str(msg.get("from_agent") or "-"))
        tgt = AGENT_SHORT_NAMES.get(str(msg.get("to_agent")), str(msg.get("to_agent") or "-"))
        message_type = escape(str(msg.get("message_type") or "message"))
        content = escape(clean_operator_text(msg.get("content")))
        conf = max(0.0, min(float(msg.get("confidence") or 0.0), 1.0))
        cards.append(
            dedent(
                f"""
                <div class="bf-message-card">
                    <div class="bf-message-path">{escape(src)} &rarr; {escape(tgt)}</div>
                    <div class="bf-message-type">{message_type} | confidence {conf:.2f}</div>
                    <div class="bf-message-content">{content}</div>
                    <div class="bf-confidence-bar"><div class="bf-confidence-fill" style="width: {conf*100:.0f}%"></div></div>
                </div>
                """
            ).strip()
        )
    cards.append("</div>")
    render_html_fragment("\n".join(cards))


render_html_fragment(APP_CSS)
st.title("Blast Furnace Operator Decision Copilot")
st.caption("GitHub/Streamlit-ready POC: embedded hybrid multi-agent advisory architecture with optional OpenRouter specialist-agent review and summary synthesis.")

with st.sidebar:
    st.header("Deployment mode")
    backend_url = DEFAULT_BACKEND_URL
    st.caption("Running as a single embedded Streamlit app. No separate FastAPI process is required for Streamlit Cloud.")

    st.header("OpenRouter LLM review")
    include_llm_agents = st.toggle("Review specialist agents with OpenRouter", value=False)
    include_llm = st.toggle("Generate final OpenRouter reasoning summary", value=False)
    llm_depth_options = {
        "Fast: 1 active agent / 12s budget": {"max_agents": 1, "timeout": 12.0, "workers": 1},
        "Balanced: 2 active agents / 18s budget": {"max_agents": 2, "timeout": 18.0, "workers": 2},
        "Deep: 4 active agents / 30s budget": {"max_agents": 4, "timeout": 30.0, "workers": 2},
        "Confidence gate: all agents below 90% / 60s budget": {"max_agents": 12, "timeout": 60.0, "workers": 4},
    }
    llm_depth_label = st.selectbox(
        "Specialist LLM depth",
        list(llm_depth_options.keys()),
        index=0,
        disabled=not include_llm_agents,
    )
    llm_depth = llm_depth_options[llm_depth_label]
    llm_agent_max_agents = int(llm_depth["max_agents"])
    llm_agent_timeout_seconds = float(llm_depth["timeout"])
    llm_agent_max_workers = int(llm_depth["workers"])
    st.caption(
        "LLM review now runs as a separate backend job. In confidence-gate mode, every deterministic specialist signal below 90% confidence is sent to OpenRouter for review."
    )

try:
    health = api_get(backend_url, "/health")
    metadata = api_get(backend_url, "/api/metadata")
except Exception as exc:
    st.error("The embedded agent runtime could not be initialized. Check the dataset files and Streamlit secrets/environment variables.")
    st.exception(exc)
    st.stop()

row_count = int(metadata["row_count"])
if "dataset_index" not in st.session_state:
    st.session_state.dataset_index = int(metadata.get("current_index", row_count - 1))

with st.sidebar:
    st.success(f"Embedded agent runtime loaded. Rows loaded: {health['rows_loaded']}")
    selected_index = st.slider("Dataset timestamp index", 0, row_count - 1, int(st.session_state.dataset_index), step=1)
    st.session_state.dataset_index = selected_index
    c1, c2 = st.columns(2)
    if c1.button("Previous hour", use_container_width=True):
        result = api_post(backend_url, "/api/simulate-step", json_payload={"index": st.session_state.dataset_index, "direction": "previous"})
        st.session_state.dataset_index = int(result["current_index"])
        api_get.clear()
        st.rerun()
    if c2.button("Next hour", use_container_width=True):
        result = api_post(backend_url, "/api/simulate-step", json_payload={"index": st.session_state.dataset_index, "direction": "next"})
        st.session_state.dataset_index = int(result["current_index"])
        api_get.clear()
        st.rerun()
    if st.button("Refresh", use_container_width=True):
        api_get.clear()
        st.rerun()

    can_run_llm_review = bool(include_llm_agents or include_llm)
    if st.button("Run OpenRouter LLM review", type="primary", use_container_width=True, disabled=not can_run_llm_review):
        params = {
            "index": st.session_state.dataset_index,
            "include_llm_summary": include_llm,
            "include_llm_agents": include_llm_agents,
            "llm_agent_max_agents": llm_agent_max_agents,
            "llm_agent_timeout_seconds": llm_agent_timeout_seconds,
            "llm_agent_max_workers": llm_agent_max_workers,
        }
        result = api_post(backend_url, "/api/llm-review/start", params=params)
        st.session_state.llm_review_job_id = result["job_id"]
        st.session_state.llm_review_job_index = int(st.session_state.dataset_index)
        api_get.clear()
        st.rerun()
    if not can_run_llm_review:
        st.caption("Select at least one OpenRouter option above to run an LLM review job.")

index = int(st.session_state.dataset_index)
state = api_get(backend_url, "/api/state", params={"index": index})
history_payload = api_get(backend_url, "/api/history", params={"index": index, "window": 48})
recommendation_payload = api_get(backend_url, "/api/recommendations", params={"index": index, "include_llm_summary": False, "include_llm_agents": False})
llm_job_payload = None
llm_job_id = st.session_state.get("llm_review_job_id")
if llm_job_id:
    try:
        llm_job_payload = api_get(backend_url, f"/api/llm-review/{llm_job_id}")
        job_index = int(llm_job_payload.get("dataset_index", -1))
        job_status = str(llm_job_payload.get("status", "unknown"))
        if job_index == index and job_status == "completed" and llm_job_payload.get("result"):
            recommendation_payload = llm_job_payload["result"]
            recommendation_payload["llm_job_status"] = "completed"
            recommendation_payload["llm_job_id"] = llm_job_id
            st.success("OpenRouter LLM review completed. The dashboard is showing the hybrid LLM-agent output for this timestamp.")
        elif job_index == index and job_status in {"queued", "running"}:
            st.info(f"OpenRouter LLM review job is {job_status}. The dashboard is showing deterministic output until the job completes. Use Refresh or Check status.")
        elif job_index == index and job_status == "failed":
            st.warning(f"OpenRouter LLM review failed; deterministic output is shown. Error: {llm_job_payload.get('error')}")
        elif job_index != index:
            st.caption(f"Existing OpenRouter review job belongs to dataset row {job_index}; current row is {index}.")
    except Exception as exc:
        st.warning(f"Could not read OpenRouter LLM review job status: {exc}")
if recommendation_payload.get("frontend_warning"):
    st.warning(recommendation_payload["frontend_warning"])
history_df = pd.DataFrame(history_payload["records"])

render_kpi_grid(
    [
        {"title": "Timestamp", "value": str(state.get("timestamp", "-")), "subtitle": f"Dataset row {index}", "level": "info"},
        {"title": "Plant risk", "value": str(state.get("plant_risk_level", "-")).upper(), "unit": fmt(state.get("plant_risk_score"), 1), "subtitle": "Overall synthesized risk", "level": level_from_text(state.get("plant_risk_level"))},
        {"title": "Scenario", "value": state.get("event_label", "-"), "subtitle": f"Mode: {state.get('operating_mode', '-')}", "level": level_from_text(state.get("event_label"))},
        {"title": "Sensor quality", "value": state.get("sensor_quality_flag", "-"), "subtitle": "Data confidence input", "level": level_from_text(state.get("sensor_quality_flag"))},
    ],
    css_class="bf-status-grid",
)

main_tab, agents_tab, cases_tab, log_tab, data_tab = st.tabs(["Operator dashboard", "Agent coordination", "Similar cases", "Decision log", "Data and playbook"])

with main_tab:
    st.subheader("Operator console")
    render_html_fragment(
        '<div class="bf-section-note">Primary operator levers and critical furnace-response KPIs are shown as wide cards to avoid the truncation that can occur with compact metric widgets.</div>'
    )
    render_kpi_grid(
        [
            {"title": "Wind volume", "value": fmt(state.get("wind_volume_nm3_min"), 0), "unit": "Nm3/min", "subtitle": "Primary gas-flow lever", "level": "info"},
            {"title": "PCI rate", "value": fmt(state.get("pci_rate_kg_thm"), 1), "unit": "kg/thm", "subtitle": "Coal injection intensity", "level": "info"},
            {"title": "Coke rate", "value": fmt(state.get("coke_rate_kg_thm"), 1), "unit": "kg/thm", "subtitle": "Structural fuel support", "level": "info"},
            {"title": "Total fuel rate", "value": fmt(state.get("total_fuel_rate_kg_thm"), 1), "unit": "kg/thm", "subtitle": "Coke plus auxiliary fuel", "level": "info"},
            {"title": "Hot metal temperature", "value": fmt(state.get("hot_metal_temp_c"), 1), "unit": "C", "subtitle": "Thermal output indicator", "level": level_from_score(state.get("thermal_risk_score"))},
            {"title": "Hot metal silicon", "value": fmt(state.get("hot_metal_si_pct"), 3), "unit": "%", "subtitle": "Quality and thermal proxy", "level": level_from_score(state.get("quality_risk_score"))},
            {"title": "Pressure drop", "value": fmt(state.get("pressure_drop_kpa"), 1), "unit": "kPa", "subtitle": "Gas-flow restriction signal", "level": level_from_score(state.get("permeability_risk_score"))},
            {"title": "Permeability index", "value": fmt(state.get("permeability_index"), 1), "unit": "index", "subtitle": "Lower value indicates restriction", "level": level_from_score(state.get("permeability_risk_score"))},
        ]
    )

    gauge_cols = st.columns(4)
    gauge_cols[0].plotly_chart(make_gauge("Health score", state.get("health_score", 0)), use_container_width=True)
    gauge_cols[1].plotly_chart(make_gauge("Thermal risk", state.get("thermal_risk_score", 0)), use_container_width=True)
    gauge_cols[2].plotly_chart(make_gauge("Permeability risk", state.get("permeability_risk_score", 0)), use_container_width=True)
    gauge_cols[3].plotly_chart(make_gauge("Quality risk", state.get("quality_risk_score", 0)), use_container_width=True)

    st.markdown("#### Furnace operating mimic")
    mimic_cols = st.columns(4)
    with mimic_cols[0]:
        st.markdown("**Blast system**")
        st.write(f"Hot blast temperature: {fmt(state.get('hot_blast_temp_c'), 0)} C")
        st.write(f"Blast pressure: {fmt(state.get('blast_pressure_kpa'), 1)} kPa")
        st.write(f"Oxygen enrichment: {fmt(state.get('oxygen_enrichment_pct'), 2)} %")
        st.write(f"Blast humidity: {fmt(state.get('blast_humidity_g_nm3'), 1)} g/Nm3")
    with mimic_cols[1]:
        st.markdown("**Furnace stack**")
        st.write(f"Pressure drop: {fmt(state.get('pressure_drop_kpa'), 1)} kPa")
        st.write(f"Permeability index: {fmt(state.get('permeability_index'), 1)}")
        st.write(f"Burden descent: {fmt(state.get('burden_descent_rate_m_h'), 2)} m/h")
        st.write(f"Distribution: {state.get('burden_distribution_mode', '-')}")
    with mimic_cols[2]:
        st.markdown("**Gas and raceway**")
        st.write(f"Gas utilization: {fmt(state.get('gas_utilization_pct'), 1)} %")
        st.write(f"Top pressure: {fmt(state.get('top_pressure_kpa'), 1)} kPa")
        st.write(f"Top gas temp: {fmt(state.get('top_gas_temp_c'), 1)} C")
        st.write(f"RAFT: {fmt(state.get('raceway_adiabatic_flame_temp_c'), 0)} C")
    with mimic_cols[3]:
        st.markdown("**Hearth and product**")
        st.write(f"Hearth liquid index: {fmt(state.get('hearth_liquid_level_index'), 1)}")
        st.write(f"Sidewall temp: {fmt(state.get('hearth_sidewall_temp_c'), 1)} C")
        st.write(f"Production: {fmt(state.get('production_tph'), 1)} tph")
        st.write(f"Slag volume: {fmt(state.get('slag_volume_kg_thm'), 1)} kg/thm")

    st.markdown("#### 48-hour trends")
    t1, t2 = st.columns(2)
    t1.plotly_chart(trend_figure(history_df, ["hot_metal_temp_c", "predicted_hot_metal_temp_4h_c", "thermal_state_index"], "Thermal response"), use_container_width=True)
    t2.plotly_chart(trend_figure(history_df, ["permeability_index", "pressure_drop_kpa", "gas_utilization_pct"], "Gas flow and permeability"), use_container_width=True)
    t3, t4 = st.columns(2)
    t3.plotly_chart(trend_figure(history_df, ["wind_volume_nm3_min", "pci_rate_kg_thm", "coke_rate_kg_thm"], "Operator levers"), use_container_width=True)
    t4.plotly_chart(trend_figure(history_df, ["hot_metal_si_pct", "predicted_si_4h_pct", "hot_metal_s_pct"], "Hot metal quality"), use_container_width=True)

    st.markdown("#### Executive reasoning summary")
    arch_status = recommendation_payload.get("architecture_status", {})
    llm_payload_active = recommendation_payload.get("summary_source") == "openrouter_or_fallback" or bool(arch_status.get("llm_specialist_agents_requested"))
    if llm_payload_active:
        requested = arch_status.get("llm_specialist_agents_requested_list") or []
        reviewed = arch_status.get("llm_specialist_agents_reviewed") or []
        requested_text = ", ".join(requested) if requested else "none"
        reviewed_text = ", ".join(reviewed) if reviewed else "none completed"
        stats = arch_status.get("agent_review_stats") or {}
        attempted_count = int(stats.get("attempted") or 0)
        failed_count = int(stats.get("failed") or 0)
        successful_count = int(stats.get("successful") or 0)
        attempted_models = arch_status.get("last_attempted_models") or []
        attempted_routes = arch_status.get("last_attempted_routes") or []
        last_model = arch_status.get("last_model_used")
        if last_model:
            model_status = f"successful model: {last_model}"
        elif attempted_count > 0 or attempted_models or attempted_routes:
            model_status = "attempted, but no successful model response"
        else:
            model_status = "no LLM call attempted"
        llm_caption = (
            f"Provider: {arch_status.get('llm_provider', 'deterministic')} | "
            f"Model status: {model_status} | "
            f"Specialist reviews requested: {requested_text} | completed: {reviewed_text} | "
            f"attempted/succeeded/failed: {attempted_count}/{successful_count}/{failed_count}"
        )
    else:
        llm_caption = "Deterministic output shown. Use the OpenRouter LLM review job in the sidebar to request hybrid specialist-agent reasoning for this timestamp."
    st.caption(llm_caption)
    if llm_payload_active and not reviewed:
        last_error = arch_status.get("last_llm_error")
        attempted_routes = arch_status.get("last_attempted_routes") or []
        if last_error or attempted_routes:
            with st.expander("OpenRouter LLM diagnostics", expanded=False):
                st.write("The confidence gate selected specialist agents, but no specialist LLM review completed successfully for this result.")
                st.write(f"OpenRouter enabled: {arch_status.get('openrouter_reasoning_enabled')}")
                st.write(f"Configured API keys: {arch_status.get('openrouter_api_key_count')}")
                st.write(f"Configured free models: {', '.join(arch_status.get('openrouter_free_models') or [])}")
                st.write(f"Reliability mode: {arch_status.get('openrouter_reliability_mode')}")
                if arch_status.get("openrouter_key_health"):
                    st.write("Key health:")
                    st.json(arch_status.get("openrouter_key_health"))
                if attempted_routes:
                    st.write("Attempted routes:")
                    for route in attempted_routes[:12]:
                        st.write(f"- {route}")
                if last_error:
                    st.write(f"Last error: {last_error}")
                reviews = recommendation_payload.get("llm_agent_reviews") or []
                if reviews:
                    st.write("Per-agent LLM review status:")
                    st.dataframe(pd.DataFrame(clean_records_for_display(reviews)), use_container_width=True, hide_index=True)
    st.info(recommendation_payload.get("executive_summary", "No summary generated."))

    st.markdown("#### Recommendations requiring operator attention")
    recommendations = recommendation_payload.get("recommendations", [])
    for rec in recommendations:
        title = f"{rec.get('decision_area')} | {rec.get('risk_level', '').upper()} | confidence {float(rec.get('confidence', 0)):.2f} | {rec.get('automation_mode')}"
        with st.expander(title, expanded=rec.get("agent_name") == "CoordinatorAgent"):
            st.markdown(f"**Action:** {rec.get('action_summary')}")
            st.progress(min(max(float(rec.get("confidence", 0)), 0.0), 1.0), text="Confidence")
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown("**Action deltas / workflow actions**")
                st.dataframe(action_table(rec.get("action_deltas", {})), use_container_width=True, hide_index=True)
                st.markdown("**Reasoning**")
                st.write(rec.get("reasoning", "-"))
            with c_right:
                st.markdown("**Evidence**")
                for item in rec.get("evidence", [])[:10]:
                    st.write(f"- {item}")
                st.markdown("**Safety notes**")
                for item in rec.get("safety_notes", [])[:8]:
                    st.write(f"- {item}")
                st.markdown("**Prerequisites**")
                prereq = rec.get("prerequisites", [])
                if prereq:
                    for item in prereq[:8]:
                        st.write(f"- {item}")
                else:
                    st.write("- None specified")

            st.markdown("**Operator feedback**")
            with st.form(f"feedback_{rec.get('id')}"):
                decision = st.radio("Decision", ["accept", "reject", "modify"], horizontal=True, key=f"decision_{rec.get('id')}")
                modified_action = st.text_area("Modified action, if applicable", key=f"modified_{rec.get('id')}")
                notes = st.text_area("Operator notes", key=f"notes_{rec.get('id')}")
                operator_id = st.text_input("Operator ID", value="POC_OPERATOR", key=f"operator_{rec.get('id')}")
                submitted = st.form_submit_button("Log operator decision")
                if submitted:
                    try:
                        result = api_post(backend_url, f"/api/recommendations/{rec.get('id')}/feedback", params={"index": index}, json_payload={"decision": decision, "modified_action": modified_action, "notes": notes, "operator_id": operator_id})
                        api_get.clear()
                        st.success(f"Decision logged: {result['record']['decision']}")
                    except Exception as exc:
                        st.error(f"Could not log feedback: {exc}")

with agents_tab:
    st.subheader("Agent coordination bus")
    architecture_status = recommendation_payload.get("architecture_status", {})
    status_cols = st.columns(4)
    requested_count = len(architecture_status.get("llm_specialist_agents_requested_list") or [])
    completed_count = len(architecture_status.get("llm_specialist_agents_reviewed") or [])
    status_cols[0].metric("LLM provider", architecture_status.get("llm_provider", "none"))
    status_cols[1].metric("OpenRouter enabled", str(architecture_status.get("openrouter_reasoning_enabled", False)))
    status_cols[2].metric("Specialist reviews", f"{completed_count}/{requested_count}")
    status_cols[3].metric("Last free model", architecture_status.get("last_model_used") or "not used")
    with st.expander("Architecture and OpenRouter status", expanded=False):
        st.json(architecture_status)

    signals = recommendation_payload.get("signals", [])
    messages = recommendation_payload.get("agent_messages", [])
    recommendations = recommendation_payload.get("recommendations", [])

    st.markdown("#### Information-flow figure")
    render_agent_flow_figure(messages, signals, recommendations)

    st.markdown("#### Current inter-agent message cards")
    render_message_cards(messages)

    reviews = recommendation_payload.get("llm_agent_reviews", [])
    if reviews:
        st.markdown("#### OpenRouter specialist-agent reviews")
        st.dataframe(pd.DataFrame(clean_records_for_display(reviews)), use_container_width=True, hide_index=True)

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        st.markdown("#### Agent signals")
        display_cols = ["agent_name", "decision_area", "severity", "confidence", "message", "proposed_actions", "risk_tags"]
        signals_display = signals_df[[c for c in display_cols if c in signals_df.columns]].copy()
        if "message" in signals_display.columns:
            signals_display["message"] = signals_display["message"].apply(clean_operator_text)
        st.dataframe(signals_display, use_container_width=True, hide_index=True)
    messages_df = pd.DataFrame(messages)
    if not messages_df.empty:
        st.markdown("#### Inter-agent messages table")
        messages_display = messages_df.copy()
        if "content" in messages_display.columns:
            messages_display["content"] = messages_display["content"].apply(clean_operator_text)
        st.dataframe(messages_display, use_container_width=True, hide_index=True)
        agents = sorted(set(messages_df["from_agent"].astype(str)).union(set(messages_df["to_agent"].astype(str))))
        agent_index = {agent: i for i, agent in enumerate(agents)}
        link_conf = [max(float(v), 0.1) for v in messages_df["confidence"]]
        link_colors = ["rgba(220,38,38,0.45)" if v >= 0.9 else "rgba(234,88,12,0.38)" if v >= 0.75 else "rgba(37,99,235,0.30)" for v in link_conf]
        sankey = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node={
                        "label": [AGENT_SHORT_NAMES.get(agent, agent) for agent in agents],
                        "pad": 22,
                        "thickness": 20,
                        "line": {"color": "#475569", "width": 0.6},
                        "color": ["#e0f2fe" for _ in agents],
                    },
                    link={
                        "source": [agent_index[a] for a in messages_df["from_agent"].astype(str)],
                        "target": [agent_index[a] for a in messages_df["to_agent"].astype(str)],
                        "value": link_conf,
                        "color": link_colors,
                        "label": messages_df["message_type"].astype(str).tolist(),
                    },
                )
            ]
        )
        sankey.update_layout(title_text="Compact inter-agent information flow", height=520, font={"size": 13}, margin={"l": 10, "r": 10, "t": 45, "b": 10})
        st.plotly_chart(sankey, use_container_width=True)
    else:
        st.write("No inter-agent messages for the current timestamp.")

with cases_tab:
    st.subheader("Similar historical synthetic cases")
    cases = pd.DataFrame(recommendation_payload.get("similar_cases", []))
    if cases.empty:
        st.write("No similar cases found.")
    else:
        st.dataframe(cases, use_container_width=True, hide_index=True)
    st.markdown("#### Matched operating playbook")
    matches = pd.DataFrame(recommendation_payload.get("playbook_matches", []))
    if matches.empty:
        st.write("No playbook match for this timestamp.")
    else:
        st.dataframe(matches, use_container_width=True, hide_index=True)

with log_tab:
    st.subheader("Operator decision log")
    log_payload = api_get(backend_url, "/api/decision-log")
    log_df = pd.DataFrame(log_payload.get("records", []))
    if log_df.empty:
        st.write("No operator feedback has been logged yet.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Dataset and playbook")
    st.write({"rows": metadata["row_count"], "first_timestamp": metadata.get("first_timestamp"), "last_timestamp": metadata.get("last_timestamp"), "columns": len(metadata.get("columns", []))})
    with st.expander("Current raw row"):
        st.json(state)
    with st.expander("Operator playbook"):
        st.dataframe(pd.DataFrame(api_get(backend_url, "/api/playbook").get("records", [])), use_container_width=True, hide_index=True)
    with st.expander("Data dictionary"):
        st.dataframe(pd.DataFrame(api_get(backend_url, "/api/data-dictionary").get("records", [])), use_container_width=True, hide_index=True)
