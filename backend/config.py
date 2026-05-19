from __future__ import annotations

import ast
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DECISION_LOG_PATH = DATA_DIR / "operator_decision_log.csv"

if load_dotenv:
    load_dotenv(BASE_DIR / ".env")


def _load_streamlit_secrets_into_env() -> None:
    """Expose Streamlit Cloud secrets as env vars for backend modules.

    Streamlit Cloud keeps secrets in st.secrets. In many deployments top-level
    TOML keys are also reflected as environment variables, but this helper makes
    the behavior explicit and avoids silent OpenRouter misconfiguration.
    """
    try:
        import streamlit as st  # type: ignore
        secrets = getattr(st, "secrets", {})
        if not secrets:
            return
        for key, value in secrets.items():
            if key not in os.environ and isinstance(value, (str, int, float, bool)):
                os.environ[key] = str(value)
    except Exception:
        return

_load_streamlit_secrets_into_env()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


DEFAULT_OPENROUTER_FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-3-27b-it:free",
    "minimax/minimax-m2.5:free",
]


def _parse_list(value: str | None) -> list[str]:
    requested: list[str] = []
    if value:
        candidate = value.strip()
        # Accept Python/JSON list syntax, for example:
        # OPENROUTER_API_KEYS=["sk-or-v1-...", "sk-or-v1-..."]
        # OPENROUTER_FREE_MODELS=["model-a:free", "model-b:free"]
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                if isinstance(parsed, (list, tuple)):
                    requested = [str(item).strip() for item in parsed if str(item).strip()]
                    break
            except Exception:
                pass
        if not requested:
            cleaned = candidate.replace(";", ",").replace("\n", ",")
            cleaned = cleaned.strip().strip("[]")
            requested = [item.strip().strip('"').strip("'") for item in cleaned.split(",") if item.strip()]
    return requested


def _parse_requested_models(value: str | None) -> list[str]:
    return _parse_list(value) or DEFAULT_OPENROUTER_FREE_MODELS.copy()


def _is_free_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.endswith(":free") or normalized == "openrouter/free"


def _looks_like_placeholder_key(value: str) -> bool:
    value = value.strip()
    if not value:
        return True
    lowered = value.lower()
    return lowered.startswith("put_") or lowered.startswith("your_") or "placeholder" in lowered or "api_key_here" in lowered


# Optional OpenRouter-backed LLM layer. Deterministic agents still run without a key.
# Use either OPENROUTER_API_KEY for one key, or OPENROUTER_API_KEYS for a comma/JSON list.
_single_key = os.getenv("OPENROUTER_API_KEY", "PUT_YOUR_OPENROUTER_API_KEY_HERE")
_multi_keys_raw = os.getenv("OPENROUTER_API_KEYS")
_key_candidates = _parse_list(_multi_keys_raw) if _multi_keys_raw else []
if _single_key and not _looks_like_placeholder_key(_single_key):
    _key_candidates.insert(0, _single_key.strip())
_seen_keys: set[str] = set()
OPENROUTER_API_KEYS = []
for key in _key_candidates:
    if key and not _looks_like_placeholder_key(key) and key not in _seen_keys:
        _seen_keys.add(key)
        OPENROUTER_API_KEYS.append(key)
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else _single_key

USE_OPENROUTER = _as_bool(os.getenv("USE_OPENROUTER", "false"))
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
_requested_models_raw = os.getenv("OPENROUTER_FREE_MODELS", os.getenv("OPENROUTER_MODELS", os.getenv("MODELS")))
OPENROUTER_REQUESTED_MODELS = _parse_requested_models(_requested_models_raw)
OPENROUTER_FREE_MODELS = [model for model in OPENROUTER_REQUESTED_MODELS if _is_free_model(model)]
OPENROUTER_REJECTED_NON_FREE_MODELS = [model for model in OPENROUTER_REQUESTED_MODELS if not _is_free_model(model)]
if not OPENROUTER_FREE_MODELS:
    OPENROUTER_FREE_MODELS = DEFAULT_OPENROUTER_FREE_MODELS.copy()
OPENROUTER_MODELS = OPENROUTER_FREE_MODELS  # Backward-compatible alias used by older code.

# OpenRouter routing strategy. The efficient mode uses OpenRouter native fallbacks
# and the free router instead of making one HTTP request per model.
OPENROUTER_FREE_ROUTER_MODEL = os.getenv("OPENROUTER_FREE_ROUTER_MODEL", "openrouter/free").strip()
OPENROUTER_USE_FREE_ROUTER = _as_bool(os.getenv("OPENROUTER_USE_FREE_ROUTER", "true"), True)
OPENROUTER_FREE_ROUTER_FIRST = _as_bool(os.getenv("OPENROUTER_FREE_ROUTER_FIRST", "true"), True)
OPENROUTER_NATIVE_MODEL_FALLBACKS = _as_bool(os.getenv("OPENROUTER_NATIVE_MODEL_FALLBACKS", "true"), True)
OPENROUTER_PROVIDER_SORT = os.getenv("OPENROUTER_PROVIDER_SORT", "throughput").strip().lower()
if OPENROUTER_PROVIDER_SORT not in {"price", "throughput", "latency", "none"}:
    OPENROUTER_PROVIDER_SORT = "throughput"
OPENROUTER_PROVIDER_SORT_PARTITION = os.getenv("OPENROUTER_PROVIDER_SORT_PARTITION", "none").strip().lower()
if OPENROUTER_PROVIDER_SORT_PARTITION not in {"model", "none"}:
    OPENROUTER_PROVIDER_SORT_PARTITION = "none"
OPENROUTER_PREFERRED_MAX_LATENCY_P90 = _as_float(os.getenv("OPENROUTER_PREFERRED_MAX_LATENCY_P90"), 12.0)
OPENROUTER_PREFERRED_MIN_THROUGHPUT_P90 = _as_float(os.getenv("OPENROUTER_PREFERRED_MIN_THROUGHPUT_P90"), 0.0)
OPENROUTER_REQUIRE_PARAMETERS = _as_bool(os.getenv("OPENROUTER_REQUIRE_PARAMETERS", "false"), False)
OPENROUTER_USE_STRUCTURED_OUTPUTS = _as_bool(os.getenv("OPENROUTER_USE_STRUCTURED_OUTPUTS", "true"), True)
OPENROUTER_USE_RESPONSE_HEALING = _as_bool(os.getenv("OPENROUTER_USE_RESPONSE_HEALING", "true"), True)

# Free-model reliability mode for specialist-agent reviews. Strict JSON-schema
# routing can reduce the eligible free-model pool; for deployed demos, prompt-only
# JSON is often more reliable while still being parsed and validated locally.
OPENROUTER_AGENT_STRUCTURED_OUTPUTS = _as_bool(os.getenv("OPENROUTER_AGENT_STRUCTURED_OUTPUTS", "false"), False)
OPENROUTER_AGENT_ROUTING_MODE = os.getenv("OPENROUTER_AGENT_ROUTING_MODE", "grouped").strip().lower()
if OPENROUTER_AGENT_ROUTING_MODE not in {"simple_free_router", "single_batch", "grouped", "single_agent"}:
    OPENROUTER_AGENT_ROUTING_MODE = "simple_free_router"

def _free_model_list(value: str | None, default: list[str]) -> list[str]:
    parsed = _parse_list(value) if value else default.copy()
    cleaned = [model for model in parsed if _is_free_model(model)]
    return cleaned or default.copy()

OPENROUTER_AGENT_THERMAL_MODELS = _free_model_list(
    os.getenv("OPENROUTER_AGENT_THERMAL_MODELS"),
    ["google/gemma-3-27b-it:free", "openai/gpt-oss-120b:free", "openrouter/free"],
)
OPENROUTER_AGENT_FLOW_MODELS = _free_model_list(
    os.getenv("OPENROUTER_AGENT_FLOW_MODELS"),
    ["google/gemma-3-27b-it:free", "minimax/minimax-m2.5:free", "openrouter/free"],
)
OPENROUTER_AGENT_FUEL_MODELS = _free_model_list(
    os.getenv("OPENROUTER_AGENT_FUEL_MODELS"),
    ["openai/gpt-oss-120b:free", "nvidia/nemotron-3-super-120b-a12b:free", "openrouter/free"],
)
OPENROUTER_AGENT_QUALITY_MODELS = _free_model_list(
    os.getenv("OPENROUTER_AGENT_QUALITY_MODELS"),
    ["minimax/minimax-m2.5:free", "google/gemma-3-27b-it:free", "openrouter/free"],
)
OPENROUTER_AGENT_DEFAULT_MODELS = _free_model_list(
    os.getenv("OPENROUTER_AGENT_DEFAULT_MODELS"),
    ["openrouter/free", "openai/gpt-oss-120b:free", "google/gemma-3-27b-it:free"],
)

OPENROUTER_MODEL_SELECTION = os.getenv("OPENROUTER_MODEL_SELECTION", os.getenv("OPENROUTER_MODEL_STRATEGY", "round_robin")).strip().lower()
if OPENROUTER_MODEL_SELECTION not in {"round_robin", "fallback", "random"}:
    OPENROUTER_MODEL_SELECTION = "round_robin"
OPENROUTER_KEY_SELECTION = os.getenv("OPENROUTER_KEY_SELECTION", "round_robin").strip().lower()
if OPENROUTER_KEY_SELECTION not in {"round_robin", "fallback", "random"}:
    OPENROUTER_KEY_SELECTION = "round_robin"
OPENROUTER_ENABLE_MODEL_FALLBACKS = _as_bool(os.getenv("OPENROUTER_ENABLE_MODEL_FALLBACKS", "true"), True)
OPENROUTER_MAX_MODEL_ATTEMPTS = max(1, _as_int(os.getenv("OPENROUTER_MAX_MODEL_ATTEMPTS"), min(2, len(OPENROUTER_FREE_MODELS))))
OPENROUTER_MAX_KEY_ATTEMPTS = max(1, _as_int(os.getenv("OPENROUTER_MAX_KEY_ATTEMPTS"), min(2, max(len(OPENROUTER_API_KEYS), 1))))
# Keep LLM synthesis bounded so the Streamlit dashboard does not stall when free models are slow.
OPENROUTER_CONNECT_TIMEOUT_SECONDS = _as_float(os.getenv("OPENROUTER_CONNECT_TIMEOUT_SECONDS"), 3.0)
OPENROUTER_TIMEOUT_SECONDS = _as_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", os.getenv("OPENROUTER_REQUEST_TIMEOUT_SECS")), 20.0)
OPENROUTER_TOTAL_TIMEOUT_SECONDS = _as_float(os.getenv("OPENROUTER_TOTAL_TIMEOUT_SECONDS"), 25.0)
OPENROUTER_MAX_TOKENS = _as_int(os.getenv("OPENROUTER_MAX_TOKENS"), 350)
OPENROUTER_AGENT_MAX_TOKENS = _as_int(os.getenv("OPENROUTER_AGENT_MAX_TOKENS"), 900)
OPENROUTER_AGENT_BATCH_MAX_TOKENS = _as_int(os.getenv("OPENROUTER_AGENT_BATCH_MAX_TOKENS"), 900)
OPENROUTER_RELIABILITY_MODE = os.getenv("OPENROUTER_RELIABILITY_MODE", "simple_free_router").strip().lower()
if OPENROUTER_RELIABILITY_MODE not in {"simple_free_router", "advanced"}:
    OPENROUTER_RELIABILITY_MODE = "simple_free_router"
OPENROUTER_ACCEPT_TEXT_FALLBACK = _as_bool(os.getenv("OPENROUTER_ACCEPT_TEXT_FALLBACK", "true"), True)
OPENROUTER_INCLUDE_KEY_HEALTH_CHECK = _as_bool(os.getenv("OPENROUTER_INCLUDE_KEY_HEALTH_CHECK", "true"), True)
OPENROUTER_TEMPERATURE = _as_float(os.getenv("OPENROUTER_TEMPERATURE"), 0.15)
OPENROUTER_AGENT_TEMPERATURE = _as_float(os.getenv("OPENROUTER_AGENT_TEMPERATURE"), 0.05)
OPENROUTER_PROMPT_CHAR_LIMIT = _as_int(os.getenv("OPENROUTER_PROMPT_CHAR_LIMIT"), 8000)
OPENROUTER_AGENT_PROMPT_CHAR_LIMIT = _as_int(os.getenv("OPENROUTER_AGENT_PROMPT_CHAR_LIMIT"), 6500)
OPENROUTER_CACHE_RESPONSES = _as_bool(os.getenv("OPENROUTER_CACHE_RESPONSES", "true"), True)
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"))
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", os.getenv("OPENROUTER_APP_NAME", "Blast Furnace Agentic Operator Copilot POC"))

# True hybrid-agent mode. When enabled/requested, selected specialist agents call
# OpenRouter themselves to validate/refine their own signal before coordination.
USE_LLM_AGENTS = _as_bool(os.getenv("USE_LLM_AGENTS", "false"), False)
LLM_AGENT_MODE = os.getenv("LLM_AGENT_MODE", "active_only").strip().lower()
if LLM_AGENT_MODE not in {"active_only", "all", "high_risk_only"}:
    LLM_AGENT_MODE = "active_only"
LLM_AGENT_MAX_AGENTS = max(1, _as_int(os.getenv("LLM_AGENT_MAX_AGENTS"), 12))
LLM_AGENT_MAX_WORKERS = max(1, _as_int(os.getenv("LLM_AGENT_MAX_WORKERS"), 1))
LLM_AGENT_TOTAL_TIMEOUT_SECONDS = max(2.0, _as_float(os.getenv("LLM_AGENT_TOTAL_TIMEOUT_SECONDS"), 45.0))
LLM_AGENT_BATCH_REVIEWS = _as_bool(os.getenv("LLM_AGENT_BATCH_REVIEWS", "true"), True)
LLM_AGENT_BATCH_SIZE = max(1, _as_int(os.getenv("LLM_AGENT_BATCH_SIZE"), 12))
LLM_AGENT_MIN_SEVERITY = os.getenv("LLM_AGENT_MIN_SEVERITY", "medium").strip().lower()
if LLM_AGENT_MIN_SEVERITY not in {"low", "medium", "high", "critical"}:
    LLM_AGENT_MIN_SEVERITY = "medium"
LLM_AGENT_CALL_ON_NORMAL_LOW_RISK = _as_bool(os.getenv("LLM_AGENT_CALL_ON_NORMAL_LOW_RISK", "false"), False)

# Confidence-gated hybrid behavior. When specialist-agent review is requested,
# any deterministic agent with confidence below this threshold is sent to
# OpenRouter even during normal / low-risk plant states. Default is 0.90 because
# the POC's rule-based agents intentionally remain conservative unless the
# evidence is very strong.
LLM_AGENT_CONFIDENCE_THRESHOLD = max(0.0, min(1.0, _as_float(os.getenv("LLM_AGENT_CONFIDENCE_THRESHOLD"), 0.90)))
LLM_AGENT_REVIEW_LOW_CONFIDENCE = _as_bool(os.getenv("LLM_AGENT_REVIEW_LOW_CONFIDENCE", "true"), True)
LLM_AGENT_LOW_CONFIDENCE_OVERRIDES_LIMIT = _as_bool(os.getenv("LLM_AGENT_LOW_CONFIDENCE_OVERRIDES_LIMIT", "true"), True)

ALLOW_DIRECT_SETPOINT_ACTIONS = _as_bool(os.getenv("ALLOW_DIRECT_SETPOINT_ACTIONS", "false"))
DIRECT_ACTION_CONFIDENCE_THRESHOLD = _as_float(os.getenv("DIRECT_ACTION_CONFIDENCE_THRESHOLD"), 0.92)

DATASET_FILE = DATA_DIR / "blast_furnace_synthetic_operations_60d_hourly.csv"
PLAYBOOK_FILE = DATA_DIR / "blast_furnace_operator_playbook.csv"
DATA_DICTIONARY_FILE = DATA_DIR / "blast_furnace_synthetic_data_dictionary.csv"
METADATA_FILE = DATA_DIR / "blast_furnace_dataset_metadata.json"

ACTION_LIMITS = {
    "wind_volume_delta_nm3_min": (-250, 250),
    "pci_delta_kg_thm": (-25, 25),
    "coke_rate_delta_kg_thm": (-25, 25),
    "oxygen_enrichment_delta_pct": (-0.8, 0.8),
    "blast_temp_delta_c": (-30, 30),
    "top_pressure_delta_kpa": (-15, 15),
}

SETPOINT_ACTION_KEYS = set(ACTION_LIMITS.keys()) | {"burden_distribution_change", "tapping_priority"}

TARGETS = {
    "hot_metal_temp_c": (1480.0, 1510.0),
    "hot_metal_si_pct": (0.38, 0.62),
    "permeability_index": (82.0, 100.0),
    "gas_utilization_pct": (46.0, 51.5),
    "pressure_drop_kpa": (130.0, 180.0),
    "thermal_state_index": (-0.65, 0.65),
    "hearth_liquid_level_index": (35.0, 78.0),
}
