from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import config
from backend.coordinator import AgentCoordinator
from backend.data_store import DataStore


def test_recommendations_payload() -> None:
    store = DataStore()
    coordinator = AgentCoordinator()
    context = store.get_context(700)
    payload = coordinator.run(context, include_llm_summary=False)
    assert payload["dataset_index"] == 700
    assert "recommendations" in payload
    assert len(payload["recommendations"]) >= 1
    assert "signals" in payload
    assert "agent_messages" in payload
    assert "architecture_status" in payload
    assert payload["architecture_status"]["llm_provider"] == "openrouter"


def test_openrouter_model_filter_keeps_only_free_models() -> None:
    assert config.OPENROUTER_FREE_MODELS
    assert all(model.endswith(":free") for model in config.OPENROUTER_FREE_MODELS)


if __name__ == "__main__":
    test_recommendations_payload()
    test_openrouter_model_filter_keeps_only_free_models()
    print("smoke test passed")
