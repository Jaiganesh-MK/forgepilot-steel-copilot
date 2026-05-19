# ForgePilot Steel Operations Copilot - Streamlit Cloud POC

This repository is a single-process Streamlit version of the ForgePilot blast-furnace operator copilot POC. It runs the dashboard and embedded agent engine in the same Streamlit app, so no separate FastAPI backend or localhost service is required.

## What the app does

- Loads the synthetic blast furnace operations dataset from `data/`.
- Runs deterministic specialist agents for thermal state, permeability, wind volume, PCI, coke rate, fuel balance, oxygen enrichment, top pressure, burden distribution, tapping, quality, and safety.
- Uses a coordinator/orchestrator to resolve conflicts and generate operator recommendations.
- Optionally runs OpenRouter free-model LLM review for specialist agents and final explanation synthesis.
- Logs operator accept/reject/modify feedback locally in `data/operator_decision_log.csv`.

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the main file path to:

```text
app.py
```

4. Add OpenRouter configuration through Streamlit app secrets. Do not commit API keys to GitHub.

Example Streamlit secrets for the most reliable free-model mode:

```toml
USE_OPENROUTER = "true"
OPENROUTER_API_KEYS = "sk-or-v1-your-key"
OPENROUTER_FREE_MODELS = "openrouter/free,nvidia/nemotron-3-super-120b-a12b:free,openai/gpt-oss-120b:free,google/gemma-3-27b-it:free,minimax/minimax-m2.5:free"

# OpenRouter free-model routing
OPENROUTER_NATIVE_MODEL_FALLBACKS = "true"
OPENROUTER_ENABLE_MODEL_FALLBACKS = "true"
OPENROUTER_USE_FREE_ROUTER = "true"
OPENROUTER_FREE_ROUTER_FIRST = "false"
OPENROUTER_PROVIDER_SORT = "throughput"
OPENROUTER_PROVIDER_SORT_PARTITION = "none"
OPENROUTER_USE_STRUCTURED_OUTPUTS = "true"
OPENROUTER_AGENT_STRUCTURED_OUTPUTS = "false"
OPENROUTER_USE_RESPONSE_HEALING = "true"

# Split specialist agents into small domain batches with different free-model fallback chains
OPENROUTER_AGENT_ROUTING_MODE = "grouped"
OPENROUTER_AGENT_THERMAL_MODELS = "google/gemma-3-27b-it:free,openai/gpt-oss-120b:free,openrouter/free"
OPENROUTER_AGENT_FLOW_MODELS = "google/gemma-3-27b-it:free,minimax/minimax-m2.5:free,openrouter/free"
OPENROUTER_AGENT_FUEL_MODELS = "openai/gpt-oss-120b:free,nvidia/nemotron-3-super-120b-a12b:free,openrouter/free"
OPENROUTER_AGENT_QUALITY_MODELS = "minimax/minimax-m2.5:free,google/gemma-3-27b-it:free,openrouter/free"

OPENROUTER_CONNECT_TIMEOUT_SECONDS = "3"
OPENROUTER_TIMEOUT_SECONDS = "22"
OPENROUTER_TOTAL_TIMEOUT_SECONDS = "25"
OPENROUTER_AGENT_BATCH_MAX_TOKENS = "900"

LLM_AGENT_CONFIDENCE_THRESHOLD = "0.90"
LLM_AGENT_REVIEW_LOW_CONFIDENCE = "true"
LLM_AGENT_LOW_CONFIDENCE_OVERRIDES_LIMIT = "true"
LLM_AGENT_BATCH_REVIEWS = "true"
LLM_AGENT_BATCH_SIZE = "12"
LLM_AGENT_TOTAL_TIMEOUT_SECONDS = "90"
LLM_AGENT_MAX_WORKERS = "1"

ALLOW_DIRECT_SETPOINT_ACTIONS = "false"
```

The grouped setting makes different agent families use different free-model fallback chains. It also avoids strict structured output for specialist-agent reviews because that can reduce the number of eligible free models.

## Notes

- The app is advisory only by default. Setpoint recommendations require operator approval unless explicitly reconfigured.
- The LLM review job is asynchronous. The deterministic dashboard loads first; OpenRouter review can be triggered for a selected timestamp.
- Free OpenRouter models may still be rate-limited or unavailable. The deterministic agent output remains available if LLM calls fail.
