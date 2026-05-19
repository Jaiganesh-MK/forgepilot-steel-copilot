# Agentic Architecture

## Decision agents

| Agent | Decision area | Main purpose |
|---|---|---|
| ThermalStateAgent | Thermal balance | Detect cold/hot drift and propose fuel/oxygen/blast-temperature support. |
| PermeabilityAgent | Gas flow and permeability | Detect pressure-drop, descent, tuyere, and permeability stress. |
| WindVolumeAgent | Wind volume | Recommend conservative wind-volume trims. |
| PCIAgent | PCI injection | Recommend coal-injection trims coordinated with coke and thermal state. |
| CokeRateAgent | Coke rate | Recommend coke support or coke reduction based on heat, permeability, and coke quality. |
| FuelRateAgent | Total fuel rate | Flag total fuel-rate deviations and force coke/PCI reconciliation. |
| OxygenEnrichmentAgent | Oxygen enrichment | Recommend small oxygen trims tied to RAFT and thermal state. |
| BlastTemperatureAgent | Hot blast temperature | Recommend low-disturbance thermal corrections if stove margin is available. |
| TopPressureAgent | Top pressure | Recommend small top-pressure changes during gas-flow stress. |
| BurdenDistributionAgent | Burden distribution | Recommend charging-program adjustments for permeability recovery. |
| TappingAgent | Tapping priority | Raise cast-house/tapping priority when hearth liquid level is high. |
| QualityAgent | Hot metal quality | Detect Si/S quality risk and data-quality concerns. |
| CoordinatorAgent | Integrated response | Merge proposals, resolve conflicts, apply safety gate, and produce the operator-facing recommendation package. |
| ReasoningSynthesizer | LLM router | Calls OpenRouter for optional specialist-agent JSON reviews and final operator summary synthesis. |

## Information flow

1. Frontend selects a timestamp index.
2. Backend creates a PlantContext with current row, recent history, playbook records, and similar cases.
3. Agents independently evaluate the context and create deterministic evidence/action scaffolds.
4. If specialist LLM mode is enabled/requested, every agent with deterministic confidence below `LLM_AGENT_CONFIDENCE_THRESHOLD` sends its scaffold to OpenRouter and receives strict JSON review. Additional active/high-risk agents can also be selected depending on `LLM_AGENT_MODE`.
5. The backend validates every LLM-returned action key and clamps numeric deltas to one-step advisory envelopes.
6. Agents publish signals and coordination messages.
7. Coordinator resolves overlapping or conflicting actions.
8. Safety gate labels recommendations as informational workflow, approval-required, or direct-action candidate.
9. Optional OpenRouter synthesis rewrites the final payload into operator-facing language.
10. Frontend displays the final recommendation, confidence, evidence, prerequisites, safety notes, OpenRouter status, and operator feedback controls.

## OpenRouter model and key routing

The LLM layer uses the OpenRouter chat completions endpoint. It is constrained to model IDs ending in `:free`.

Default pool:

```python
[
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-3-27b-it:free",
    "minimax/minimax-m2.5:free",
]
```

The default model strategy is `round_robin`. Each request starts with the next model in the pool. If that model fails and `OPENROUTER_ENABLE_MODEL_FALLBACKS=true`, the synthesizer tries the remaining free models before returning the deterministic fallback.

Multiple API keys are accepted through `OPENROUTER_API_KEYS`. The router can rotate/fallback across keys, but OpenRouter documents that multiple keys on the same account do not increase global account-level limits. Treat key rotation as resilience/key separation unless the keys genuinely belong to separately governed accounts.

## Hybrid specialist-agent review

Default confidence gate:

```text
If specialist LLM review is requested and agent.confidence < 0.90
        -> specialist agent calls OpenRouter
```

This gate overrides the normal / low-risk suppression rule. A stable furnace row can therefore still request LLM review when the rule-based confidence is below the threshold. With `LLM_AGENT_LOW_CONFIDENCE_OVERRIDES_LIMIT=true`, the max-agent cap does not suppress sub-threshold agents; it only limits extra active/high-risk agents above the threshold.

```text
PlantContext + deterministic agent scaffold
        |
        v
Confidence-gated or active specialist agent calls OpenRouter
        |
        v
LLM returns JSON only
        |
        v
Backend validates schema and clamps setpoint deltas
        |
        v
Coordinator resolves conflicts
        |
        v
Safety gate marks operator-approval requirement
```

The LLM is allowed to confirm, soften, remove, or refine a specialist signal. It cannot execute setpoints, bypass safety gating, use non-free model IDs, invent measurements, or return action keys outside the configured schema.

## Default safety principle

A recommendation may be high confidence and still require operator approval. In this POC, confidence is not treated as authority to act on process setpoints. It is treated as a prioritization and review signal.

The POC does not connect to DCS, PLC, SCADA, or any live control layer. All wind volume, PCI, coke rate, oxygen enrichment, blast temperature, top pressure, burden-distribution, and tapping changes remain advisory unless the safety flag is intentionally changed for labeling experiments.

## OpenRouter free-model efficiency pattern

The LLM path is optimized for OpenRouter free models as follows:

1. Deterministic agents run first and calculate confidence.
2. Agents below the configured confidence threshold are selected for LLM review.
3. Selected agents are batched into a single OpenRouter request.
4. The request uses OpenRouter's native `models` fallback array and can include `openrouter/free` as the first candidate.
5. Provider routing is set to prioritize throughput across models/providers.
6. The response is requested as strict JSON and optionally repaired by OpenRouter response-healing.
7. The deterministic safety gate still validates all actions before the operator dashboard displays recommendations.

This design is intentionally different from making one LLM call per specialist agent, because free-model request limits and variable latency make per-agent parallel calls unreliable.
