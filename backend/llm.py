from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import replace
from typing import Any

import requests

from . import config
from .agent_core import AgentSignal, PlantContext, clamp, safe_float


class ReasoningSynthesizer:
    """OpenRouter LLM router for summary and hybrid specialist-agent review.

    Efficient OpenRouter mode:
    - Uses OpenRouter native `models` fallback array instead of local per-model loops.
    - Can route through `openrouter/free` so OpenRouter selects an available free model.
    - Uses provider throughput/latency preferences for faster free-model routing.
    - Batches specialist-agent reviews into one JSON request per timestamp.
    - Uses structured outputs + response-healing where available to reduce invalid JSON.
    """

    def __init__(self) -> None:
        self.models = list(config.OPENROUTER_FREE_MODELS)
        self.api_keys = list(config.OPENROUTER_API_KEYS)
        self.enabled = config.USE_OPENROUTER and bool(self.api_keys) and bool(self.models)
        self._lock = threading.Lock()
        self._rotation_index = 0
        self._key_rotation_index = 0
        self._cache: dict[str, dict[str, str]] = {}
        self.last_successful_model: str | None = None
        self.last_successful_key: str | None = None
        self.last_attempted_models: list[str] = []
        self.last_attempted_keys: list[str] = []
        self.last_attempted_routes: list[str] = []
        self.last_error: str | None = None
        self.agent_review_stats: dict[str, Any] = {
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "last_reviews": [],
        }

    def status(self) -> dict[str, Any]:
        return {
            "llm_provider": "openrouter",
            "openrouter_reasoning_enabled": self.enabled,
            "openrouter_base_url": config.OPENROUTER_BASE_URL,
            "openrouter_model_selection": config.OPENROUTER_MODEL_SELECTION,
            "openrouter_key_selection": config.OPENROUTER_KEY_SELECTION,
            "openrouter_model_fallbacks_enabled": config.OPENROUTER_ENABLE_MODEL_FALLBACKS,
            "openrouter_native_model_fallbacks": config.OPENROUTER_NATIVE_MODEL_FALLBACKS,
            "openrouter_use_free_router": config.OPENROUTER_USE_FREE_ROUTER,
            "openrouter_free_router_first": config.OPENROUTER_FREE_ROUTER_FIRST,
            "openrouter_free_router_model": config.OPENROUTER_FREE_ROUTER_MODEL,
            "openrouter_provider_sort": config.OPENROUTER_PROVIDER_SORT,
            "openrouter_provider_sort_partition": config.OPENROUTER_PROVIDER_SORT_PARTITION,
            "openrouter_preferred_max_latency_p90": config.OPENROUTER_PREFERRED_MAX_LATENCY_P90,
            "openrouter_preferred_min_throughput_p90": config.OPENROUTER_PREFERRED_MIN_THROUGHPUT_P90,
            "openrouter_require_parameters": config.OPENROUTER_REQUIRE_PARAMETERS,
            "openrouter_use_structured_outputs": config.OPENROUTER_USE_STRUCTURED_OUTPUTS,
            "openrouter_agent_structured_outputs": config.OPENROUTER_AGENT_STRUCTURED_OUTPUTS,
            "openrouter_agent_routing_mode": config.OPENROUTER_AGENT_ROUTING_MODE,
            "openrouter_reliability_mode": config.OPENROUTER_RELIABILITY_MODE,
            "openrouter_accept_text_fallback": config.OPENROUTER_ACCEPT_TEXT_FALLBACK,
            "openrouter_agent_thermal_models": config.OPENROUTER_AGENT_THERMAL_MODELS,
            "openrouter_agent_flow_models": config.OPENROUTER_AGENT_FLOW_MODELS,
            "openrouter_agent_fuel_models": config.OPENROUTER_AGENT_FUEL_MODELS,
            "openrouter_agent_quality_models": config.OPENROUTER_AGENT_QUALITY_MODELS,
            "openrouter_use_response_healing": config.OPENROUTER_USE_RESPONSE_HEALING,
            "openrouter_cache_responses": config.OPENROUTER_CACHE_RESPONSES,
            "openrouter_free_models": self.models,
            "free_model_pool": self.models,
            "openrouter_rejected_non_free_models": config.OPENROUTER_REJECTED_NON_FREE_MODELS,
            "openrouter_api_key_count": len(self.api_keys),
            "openrouter_redacted_api_keys": [self._redact_key(key) for key in self.api_keys],
            "openrouter_max_model_attempts": config.OPENROUTER_MAX_MODEL_ATTEMPTS,
            "openrouter_max_key_attempts": config.OPENROUTER_MAX_KEY_ATTEMPTS,
            "openrouter_per_model_timeout_seconds": config.OPENROUTER_TIMEOUT_SECONDS,
            "openrouter_total_timeout_seconds": config.OPENROUTER_TOTAL_TIMEOUT_SECONDS,
            "openrouter_max_tokens": config.OPENROUTER_MAX_TOKENS,
            "openrouter_agent_max_tokens": config.OPENROUTER_AGENT_MAX_TOKENS,
            "openrouter_agent_batch_max_tokens": config.OPENROUTER_AGENT_BATCH_MAX_TOKENS,
            "openrouter_temperature": config.OPENROUTER_TEMPERATURE,
            "openrouter_agent_temperature": config.OPENROUTER_AGENT_TEMPERATURE,
            "llm_specialist_agents_config_enabled": config.USE_LLM_AGENTS,
            "llm_agent_mode": config.LLM_AGENT_MODE,
            "llm_agent_reasoning_mode": config.LLM_AGENT_REASONING_MODE,
            "llm_agent_max_agents": config.LLM_AGENT_MAX_AGENTS,
            "llm_agent_max_workers": config.LLM_AGENT_MAX_WORKERS,
            "llm_agent_total_timeout_seconds": config.LLM_AGENT_TOTAL_TIMEOUT_SECONDS,
            "llm_agent_batch_reviews": config.LLM_AGENT_BATCH_REVIEWS,
            "llm_agent_batch_size": config.LLM_AGENT_BATCH_SIZE,
            "llm_agent_min_severity": config.LLM_AGENT_MIN_SEVERITY,
            "last_successful_model": self.last_successful_model,
            "last_model_used": self.last_successful_model,
            "last_successful_key": self.last_successful_key,
            "last_attempted_models": self.last_attempted_models,
            "last_attempted_keys": self.last_attempted_keys,
            "last_attempted_routes": self.last_attempted_routes,
            "last_llm_error": self.last_error,
            "agent_review_stats": self.agent_review_stats,
        }

    @staticmethod
    def _metadata_is_llm_first_mode(metadata: dict[str, Any] | None) -> bool:
        mode = str((metadata or {}).get("llm_reasoning_mode") or "").strip().lower()
        return mode in {"llm_first", "langgraph_llm_first"}

    @staticmethod
    def _is_llm_first_mode(signal: AgentSignal) -> bool:
        return ReasoningSynthesizer._metadata_is_llm_first_mode(signal.metadata or {})

    def key_health(self) -> dict[str, Any]:
        if not self.api_keys:
            return {"ok": False, "error": "No OpenRouter API key configured"}
        url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/key"
        key = self.api_keys[0]
        try:
            response = requests.get(url, headers=self._headers(key), timeout=(3, 8))
            data = response.json() if response.text else {}
            if response.status_code < 200 or response.status_code >= 300:
                return {"ok": False, "status_code": response.status_code, "error": data}
            return {"ok": True, "status_code": response.status_code, "data": data}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def deterministic_summary(self, payload: dict[str, Any]) -> str:
        return self._fallback_summary(payload)

    def synthesize(self, payload: dict[str, Any]) -> str:
        fallback = self._fallback_summary(payload)
        if not self.enabled:
            return fallback

        prompt = self._build_summary_prompt(payload)
        system_prompt = (
            "You are an industrial blast furnace operator decision-support copilot. "
            "You summarize coordinated agent outputs only. Do not invent measurements, "
            "do not remove approval requirements, and do not recommend closed-loop setpoint action."
        )
        try:
            result = self._invoke_openrouter(
                prompt=prompt,
                system_prompt=system_prompt,
                purpose="summary",
                max_tokens=config.OPENROUTER_MAX_TOKENS,
                temperature=config.OPENROUTER_TEMPERATURE,
                response_schema=None,
                total_timeout_seconds=config.OPENROUTER_TOTAL_TIMEOUT_SECONDS,
            )
            cleaned = self._sanitize_summary_text(result.get("text", ""), fallback)
            return cleaned
        except requests.exceptions.Timeout:
            return fallback + "\n\nOpenRouter synthesis timed out; deterministic agent summary shown instead."
        except Exception as exc:
            self.last_error = str(exc)
            return fallback + f"\n\nOpenRouter synthesis unavailable; deterministic agent summary shown instead. Last error: {exc}"

    def review_agent_signal(self, signal: AgentSignal, context: PlantContext) -> AgentSignal:
        """Review one signal. Kept for compatibility; batch review is preferred."""
        if not self.enabled:
            return signal

        self.agent_review_stats["attempted"] = int(self.agent_review_stats.get("attempted", 0)) + 1
        prompt = self._build_agent_review_prompt(signal, context)
        system_prompt = self._agent_system_prompt(signal)
        try:
            result = self._invoke_openrouter(
                prompt=prompt,
                system_prompt=system_prompt,
                purpose=f"agent_review:{signal.agent_name}",
                max_tokens=config.OPENROUTER_AGENT_MAX_TOKENS,
                temperature=config.OPENROUTER_AGENT_TEMPERATURE,
                response_schema=self._single_agent_response_schema(),
                total_timeout_seconds=config.OPENROUTER_TOTAL_TIMEOUT_SECONDS,
            )
            data = self._parse_json_response(result["text"])
            refined = self._merge_agent_review(signal, context, data, result)
            self.agent_review_stats["successful"] = int(self.agent_review_stats.get("successful", 0)) + 1
            self._record_agent_review(refined.agent_name, True, result.get("model"), None)
            return refined
        except Exception as exc:
            self.agent_review_stats["failed"] = int(self.agent_review_stats.get("failed", 0)) + 1
            self.last_error = f"{signal.agent_name}: {exc}"
            self._record_agent_review(signal.agent_name, False, None, str(exc))
            metadata = dict(signal.metadata or {})
            metadata.update({"llm_used": False, "llm_error": str(exc), "decision_basis": "deterministic_rules_fallback"})
            return replace(signal, metadata=metadata)

    def review_agent_signals_batch(
        self,
        signals: list[AgentSignal],
        context: PlantContext,
        timeout_seconds: float | None = None,
    ) -> list[AgentSignal]:
        """Review specialist signals through OpenRouter.

        Reliability strategy for free models:
        - Default grouped mode sends 2-4 small domain batches instead of one large
          strict-schema request or 10 parallel requests.
        - Each group has a different free-model fallback chain, so thermal/flow/fuel/quality
          reviews do not all depend on one free model.
        - JSON is requested in the prompt and parsed/validated locally; strict schema
          is optional because it can shrink the eligible free-model pool.
        """
        if not self.enabled or not signals:
            return signals

        max_batch = max(1, int(config.LLM_AGENT_BATCH_SIZE))
        signals = signals[:max_batch]

        if config.OPENROUTER_RELIABILITY_MODE == "simple_free_router" or config.OPENROUTER_AGENT_ROUTING_MODE == "simple_free_router":
            return self._review_agent_signals_batch_core(
                signals,
                context,
                timeout_seconds=timeout_seconds or config.LLM_AGENT_TOTAL_TIMEOUT_SECONDS,
                model_candidates=[config.OPENROUTER_FREE_ROUTER_MODEL],
                group_name="simple_free_router",
            )

        if config.OPENROUTER_AGENT_ROUTING_MODE == "grouped" and len(signals) > 1:
            return self._review_agent_signal_groups(signals, context, timeout_seconds=timeout_seconds)

        if config.OPENROUTER_AGENT_ROUTING_MODE == "single_agent" and len(signals) > 1:
            reviewed: list[AgentSignal] = []
            per_agent_budget = max(4.0, float(timeout_seconds or config.LLM_AGENT_TOTAL_TIMEOUT_SECONDS) / max(len(signals), 1))
            for signal in signals:
                models = self._models_for_agent_group(self._agent_group_name(signal.agent_name))
                reviewed.extend(
                    self._review_agent_signals_batch_core(
                        [signal],
                        context,
                        timeout_seconds=per_agent_budget,
                        model_candidates=models,
                        group_name=self._agent_group_name(signal.agent_name),
                    )
                )
            return reviewed

        return self._review_agent_signals_batch_core(
            signals,
            context,
            timeout_seconds=timeout_seconds or config.LLM_AGENT_TOTAL_TIMEOUT_SECONDS,
            model_candidates=None,
            group_name="single_batch",
        )

    def _review_agent_signal_groups(
        self,
        signals: list[AgentSignal],
        context: PlantContext,
        timeout_seconds: float | None = None,
    ) -> list[AgentSignal]:
        groups: dict[str, list[AgentSignal]] = {}
        for signal in signals:
            groups.setdefault(self._agent_group_name(signal.agent_name), []).append(signal)

        total_budget = max(8.0, float(timeout_seconds or config.LLM_AGENT_TOTAL_TIMEOUT_SECONDS))
        # Keep a useful budget per group while allowing smaller groups to complete.
        per_group_budget = max(8.0, min(config.OPENROUTER_TIMEOUT_SECONDS, total_budget / max(len(groups), 1) + 2.0))
        reviewed_by_agent: dict[str, AgentSignal] = {}
        for group_name, group_signals in groups.items():
            models = self._models_for_agent_group(group_name)
            reviewed = self._review_agent_signals_batch_core(
                group_signals,
                context,
                timeout_seconds=per_group_budget,
                model_candidates=models,
                group_name=group_name,
            )
            for signal in reviewed:
                reviewed_by_agent[signal.agent_name] = signal

        return [reviewed_by_agent.get(signal.agent_name, signal) for signal in signals]

    def _review_agent_signals_batch_core(
        self,
        signals: list[AgentSignal],
        context: PlantContext,
        timeout_seconds: float | None,
        model_candidates: list[str] | None,
        group_name: str,
    ) -> list[AgentSignal]:
        if not signals:
            return []
        self.agent_review_stats["attempted"] = int(self.agent_review_stats.get("attempted", 0)) + len(signals)

        prompt = self._build_agent_batch_review_prompt(signals, context)
        if any(self._is_llm_first_mode(signal) for signal in signals):
            system_prompt = (
                "You are a team of specialist blast furnace AI decision agents. "
                "Reason from the supplied plant state, trends, allowed action bounds, and each agent's responsibility. "
                "Do not rely on deterministic recommendations. Return concise JSON only. "
                "Never invent measurements or claim automatic execution."
            )
        else:
            system_prompt = (
                "You are a blast furnace operations review copilot. "
                "Check deterministic specialist agent recommendations for consistency and safety. "
                "Prefer concise JSON. Do not invent measurements or claim automatic execution."
            )
        response_schema = self._batch_agent_response_schema() if config.OPENROUTER_AGENT_STRUCTURED_OUTPUTS else None
        try:
            result = self._invoke_openrouter(
                prompt=prompt,
                system_prompt=system_prompt,
                purpose=f"agent_group_review:{group_name}:" + ",".join(signal.agent_name for signal in signals),
                max_tokens=config.OPENROUTER_AGENT_BATCH_MAX_TOKENS,
                temperature=config.OPENROUTER_AGENT_TEMPERATURE,
                response_schema=response_schema,
                total_timeout_seconds=timeout_seconds or config.LLM_AGENT_TOTAL_TIMEOUT_SECONDS,
                model_candidates_override=model_candidates,
            )
            try:
                data = self._parse_json_response(result["text"])
                reviews = data.get("agent_reviews", []) if isinstance(data, dict) else []
                if not isinstance(reviews, list):
                    raise ValueError("Batch LLM response did not contain an agent_reviews list")
            except Exception as parse_exc:
                if not config.OPENROUTER_ACCEPT_TEXT_FALLBACK or not str(result.get("text", "")).strip():
                    raise parse_exc
                # Free models sometimes return useful text but not strict JSON.
                # For a POC, treat this as a completed LLM review and preserve deterministic actions,
                # adding the model's text as reasoning rather than failing all agents.
                reviews = self._text_fallback_reviews(signals, str(result.get("text", "")))
            by_name: dict[str, dict[str, Any]] = {}
            by_norm: dict[str, dict[str, Any]] = {}
            placeholders = {"same as input", "input", "agent_name", "same_as_input", ""}
            for item in reviews:
                if not isinstance(item, dict):
                    continue
                raw_name = str(item.get("agent_name") or "").strip()
                if raw_name.lower() in placeholders:
                    continue
                by_name[raw_name] = item
                by_norm[self._normalize_name(raw_name)] = item

            # If names are imperfect but the model returned the same number of review
            # objects, map them positionally. This is safer than discarding every review.
            valid_review_items = [item for item in reviews if isinstance(item, dict)]
            if len(valid_review_items) == len(signals) and len(by_name) < len(signals):
                for signal, item in zip(signals, valid_review_items):
                    by_name.setdefault(signal.agent_name, item)
                    by_norm.setdefault(self._normalize_name(signal.agent_name), item)

            # If a single-agent call returned a placeholder name, map it to that agent.
            if len(signals) == 1 and not by_name and reviews and isinstance(reviews[0], dict):
                by_name[signals[0].agent_name] = reviews[0]
                by_norm[self._normalize_name(signals[0].agent_name)] = reviews[0]

            refined: list[AgentSignal] = []
            for signal in signals:
                review = by_name.get(signal.agent_name) or by_norm.get(self._normalize_name(signal.agent_name))
                if review is None:
                    error = "OpenRouter response did not include a valid review for this specialist; deterministic scaffold retained."
                    refined_signal = self._mark_signal_llm_fallback(signal, error)
                    self.agent_review_stats["failed"] = int(self.agent_review_stats.get("failed", 0)) + 1
                    self._record_agent_review(signal.agent_name, False, result.get("model"), error)
                else:
                    refined_signal = self._merge_agent_review(signal, context, review, result)
                    self.agent_review_stats["successful"] = int(self.agent_review_stats.get("successful", 0)) + 1
                    self._record_agent_review(signal.agent_name, True, result.get("model"), None)
                refined.append(refined_signal)
            return refined
        except Exception as exc:
            self.last_error = f"{group_name}_batch_review: {exc}"
            refined = []
            for signal in signals:
                self.agent_review_stats["failed"] = int(self.agent_review_stats.get("failed", 0)) + 1
                self._record_agent_review(signal.agent_name, False, None, str(exc))
                refined.append(self._mark_signal_llm_fallback(signal, str(exc)))
            return refined

    @staticmethod
    def _agent_group_name(agent_name: str) -> str:
        name = str(agent_name).lower()
        if any(token in name for token in ["thermal", "blasttemperature", "oxygen"]):
            return "thermal"
        if any(token in name for token in ["permeability", "wind", "toppressure", "burden"]):
            return "flow"
        if any(token in name for token in ["pci", "coke", "fuel"]):
            return "fuel"
        if any(token in name for token in ["quality", "tapping"]):
            return "quality"
        return "default"

    @staticmethod
    def _models_for_agent_group(group_name: str) -> list[str]:
        if group_name == "thermal":
            return config.OPENROUTER_AGENT_THERMAL_MODELS
        if group_name == "flow":
            return config.OPENROUTER_AGENT_FLOW_MODELS
        if group_name == "fuel":
            return config.OPENROUTER_AGENT_FUEL_MODELS
        if group_name == "quality":
            return config.OPENROUTER_AGENT_QUALITY_MODELS
        return config.OPENROUTER_AGENT_DEFAULT_MODELS

    def _invoke_openrouter(
        self,
        prompt: str,
        system_prompt: str,
        purpose: str,
        max_tokens: int,
        temperature: float,
        response_schema: dict[str, Any] | None = None,
        total_timeout_seconds: float | None = None,
        model_candidates_override: list[str] | None = None,
    ) -> dict[str, str]:
        if not self.enabled:
            raise RuntimeError("OpenRouter is not enabled or no valid API key is configured.")

        model_candidates = self._sanitize_model_candidates(model_candidates_override) if model_candidates_override else self._model_candidates()
        cache_key = self._cache_key(
            purpose,
            system_prompt,
            prompt,
            str(max_tokens),
            str(temperature),
            json.dumps(model_candidates),
            "structured" if response_schema else "text",
        )
        if config.OPENROUTER_CACHE_RESPONSES and cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            self.last_successful_model = cached.get("model") or self.last_successful_model
            self.last_successful_key = cached.get("key") or self.last_successful_key
            return cached

        self.last_attempted_models = list(model_candidates)
        self.last_attempted_keys = []
        self.last_attempted_routes = []
        self.last_error = None

        key_order = self._key_order()[: max(1, config.OPENROUTER_MAX_KEY_ATTEMPTS)]
        deadline = time.monotonic() + max(float(total_timeout_seconds or config.OPENROUTER_TOTAL_TIMEOUT_SECONDS), 1.0)
        last_exc: Exception | None = None

        for key in key_order:
            redacted_key = self._redact_key(key)
            if redacted_key not in self.last_attempted_keys:
                self.last_attempted_keys.append(redacted_key)
            route_label = f"OpenRouter native route {model_candidates} via {redacted_key}"
            self.last_attempted_routes.append(route_label)
            remaining = deadline - time.monotonic()
            if remaining <= 0.25:
                self.last_error = "OpenRouter total LLM budget exhausted before a native fallback request could complete"
                raise requests.exceptions.Timeout(self.last_error)
            try:
                result = self._call_openrouter(
                    model_candidates=model_candidates,
                    api_key=key,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_schema=response_schema,
                    timeout_seconds=min(config.OPENROUTER_TIMEOUT_SECONDS, max(1.0, remaining)),
                )
                text = result.get("text", "")
                if text:
                    out = {"text": text, "model": result.get("model") or model_candidates[0], "key": redacted_key, "purpose": purpose}
                    self.last_successful_model = out["model"]
                    self.last_successful_key = redacted_key
                    if config.OPENROUTER_CACHE_RESPONSES:
                        self._cache[cache_key] = out
                    return out
                last_exc = RuntimeError(f"{route_label}: empty response")
                self.last_error = str(last_exc)
            except Exception as exc:
                last_exc = exc
                self.last_error = f"{route_label}: {exc}"
                continue

        if last_exc:
            raise last_exc
        raise RuntimeError("OpenRouter call failed without a specific error.")

    @staticmethod
    def _sanitize_model_candidates(models: list[str] | None) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for model in models or []:
            model = str(model).strip()
            lowered = model.lower()
            if not model or not (lowered.endswith(":free") or lowered == "openrouter/free"):
                continue
            if lowered not in seen:
                seen.add(lowered)
                deduped.append(model)
        return deduped or config.OPENROUTER_AGENT_DEFAULT_MODELS

    def _model_candidates(self) -> list[str]:
        models = self._model_order()
        if not config.OPENROUTER_ENABLE_MODEL_FALLBACKS:
            models = models[:1]
        else:
            models = models[: max(1, config.OPENROUTER_MAX_MODEL_ATTEMPTS)]

        candidates: list[str] = []
        free_router = config.OPENROUTER_FREE_ROUTER_MODEL
        if config.OPENROUTER_USE_FREE_ROUTER and config.OPENROUTER_FREE_ROUTER_FIRST:
            candidates.append(free_router)
        candidates.extend(models)
        if config.OPENROUTER_USE_FREE_ROUTER and not config.OPENROUTER_FREE_ROUTER_FIRST:
            candidates.append(free_router)

        deduped: list[str] = []
        seen: set[str] = set()
        for model in candidates:
            model = str(model).strip()
            if not model:
                continue
            lowered = model.lower()
            if not (lowered.endswith(":free") or lowered == "openrouter/free"):
                continue
            if lowered not in seen:
                seen.add(lowered)
                deduped.append(model)
        return deduped or self.models[:1]

    def _model_order(self) -> list[str]:
        models = self.models.copy()
        if not models:
            return []
        selection = config.OPENROUTER_MODEL_SELECTION
        if selection == "random":
            random.shuffle(models)
            return models
        if selection == "fallback":
            return models
        with self._lock:
            start = self._rotation_index % len(models)
            self._rotation_index += 1
        return models[start:] + models[:start]

    def _key_order(self) -> list[str]:
        keys = self.api_keys.copy()
        if not keys:
            return []
        selection = config.OPENROUTER_KEY_SELECTION
        if selection == "random":
            random.shuffle(keys)
            return keys
        if selection == "fallback":
            return keys
        with self._lock:
            start = self._key_rotation_index % len(keys)
            self._key_rotation_index += 1
        return keys[start:] + keys[:start]

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if config.OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = config.OPENROUTER_HTTP_REFERER
        if config.OPENROUTER_APP_TITLE:
            headers["X-OpenRouter-Title"] = config.OPENROUTER_APP_TITLE
        return headers

    def _call_openrouter(
        self,
        model_candidates: list[str],
        api_key: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        response_schema: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, str]:
        url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        simple_free_mode = config.OPENROUTER_RELIABILITY_MODE == "simple_free_router" and model_candidates == [config.OPENROUTER_FREE_ROUTER_MODEL]
        if simple_free_mode:
            # Most reliable free-model request shape: one model, no provider filters, no strict schema, no plugins.
            # OpenRouter docs note that the free router filters by requested capabilities; keeping the
            # request plain maximizes the eligible free-model pool.
            body["model"] = config.OPENROUTER_FREE_ROUTER_MODEL
        elif config.OPENROUTER_NATIVE_MODEL_FALLBACKS and len(model_candidates) > 1:
            body["models"] = model_candidates
        else:
            body["model"] = model_candidates[0]

        if not simple_free_mode:
            provider = self._provider_preferences(response_schema=response_schema)
            if provider:
                body["provider"] = provider

            if response_schema is not None and config.OPENROUTER_USE_STRUCTURED_OUTPUTS:
                body["response_format"] = {"type": "json_schema", "json_schema": response_schema}
                if config.OPENROUTER_USE_RESPONSE_HEALING:
                    body["plugins"] = [{"id": "response-healing"}]

        read_timeout = max(1.0, float(timeout_seconds if timeout_seconds is not None else config.OPENROUTER_TIMEOUT_SECONDS))
        connect_timeout = max(1.0, min(float(config.OPENROUTER_CONNECT_TIMEOUT_SECONDS), read_timeout))
        response = requests.post(url, headers=self._headers(api_key), json=body, timeout=(connect_timeout, read_timeout))
        try:
            data = response.json()
        except Exception:
            data = {"raw_text": response.text[:500]}

        if response.status_code < 200 or response.status_code >= 300:
            error = data.get("error", data) if isinstance(data, dict) else data
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or str(error)
            else:
                message = str(error)
            raise RuntimeError(f"HTTP {response.status_code}: {message}")

        if isinstance(data, dict) and data.get("error"):
            error = data.get("error")
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"OpenRouter error: {message}")

        try:
            first_choice = data["choices"][0]
            if isinstance(first_choice, dict) and first_choice.get("error"):
                err = first_choice.get("error")
                raise RuntimeError(f"Model returned error choice: {err}")
            content = first_choice["message"].get("content", "")
        except Exception as exc:
            raise RuntimeError(f"Unexpected response format: {data}") from exc

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            content = "\n".join(part for part in parts if part)
        actual_model = str(data.get("model") or model_candidates[0]) if isinstance(data, dict) else model_candidates[0]
        return {"text": str(content).strip(), "model": actual_model}

    @staticmethod
    def _provider_preferences(response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        provider: dict[str, Any] = {"allow_fallbacks": True}
        if config.OPENROUTER_PROVIDER_SORT != "none":
            if config.OPENROUTER_NATIVE_MODEL_FALLBACKS:
                provider["sort"] = {"by": config.OPENROUTER_PROVIDER_SORT, "partition": config.OPENROUTER_PROVIDER_SORT_PARTITION}
            else:
                provider["sort"] = config.OPENROUTER_PROVIDER_SORT
        if config.OPENROUTER_PREFERRED_MAX_LATENCY_P90 > 0:
            provider["preferred_max_latency"] = {"p90": config.OPENROUTER_PREFERRED_MAX_LATENCY_P90}
        if config.OPENROUTER_PREFERRED_MIN_THROUGHPUT_P90 > 0:
            provider["preferred_min_throughput"] = {"p90": config.OPENROUTER_PREFERRED_MIN_THROUGHPUT_P90}
        if response_schema is not None and config.OPENROUTER_REQUIRE_PARAMETERS:
            provider["require_parameters"] = True
        return provider

    @staticmethod
    def _build_summary_prompt(payload: dict[str, Any]) -> str:
        state = payload.get("state", {})
        recommendations = payload.get("recommendations", [])[:4]
        signals = payload.get("signals", [])[:12]
        llm_reviews = payload.get("llm_agent_reviews", [])
        compact = {
            "state": state,
            "top_recommendations": [
                {
                    "agent_name": r.get("agent_name"),
                    "decision_area": r.get("decision_area"),
                    "action_summary": r.get("action_summary"),
                    "confidence": r.get("confidence"),
                    "risk_level": r.get("risk_level"),
                    "reasoning": r.get("reasoning"),
                    "approval_required": r.get("approval_required"),
                }
                for r in recommendations
            ],
            "agent_signals": [
                {
                    "agent_name": s.get("agent_name"),
                    "decision_area": s.get("decision_area"),
                    "severity": s.get("severity"),
                    "confidence": s.get("confidence"),
                    "message": s.get("message"),
                    "proposed_actions": s.get("proposed_actions"),
                    "llm_used": (s.get("metadata") or {}).get("llm_used"),
                    "llm_model": (s.get("metadata") or {}).get("llm_model"),
                }
                for s in signals
            ],
            "llm_reviews": llm_reviews,
        }
        return (
            "Create the operator-facing executive summary only. Do not repeat these instructions. "
            "Do not say what you need to do. Do not include raw JSON. "
            "Write exactly three concise paragraphs, each starting with one of these labels: "
            "Current state:, Recommended operator focus:, Safety and validation checks:. "
            "Use only the provided facts and keep numeric values exactly as provided. "
            "Mention specialist-agent LLM review only if llm_used is true for at least one signal.\n\n"
            f"FACTS_JSON:\n{json.dumps(compact, default=str, separators=(",", ":"))[: config.OPENROUTER_PROMPT_CHAR_LIMIT]}"
        )

    @staticmethod
    def _sanitize_summary_text(text: str, fallback: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return fallback + "\n\nOpenRouter returned an empty summary; deterministic agent summary shown instead."

        # Remove common wrappers and accidental markdown fences.
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            if cleaned.lower().startswith("text"):
                cleaned = cleaned[4:].strip()

        lower = cleaned.lower()
        instruction_echo_markers = [
            "we need to produce",
            "we have dataset",
            "must keep numeric",
            "return three short paragraphs",
            "facts_json",
            "payload json",
            "rewrite the deterministic",
            "do not repeat these instructions",
        ]
        if any(marker in lower for marker in instruction_echo_markers):
            return fallback + "\n\nOpenRouter summary echoed prompt instructions; deterministic agent summary shown instead."

        # Keep only the three requested paragraphs if a model adds preamble.
        start_markers = ["Current state:", "Current State:"]
        starts = [cleaned.find(m) for m in start_markers if cleaned.find(m) >= 0]
        if starts:
            cleaned = cleaned[min(starts):].strip()

        return cleaned[:2500]

    @staticmethod
    def _agent_system_prompt(signal: AgentSignal) -> str:
        return (
            f"You are {signal.agent_name}, a specialist process-plant decision agent for {signal.decision_area}. "
            "You receive plant measurements and a deterministic rule-based scaffold. "
            "Return JSON only. You may refine, reduce, remove, or confirm the proposed advisory action, "
            "but you must not invent measurements and must not claim that setpoints will be executed automatically. "
            "Prefer stability and operator approval over production optimization."
        )

    def _build_agent_review_prompt(self, signal: AgentSignal, context: PlantContext) -> str:
        prompt = self._agent_prompt_payload([signal], context)
        prompt["task"] = "Review and refine this one specialist agent signal before the coordinator combines all agents."
        return json.dumps(prompt, default=str)[: config.OPENROUTER_AGENT_PROMPT_CHAR_LIMIT]

    def _build_agent_batch_review_prompt(self, signals: list[AgentSignal], context: PlantContext) -> str:
        if any(self._is_llm_first_mode(signal) for signal in signals):
            return self._build_agent_llm_first_prompt(signals, context)

        row = self._compact_state(context)
        compact_state_keys = [
            "timestamp", "event_label", "operating_mode", "plant_risk_level", "plant_risk_score",
            "thermal_state_index", "permeability_index", "pressure_drop_kpa", "gas_utilization_pct",
            "hot_metal_temp_c", "hot_metal_si_pct", "wind_volume_nm3_min", "pci_rate_kg_thm",
            "coke_rate_kg_thm", "top_pressure_kpa", "oxygen_enrichment_pct", "hot_blast_temp_c",
            "predicted_hot_metal_temp_4h_c", "predicted_si_4h_pct",
        ]
        state = {k: row.get(k) for k in compact_state_keys if k in row}
        scaffolds = []
        for s in signals:
            scaffolds.append({
                "agent_name": s.agent_name,
                "decision_area": s.decision_area,
                "severity": s.severity,
                "confidence": round(float(s.confidence), 3),
                "message": s.message,
                "proposed_actions": s.proposed_actions,
                "evidence": s.evidence[:3],
                "risk_tags": s.risk_tags[:5],
            })

        # Keep this prompt deliberately small. Free OpenRouter models frequently fail
        # when asked for long, strict-schema, all-field JSON. We ask for a minimal
        # object and the local merger fills missing optional fields safely.
        prompt = {
            "task": "Review every selected blast-furnace specialist agent and return valid JSON only.",
            "critical_rules": [
                "Return JSON only. Do not use markdown fences.",
                "The response must be a JSON object with key agent_reviews.",
                "Return exactly one review per expected_agent_names entry.",
                "Each review.agent_name must exactly match one expected_agent_names value.",
                "Do not return placeholders such as 'same as input'.",
                "Use only the provided plant_state and deterministic_scaffolds; do not invent measurements.",
                "Never claim automatic execution. Setpoint changes remain operator recommendations.",
            ],
            "review_item_shape": {
                "agent_name": "exact name from expected_agent_names",
                "severity": "low|medium|high|critical",
                "confidence": "0.05 to 0.98",
                "message": "one short operator-facing recommendation",
                "proposed_actions": "object; keep deterministic actions unless unsafe",
                "reasoning_addendum": "one short reason using plant numbers",
                "evidence_additions": "array of short strings",
                "prerequisites": "array of checks/approvals",
                "risk_tags": "array of short tags",
            },
            "expected_agent_names": [s.agent_name for s in signals],
            "plant_state": state,
            "deterministic_scaffolds": scaffolds,
        }
        return json.dumps(prompt, default=str, separators=(",", ":"))[: config.OPENROUTER_AGENT_PROMPT_CHAR_LIMIT]

    def _build_agent_llm_first_prompt(self, signals: list[AgentSignal], context: PlantContext) -> str:
        """Build a true LLM-first prompt with compact persistent domain memory.

        The model is still stateless on every API call, so this function sends a
        compact shared memory block plus agent-specific playbooks. The memory block
        is deliberately stable across calls to make provider-side prompt caching or
        sticky routing more likely when OpenRouter/provider supports it, while the
        dynamic plant payload stays small.
        """
        expected_names = [s.agent_name for s in signals]
        playbooks = self._agent_playbooks()
        agent_requests = []
        for signal in signals:
            agent_requests.append({
                "agent_name": signal.agent_name,
                "decision_area": signal.decision_area,
                "playbook": playbooks.get(signal.agent_name, playbooks["DefaultAgent"]),
                "allowed_action_keys": self._allowed_action_keys_for_agent(signal.agent_name),
            })

        allowed_actions = {
            **{key: {"min": lo, "max": hi} for key, (lo, hi) in config.ACTION_LIMITS.items()},
            "burden_distribution_change": "short charging-pattern advisory, or omit",
            "tapping_priority": "Normal / Expedite / Delay, or omit",
            "monitoring_action": "non-control workflow instruction, or omit",
        }
        prompt = {
            "task": "Act as the listed LLM-first specialist blast-furnace agents. Reason independently for each agent and return valid JSON only.",
            "common_agent_memory_version": "forgepilot_bf_operator_copilot_v2",
            "common_agent_memory": self._common_agent_memory(),
            "response_rules": [
                "Return JSON only. Do not use markdown fences or explanatory preamble.",
                "The response must be a JSON object with key agent_reviews.",
                "Return exactly one review per expected_agent_names entry.",
                "Each review.agent_name must exactly match one expected_agent_names value.",
                "Use only the supplied plant_state, recent_trends, similar_cases, common_agent_memory, agent_playbooks, and allowed_actions.",
                "Do not invent measurements, targets, causes, or plant events.",
                "Do not say 'no setpoint action proposed' when the agent playbook thresholds clearly indicate a corrective advisory.",
                "All setpoint changes are recommendations requiring operator approval; never claim automatic execution.",
                "Use small bounded setpoint deltas inside allowed_actions; if uncertain, propose a monitoring_action plus the safest small corrective action.",
            ],
            "review_item_shape": {
                "agent_name": "exact name from expected_agent_names",
                "severity": "low|medium|high|critical",
                "confidence": "0.05 to 0.98; lower when evidence conflicts, higher when multiple indicators agree",
                "message": "operator-facing diagnosis and recommendation in one sentence",
                "proposed_actions": "object with allowed action keys only; empty only if playbook says hold/monitor",
                "reasoning_addendum": "one or two sentences using specific plant numbers",
                "evidence_additions": "array of short strings using provided plant numbers",
                "prerequisites": "array of checks/approvals",
                "risk_tags": "array of short tags",
            },
            "expected_agent_names": expected_names,
            "agent_playbooks": agent_requests,
            "allowed_actions": allowed_actions,
            "plant_state": self._compact_state(context),
            "recent_trends": self._compact_trends(context),
            "similar_cases": context.similar_cases[:2],
            "examples": self._llm_first_examples(),
        }
        # Do not slice the JSON prompt aggressively. Truncating JSON creates the exact
        # failure mode seen in the UI: the model loses the operating context and returns
        # generic placeholders. The payload is still compact and batched, so the shared
        # memory is sent once per LLM review rather than once per specialist agent.
        return json.dumps(prompt, default=str, separators=(",", ":"))

    @staticmethod
    def _common_agent_memory() -> dict[str, Any]:
        return {
            "mission": "ForgePilot is an advisory copilot for blast furnace operators. It augments, never replaces, operator judgement.",
            "operating_principles": [
                "Stability and safety outrank productivity.",
                "Do not optimize one lever without considering thermal state, permeability, fuel balance, quality, and hearth/tapping state.",
                "High pressure drop plus low permeability index is a gas-flow restriction/permeability recovery case.",
                "Cold thermal state is indicated by low hot metal temperature, low silicon, negative thermal index, weak gas utilization, or falling predicted temperature.",
                "Hot thermal state is indicated by high hot metal temperature, high silicon, positive thermal index, or excessive RAFT/thermal reserve.",
                "Poor permeability generally argues against aggressive wind increase, aggressive PCI increase, or production push.",
                "Weak thermal reserve generally argues for reducing PCI pressure, increasing coke/thermal support, or increasing hot blast/oxygen carefully.",
                "Use small corrective deltas. All setpoint changes require operator approval.",
            ],
            "reference_thresholds": {
                "hot_metal_temp_c_target": "1480-1510; below 1480 is cold/weak, above 1510 is hot",
                "hot_metal_si_pct_target": "0.38-0.62; below 0.38 can indicate cold/weak thermal state, above 0.62 can indicate hot state",
                "thermal_state_index_target": "-0.65 to +0.65; below -0.65 cold, above +0.65 hot",
                "permeability_index_target": "82-100 typical stable range; below 70 concerning; below 55 severe restriction",
                "pressure_drop_kpa_target": "130-180 typical operating band; above 180 severe restriction, above 160 elevated",
                "gas_utilization_pct_target": "46-51.5; below 44 weak efficiency/gas-flow concern",
                "plant_risk_score": "<30 low, 30-60 medium, 60-80 high, >80 critical",
            },
            "allowed_action_meaning": {
                "wind_volume_delta_nm3_min": "negative reduces wind/gas push; positive increases production/gas flow",
                "pci_delta_kg_thm": "negative reduces coal injection to protect permeability/thermal reserve; positive improves fuel economy only if stable",
                "coke_rate_delta_kg_thm": "positive adds thermal/structural support; negative improves cost only when stable",
                "oxygen_enrichment_delta_pct": "positive raises thermal intensity/productivity; avoid aggressive increase during severe restriction",
                "blast_temp_delta_c": "positive adds thermal input; use for cold/weak furnace within stove capability",
                "top_pressure_delta_kpa": "small increase may improve gas utilization; avoid if pressure-drop/slip risk is severe",
                "burden_distribution_change": "charging-pattern advisory for center/edge gas flow and permeability recovery",
                "tapping_priority": "hearth/cast-house urgency: Normal, Expedite, or Delay",
            },
        }

    @staticmethod
    def _agent_playbooks() -> dict[str, Any]:
        return {
            "ThermalStateAgent": {
                "role": "Assess hot/cold thermal drift and thermal reserve.",
                "watch": ["hot_metal_temp_c", "hot_metal_si_pct", "thermal_state_index", "gas_utilization_pct", "predicted_hot_metal_temp_4h_c", "blast_temp_delta trend", "fuel rate"],
                "decision_logic": [
                    "If thermal_state_index < -0.65 or hot_metal_temp_c < 1480 or silicon < 0.38, diagnose cold/weak thermal state.",
                    "For cold/weak state, consider coke_rate_delta_kg_thm +5 to +15, blast_temp_delta_c +10 to +20, oxygen_enrichment_delta_pct +0.1 to +0.3, and avoid PCI increase.",
                    "If thermal_state_index > 0.65 or hot_metal_temp_c > 1510 or silicon > 0.62, diagnose hot state and consider reducing thermal input.",
                    "If permeability is severely poor, prioritize stabilization before productivity push.",
                ],
                "preferred_actions": ["coke_rate_delta_kg_thm", "blast_temp_delta_c", "oxygen_enrichment_delta_pct", "monitoring_action"],
            },
            "PermeabilityAgent": {
                "role": "Assess gas-flow restriction, burden permeability, hang/slip risk.",
                "watch": ["permeability_index", "pressure_drop_kpa", "gas_utilization_pct", "top_pressure_kpa", "burden_distribution", "slip risk"],
                "decision_logic": [
                    "If pressure_drop_kpa > 180 and permeability_index < 55, diagnose severe permeability loss.",
                    "If pressure_drop_kpa > 160 or permeability_index < 70, diagnose elevated restriction risk.",
                    "For severe restriction, recommend wind_volume_delta_nm3_min -50 to -150, pci_delta_kg_thm -5 to -15, burden_distribution_change toward center-coke/permeability recovery, and possibly Expedite tapping if hearth/liquid level supports it.",
                    "Avoid wind increase, aggressive top pressure increase, and PCI increase during severe restriction.",
                ],
                "preferred_actions": ["wind_volume_delta_nm3_min", "pci_delta_kg_thm", "burden_distribution_change", "top_pressure_delta_kpa", "monitoring_action"],
            },
            "WindVolumeAgent": {
                "role": "Recommend wind-volume hold/increase/reduction.",
                "watch": ["wind_volume_nm3_min", "pressure_drop_kpa", "permeability_index", "thermal_state_index", "production_tph", "gas_utilization_pct"],
                "decision_logic": [
                    "Increase wind only when permeability is stable, pressure drop is not elevated, and thermal reserve is adequate.",
                    "If permeability_index < 70 or pressure_drop_kpa > 160, recommend holding or reducing wind.",
                    "If permeability_index < 55 and pressure_drop_kpa > 180, recommend wind_volume_delta_nm3_min -50 to -150.",
                ],
                "preferred_actions": ["wind_volume_delta_nm3_min", "monitoring_action"],
            },
            "PCIAgent": {
                "role": "Optimize PCI without harming permeability or thermal reserve.",
                "watch": ["pci_rate_kg_thm", "thermal_state_index", "permeability_index", "pressure_drop_kpa", "hot_metal_temp_c", "gas_utilization_pct"],
                "decision_logic": [
                    "If permeability is poor or pressure drop is high, reduce PCI temporarily by -5 to -15 kg/thm.",
                    "If furnace is cold/weak, do not increase PCI; reduce or hold PCI and coordinate with coke/thermal support.",
                    "Increase PCI only when thermal state and permeability are stable.",
                ],
                "preferred_actions": ["pci_delta_kg_thm", "coke_rate_delta_kg_thm", "monitoring_action"],
            },
            "CokeRateAgent": {
                "role": "Maintain coke support for thermal reserve and bed permeability.",
                "watch": ["coke_rate_kg_thm", "pci_rate_kg_thm", "thermal_state_index", "permeability_index", "pressure_drop_kpa"],
                "decision_logic": [
                    "If furnace is cold/weak or permeability is poor, consider coke_rate_delta_kg_thm +5 to +15.",
                    "If operation is stable and fuel rate is high, consider holding or small reduction only if other agents agree.",
                ],
                "preferred_actions": ["coke_rate_delta_kg_thm", "monitoring_action"],
            },
            "FuelRateAgent": {
                "role": "Assess total fuel balance and energy efficiency while protecting stability.",
                "watch": ["total_fuel_rate_kg_thm", "coke_rate_kg_thm", "pci_rate_kg_thm", "gas_utilization_pct", "thermal_state_index"],
                "decision_logic": [
                    "If stability risk is high, fuel efficiency optimization is secondary.",
                    "For cold or permeability-loss cases, support coke/thermal recovery rather than chasing low fuel rate.",
                    "If stable but fuel rate high, recommend monitoring/optimization rather than aggressive cuts.",
                ],
                "preferred_actions": ["coke_rate_delta_kg_thm", "pci_delta_kg_thm", "monitoring_action"],
            },
            "OxygenEnrichmentAgent": {
                "role": "Recommend oxygen enrichment adjustments within raceway/thermal safety limits.",
                "watch": ["oxygen_enrichment_pct", "thermal_state_index", "permeability_index", "pressure_drop_kpa", "gas_utilization_pct", "raceway_adiabatic_flame_temp_c"],
                "decision_logic": [
                    "For cold/weak thermal state with manageable permeability, consider +0.1 to +0.3 pct-pt oxygen enrichment.",
                    "During severe permeability restriction, avoid aggressive enrichment; use at most small +0.1 to +0.2 if thermal support is needed and operator confirms raceway stability.",
                    "For hot state, consider hold or slight reduction.",
                ],
                "preferred_actions": ["oxygen_enrichment_delta_pct", "monitoring_action"],
            },
            "BlastTemperatureAgent": {
                "role": "Recommend hot-blast temperature adjustment for thermal stabilization.",
                "watch": ["hot_blast_temp_c", "thermal_state_index", "hot_metal_temp_c", "hot_metal_si_pct", "predicted_hot_metal_temp_4h_c"],
                "decision_logic": [
                    "If cold/weak thermal state and stove capacity permits, recommend blast_temp_delta_c +10 to +20.",
                    "If hot state, recommend hold or small reduction.",
                    "Do not use blast temperature as the only correction for severe permeability loss.",
                ],
                "preferred_actions": ["blast_temp_delta_c", "monitoring_action"],
            },
            "TopPressureAgent": {
                "role": "Recommend top-pressure strategy for gas utilization without worsening restriction.",
                "watch": ["top_pressure_kpa", "pressure_drop_kpa", "permeability_index", "gas_utilization_pct", "slip risk"],
                "decision_logic": [
                    "If gas utilization is low but permeability is manageable, small top_pressure_delta_kpa +2 to +5 may help.",
                    "If severe pressure drop/restriction exists, do not aggressively raise top pressure; prefer hold or very small adjustment with operator confirmation.",
                ],
                "preferred_actions": ["top_pressure_delta_kpa", "monitoring_action"],
            },
            "BurdenDistributionAgent": {
                "role": "Recommend charging/burden distribution changes for gas-flow balance and permeability recovery.",
                "watch": ["permeability_index", "pressure_drop_kpa", "gas_utilization_pct", "burden distribution mode", "stockline asymmetry"],
                "decision_logic": [
                    "If edge-heavy/poor permeability signals exist, recommend center-coke/permeability-recovery burden pattern.",
                    "If gas utilization and permeability are stable, hold distribution and monitor.",
                    "Charging-pattern changes are advisory and must follow plant-approved burden matrix.",
                ],
                "preferred_actions": ["burden_distribution_change", "monitoring_action"],
            },
            "TappingAgent": {
                "role": "Assess hearth/tapping urgency and cast-house priority.",
                "watch": ["hearth_liquid_level_index", "hot_metal_temp_c", "plant_risk_score", "tapping indicators", "pressure drop"],
                "decision_logic": [
                    "If permeability/pressure-drop risk is high and hearth liquid level is elevated, recommend Expedite tapping.",
                    "If hearth condition is normal, recommend Normal tapping priority and monitoring.",
                    "Do not create tapping urgency without supporting hearth/liquid-level evidence.",
                ],
                "preferred_actions": ["tapping_priority", "monitoring_action"],
            },
            "QualityAgent": {
                "role": "Assess hot-metal quality stability, especially silicon and sulfur drift.",
                "watch": ["hot_metal_si_pct", "hot_metal_s_pct", "hot_metal_temp_c", "thermal_state_index", "predicted_si_4h_pct"],
                "decision_logic": [
                    "Low silicon with cold thermal indicators supports thermal recovery recommendations.",
                    "High silicon with hot thermal indicators supports thermal input reduction/hold.",
                    "If quality is in range, propose monitoring rather than setpoint action.",
                ],
                "preferred_actions": ["monitoring_action", "blast_temp_delta_c", "coke_rate_delta_kg_thm"],
            },
            "DefaultAgent": {
                "role": "Reason over the assigned decision area using only supplied plant data.",
                "watch": ["plant_state", "recent_trends"],
                "decision_logic": ["Recommend monitoring if evidence is weak; otherwise use small bounded corrective actions."],
                "preferred_actions": ["monitoring_action"],
            },
        }

    @staticmethod
    def _allowed_action_keys_for_agent(agent_name: str) -> list[str]:
        playbook = ReasoningSynthesizer._agent_playbooks().get(agent_name, {})
        keys = list(playbook.get("preferred_actions") or ["monitoring_action"])
        return keys

    @staticmethod
    def _llm_first_examples() -> list[dict[str, Any]]:
        return [
            {
                "case": "severe permeability loss",
                "signals": "pressure_drop_kpa=182.8, permeability_index=48.7, gas_utilization_pct=41.3, thermal_state_index=-0.76",
                "good_review": {
                    "agent_name": "PermeabilityAgent",
                    "severity": "critical",
                    "confidence": 0.9,
                    "message": "Permeability loss is severe; prioritize flow recovery and avoid production push.",
                    "proposed_actions": {"wind_volume_delta_nm3_min": -75, "pci_delta_kg_thm": -8, "burden_distribution_change": "Use center-coke permeability-recovery charging matrix"},
                    "reasoning_addendum": "Pressure drop is above 180 kPa while permeability index is below 55, indicating severe gas-flow restriction.",
                    "evidence_additions": ["pressure_drop_kpa=182.8", "permeability_index=48.7", "gas_utilization_pct=41.3"],
                    "prerequisites": ["operator approval required", "confirm pressure/drop and burden probes", "check recent slips/hangs"],
                    "risk_tags": ["permeability", "gas_flow", "operator_approval"],
                },
            },
            {
                "case": "normal low-risk state",
                "signals": "plant_risk_score=16.5, permeability_index=145, pressure_drop_kpa=105",
                "good_review": {
                    "agent_name": "WindVolumeAgent",
                    "severity": "low",
                    "confidence": 0.72,
                    "message": "Wind volume should be held; no production push is required from the available evidence.",
                    "proposed_actions": {"monitoring_action": "Continue normal monitoring of wind, pressure drop, and permeability trend."},
                    "reasoning_addendum": "Low plant risk and low pressure drop do not justify a setpoint change.",
                    "evidence_additions": ["plant_risk_score=16.5", "pressure_drop_kpa=105", "permeability_index=145"],
                    "prerequisites": ["operator approval required for any setpoint change"],
                    "risk_tags": ["normal_monitoring"],
                },
            },
        ]

    def _agent_prompt_payload(self, signals: list[AgentSignal], context: PlantContext) -> dict[str, Any]:
        allowed_actions = {
            **{key: {"min": lo, "max": hi} for key, (lo, hi) in config.ACTION_LIMITS.items()},
            "burden_distribution_change": "short operator-readable charging-pattern advisory, or omit",
            "tapping_priority": "Normal / Expedite / Delay, or omit",
            "monitoring_action": "non-control workflow instruction, or omit",
        }
        return {
            "hard_constraints": [
                "Return valid JSON only. No markdown.",
                "Do not invent measurements. Use only plant_state, recent_trends, similar_cases, and supplied evidence.",
                "If plant risk is low and evidence is weak or conflicting, prefer proposed_actions={} and monitoring.",
                "If restriction, thermal, quality, or hearth risk is credible, keep changes small and inside allowed_actions.",
                "Never state that an action is executed automatically. All setpoint changes are recommendations for operator approval.",
            ],
            "allowed_actions": allowed_actions,
            "plant_state": self._compact_state(context),
            "recent_trends": self._compact_trends(context),
            "similar_cases": context.similar_cases[:2],
            "deterministic_scaffolds": [signal.to_dict() for signal in signals],
            "required_review_fields": [
                "agent_name",
                "severity",
                "confidence",
                "message",
                "proposed_actions",
                "reasoning_addendum",
                "evidence_additions",
                "prerequisites",
                "risk_tags",
            ],
        }

    @staticmethod
    @staticmethod
    def _text_fallback_reviews(signals: list[AgentSignal], text: str) -> list[dict[str, Any]]:
        # Free models sometimes return commentary or malformed JSON. Do not surface
        # raw model output in operator cards. In LLM-first mode, avoid repeating the
        # neutral seed text such as "No setpoint action proposed"; show a concise
        # review-status message instead.
        clean_text = ReasoningSynthesizer._strip_markdown_code_fences(str(text or "")).strip()
        clean_text = " ".join(clean_text.split())[:280]
        reviews: list[dict[str, Any]] = []
        for signal in signals:
            is_llm_first = ReasoningSynthesizer._is_llm_first_mode(signal)
            if is_llm_first:
                message = f"{signal.decision_area}: OpenRouter returned an unstructured LLM-first review; no validated setpoint change was accepted."
                actions = {"monitoring_action": "Review OpenRouter diagnostics and continue operator-supervised monitoring; no LLM setpoint action accepted."}
                reasoning = clean_text if clean_text and not clean_text.startswith("{") else "The model response was not valid specialist JSON, so the safety validator blocked action adoption."
                confidence = 0.25
            else:
                message = signal.message
                actions = signal.proposed_actions
                reasoning = ""
                confidence = max(0.0, min(1.0, float(signal.confidence)))
            reviews.append({
                "agent_name": signal.agent_name,
                "severity": signal.severity,
                "confidence": confidence,
                "message": message,
                "proposed_actions": actions,
                "reasoning_addendum": reasoning,
                "evidence_additions": ["OpenRouter returned an unstructured review; deterministic safety validation retained."],
                "prerequisites": signal.prerequisites or ["Operator approval required for setpoint changes"],
                "risk_tags": signal.risk_tags,
                "raw_model_output_redacted": True,
            })
        return reviews

    @staticmethod
    def _single_agent_response_schema() -> dict[str, Any]:
        schema = ReasoningSynthesizer._agent_review_item_schema()
        return {"name": "AgentReview", "strict": True, "schema": schema}

    @staticmethod
    def _batch_agent_response_schema() -> dict[str, Any]:
        return {
            "name": "BatchAgentReview",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "agent_reviews": {
                        "type": "array",
                        "items": ReasoningSynthesizer._agent_review_item_schema(),
                    }
                },
                "required": ["agent_reviews"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def _agent_review_item_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string"},
                "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "message": {"type": "string"},
                "proposed_actions": {"type": "object"},
                "reasoning_addendum": {"type": "string"},
                "evidence_additions": {"type": "array", "items": {"type": "string"}},
                "prerequisites": {"type": "array", "items": {"type": "string"}},
                "risk_tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "agent_name",
                "severity",
                "confidence",
                "message",
                "proposed_actions",
                "reasoning_addendum",
                "evidence_additions",
                "prerequisites",
                "risk_tags",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _compact_state(context: PlantContext) -> dict[str, Any]:
        row = context.current
        keys = [
            "timestamp", "furnace_id", "shift", "operating_mode", "event_label", "sensor_quality_flag",
            "wind_volume_nm3_min", "hot_blast_temp_c", "oxygen_enrichment_pct", "top_pressure_kpa",
            "pci_rate_kg_thm", "coke_rate_kg_thm", "total_fuel_rate_kg_thm", "gas_utilization_pct",
            "pressure_drop_kpa", "permeability_index", "raceway_adiabatic_flame_temp_c", "hearth_liquid_level_index",
            "hot_metal_temp_c", "hot_metal_si_pct", "hot_metal_s_pct", "production_tph", "thermal_state_index",
            "thermal_risk_score", "permeability_risk_score", "quality_risk_score", "plant_risk_score", "plant_risk_level",
            "health_score", "operating_summary", "predicted_hot_metal_temp_4h_c", "predicted_si_4h_pct",
        ]
        return {key: row.get(key) for key in keys if key in row}

    @staticmethod
    def _compact_trends(context: PlantContext) -> dict[str, float]:
        trends: dict[str, float] = {}
        if context.history is None or context.history.empty:
            return trends
        columns = [
            "hot_metal_temp_c", "hot_metal_si_pct", "thermal_state_index", "permeability_index",
            "pressure_drop_kpa", "gas_utilization_pct", "wind_volume_nm3_min", "pci_rate_kg_thm",
            "coke_rate_kg_thm", "top_pressure_kpa", "production_tph", "plant_risk_score",
        ]
        for col in columns:
            if col not in context.history.columns:
                continue
            series = context.history[col].dropna().tail(6)
            if len(series) >= 2:
                trends[f"6h_delta_{col}"] = round(safe_float(series.iloc[-1]) - safe_float(series.iloc[0]), 3)
        return trends

    def _merge_agent_review(self, signal: AgentSignal, context: PlantContext, data: dict[str, Any], result: dict[str, str]) -> AgentSignal:
        original_metadata = dict(signal.metadata or {})
        review_message = str(data.get("message") or "").strip()
        review_reasoning = str(
            data.get("reasoning_addendum")
            or data.get("reasoning")
            or data.get("rationale")
            or data.get("explanation")
            or ""
        ).strip()
        review_actions = data.get("proposed_actions") if isinstance(data.get("proposed_actions"), dict) else {}
        original_metadata.update(
            {
                "llm_used": True,
                "llm_model": result.get("model"),
                "llm_key": result.get("key"),
                "decision_basis": "llm_first_specialist_reasoning_plus_deterministic_safety_validation" if self._metadata_is_llm_first_mode(original_metadata) else "deterministic_rules_plus_openrouter_specialist_review",
                "llm_review_message": review_message,
                "llm_review_confidence": data.get("confidence"),
                "llm_review_severity": data.get("severity"),
                "llm_review_proposed_actions": review_actions,
                "llm_reasoning_addendum": review_reasoning,
            }
        )
        if self._metadata_is_llm_first_mode(original_metadata):
            original_metadata["deterministic_scaffold_text"] = "Not used in LLM-first mode. The LLM specialist reasoned from plant state, shared operating memory, agent playbook, trends, similar cases, and allowed action bounds; deterministic logic only validates/falls back."

        original_severity = str(signal.severity or "low").lower()
        requested_severity = str(data.get("severity") or original_severity).lower()
        severity = self._safe_severity(original_severity, requested_severity, context)

        confidence = signal.confidence
        try:
            if data.get("confidence") is not None:
                confidence = clamp(float(data.get("confidence")), 0.05, 0.98)
        except Exception:
            confidence = signal.confidence

        message = review_message or signal.message
        addendum = review_reasoning
        if addendum and addendum.lower() not in message.lower():
            message = f"{message} LLM review: {addendum}"

        actions = signal.proposed_actions
        if isinstance(data.get("proposed_actions"), dict):
            actions = self._sanitize_actions(data.get("proposed_actions") or {})
            plant_score = safe_float(context.current.get("plant_risk_score"))
            if not actions and signal.proposed_actions and plant_score >= 70 and original_severity in {"critical", "high"}:
                actions = signal.proposed_actions
                original_metadata["llm_action_clearance_blocked"] = "Plant risk is high/critical; deterministic safety actions retained."

        evidence = list(signal.evidence)
        for item in data.get("evidence_additions") or []:
            item = str(item).strip()
            if item and item not in evidence:
                evidence.append(f"LLM review: {item}")

        prerequisites = list(signal.prerequisites)
        for item in data.get("prerequisites") or []:
            item = str(item).strip()
            if item and item not in prerequisites:
                prerequisites.append(item)

        risk_tags = list(signal.risk_tags)
        for item in data.get("risk_tags") or []:
            tag = str(item).strip().lower().replace(" ", "_")
            if tag and tag not in risk_tags:
                risk_tags.append(tag)

        return replace(
            signal,
            severity=severity,
            confidence=round(float(confidence), 3),
            message=message,
            evidence=evidence[:12],
            proposed_actions=actions,
            prerequisites=prerequisites[:10],
            risk_tags=risk_tags[:10],
            metadata=original_metadata,
        )

    @staticmethod
    def _mark_signal_llm_fallback(signal: AgentSignal, error: str) -> AgentSignal:
        metadata = dict(signal.metadata or {})
        metadata.update({"llm_used": False, "llm_error": error, "decision_basis": "deterministic_rules_fallback"})
        return replace(signal, metadata=metadata)

    @staticmethod
    def _safe_severity(original: str, requested: str, context: PlantContext) -> str:
        order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        reverse = {v: k for k, v in order.items()}
        original_score = order.get(original, 1)
        requested_score = order.get(requested, original_score)
        plant_score = safe_float(context.current.get("plant_risk_score"))
        if plant_score >= 70 and original_score >= 3:
            requested_score = max(requested_score, 3)
        normal_low_risk = str(context.current.get("event_label", "")).lower() == "normal" and plant_score < 30
        if not normal_low_risk and original_score - requested_score > 1:
            requested_score = original_score - 1
        return reverse.get(max(1, min(4, requested_score)), original)

    @staticmethod
    def _sanitize_actions(actions: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in actions.items():
            if key in config.ACTION_LIMITS:
                numeric = safe_float(value)
                if abs(numeric) < 1e-9:
                    continue
                low, high = config.ACTION_LIMITS[key]
                cleaned[key] = round(clamp(numeric, low, high), 2)
            elif key in {"burden_distribution_change", "tapping_priority", "monitoring_action"}:
                text = str(value).strip()
                if text:
                    cleaned[key] = text[:240]
        return cleaned

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        stripped = ReasoningSynthesizer._strip_markdown_code_fences(raw)
        candidates: list[str] = []

        for value in [stripped, raw]:
            value = value.strip()
            if not value:
                continue
            candidates.append(value)
            lower = value.lower()
            if lower.startswith("json"):
                candidates.append(value[4:].strip())
            try:
                candidates.append(ReasoningSynthesizer._extract_first_json_object(value))
            except Exception:
                pass
            try:
                candidates.append(ReasoningSynthesizer._extract_first_json_array(value))
            except Exception:
                pass

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_candidates: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)

        last_error: Exception | None = None
        for candidate in unique_candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"agent_reviews": parsed}
                raise ValueError("LLM response was not a JSON object or array")
            except Exception as exc:
                last_error = exc
                continue
        raise ValueError(f"Could not parse LLM response as JSON: {last_error}")

    @staticmethod
    def _strip_markdown_code_fences(text: str) -> str:
        value = str(text or "").strip()
        if "```" not in value:
            return value
        # Prefer the content inside the first fenced block, wherever it occurs.
        parts = value.split("```")
        if len(parts) >= 3:
            block = parts[1].strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            return block
        return value.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    @staticmethod
    def _extract_first_json_array(text: str) -> str:
        # Prefer an array following the agent_reviews key if present.
        marker = '"agent_reviews"'
        start_search = 0
        marker_index = text.find(marker)
        if marker_index >= 0:
            bracket = text.find("[", marker_index)
            if bracket >= 0:
                start_search = bracket
        start = text.find("[", start_search)
        if start < 0:
            raise ValueError("No JSON array found in LLM response")
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        raise ValueError("Unterminated JSON array in LLM response")

    @staticmethod
    def _normalize_name(value: str) -> str:
        return "".join(ch for ch in str(value or "").lower() if ch.isalnum())

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object found in LLM response")
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        raise ValueError("Unterminated JSON object in LLM response")

    @staticmethod
    def _cache_key(*parts: str) -> str:
        raw = "\n---\n".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _redact_key(key: str | None) -> str | None:
        if not key:
            return None
        if len(key) <= 10:
            return "***"
        return f"{key[:6]}...{key[-4:]}"

    def _record_agent_review(self, agent_name: str, success: bool, model: str | None, error: str | None) -> None:
        rows = list(self.agent_review_stats.get("last_reviews", []))
        rows.append({"agent_name": agent_name, "success": success, "model": model, "error": error})
        self.agent_review_stats["last_reviews"] = rows[-12:]

    @staticmethod
    def _fallback_summary(payload: dict[str, Any]) -> str:
        state = payload.get("state", {})
        recommendations = payload.get("recommendations", [])
        top = recommendations[0] if recommendations else {}
        main = f"Top advisory: {top.get('action_summary', 'No action summary')}" if top else "No active setpoint advisory; continue monitoring."
        return (
            f"Current state: plant risk is {state.get('plant_risk_level', 'unknown')} with score {state.get('plant_risk_score', 'unknown')}; "
            f"detected scenario is {state.get('event_label', 'unknown')}. Thermal index={state.get('thermal_state_index')}, "
            f"permeability index={state.get('permeability_index')}, pressure drop={state.get('pressure_drop_kpa')} kPa.\n\n"
            f"Recommended operator focus: {main}\n\n"
            "Safety/validation checks: confirm sensor and lab quality, check recent operator actions, and apply only plant-approved setpoint changes."
        )
