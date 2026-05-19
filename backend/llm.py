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
            return result["text"]
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
            by_name = {str(item.get("agent_name")): item for item in reviews if isinstance(item, dict) and item.get("agent_name")}
            refined: list[AgentSignal] = []
            for signal in signals:
                review = by_name.get(signal.agent_name)
                if review is None:
                    error = "Batch response missing this agent; deterministic scaffold retained."
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
        return (
            "Rewrite the deterministic/hybrid multi-agent blast furnace findings into a concise shift-console summary. "
            "Return three short paragraphs with these labels: Current state, Recommended operator focus, "
            "Safety and validation checks. Keep numeric values exactly as provided. Mention when specialist-agent LLM reviews were used.\n\n"
            f"Payload JSON:\n{json.dumps(payload, default=str)[: config.OPENROUTER_PROMPT_CHAR_LIMIT]}"
        )

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
        # Keep this deliberately small for OpenRouter free models. Full deterministic context stays local;
        # the LLM only sanity-checks the selected low-confidence agents.
        row = self._compact_state(context)
        compact_state_keys = [
            "timestamp", "event_label", "operating_mode", "plant_risk_level", "plant_risk_score",
            "thermal_state_index", "permeability_index", "pressure_drop_kpa", "gas_utilization_pct",
            "hot_metal_temp_c", "hot_metal_si_pct", "wind_volume_nm3_min", "pci_rate_kg_thm",
            "coke_rate_kg_thm", "top_pressure_kpa", "oxygen_enrichment_pct", "hot_blast_temp_c",
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
        prompt = {
            "task": "Review each selected specialist agent. Return compact JSON only with key agent_reviews. If uncertain, keep deterministic actions unchanged and lower confidence.",
            "output_shape": {
                "agent_reviews": [{
                    "agent_name": "same as input",
                    "severity": "low|medium|high|critical",
                    "confidence": 0.0,
                    "message": "short operator-facing message",
                    "proposed_actions": {},
                    "reasoning_addendum": "one sentence",
                    "evidence_additions": [],
                    "prerequisites": ["operator approval required for setpoint changes"],
                    "risk_tags": []
                }]
            },
            "rules": [
                "Use only provided plant_state and scaffolds",
                "Do not claim automatic execution",
                "Do not invent measurements",
                "Keep proposed_actions within the scaffold unless there is a clear safety reason to remove/soften them",
            ],
            "plant_state": state,
            "deterministic_scaffolds": scaffolds,
        }
        return json.dumps(prompt, default=str, separators=(",", ":"))[: config.OPENROUTER_AGENT_PROMPT_CHAR_LIMIT]

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
                "Do not invent measurements. Use only plant_state, recent_trends, similar_cases, and scaffold evidence.",
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
    def _text_fallback_reviews(signals: list[AgentSignal], text: str) -> list[dict[str, Any]]:
        short_text = " ".join(str(text).strip().split())[:800]
        reviews: list[dict[str, Any]] = []
        for signal in signals:
            reviews.append({
                "agent_name": signal.agent_name,
                "severity": signal.severity,
                "confidence": max(0.0, min(1.0, float(signal.confidence))),
                "message": signal.message,
                "proposed_actions": signal.proposed_actions,
                "reasoning_addendum": f"OpenRouter returned a non-JSON review; deterministic scaffold retained. Model note: {short_text}",
                "evidence_additions": [],
                "prerequisites": signal.prerequisites or ["Operator approval required for setpoint changes"],
                "risk_tags": signal.risk_tags,
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
        original_metadata.update(
            {
                "llm_used": True,
                "llm_model": result.get("model"),
                "llm_key": result.get("key"),
                "decision_basis": "deterministic_rules_plus_openrouter_specialist_review",
                "llm_reasoning_addendum": str(data.get("reasoning_addendum") or "").strip(),
            }
        )

        original_severity = str(signal.severity or "low").lower()
        requested_severity = str(data.get("severity") or original_severity).lower()
        severity = self._safe_severity(original_severity, requested_severity, context)

        confidence = signal.confidence
        try:
            if data.get("confidence") is not None:
                confidence = clamp(float(data.get("confidence")), 0.05, 0.98)
        except Exception:
            confidence = signal.confidence

        message = str(data.get("message") or signal.message).strip() or signal.message
        addendum = str(data.get("reasoning_addendum") or "").strip()
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
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        try:
            parsed = json.loads(stripped)
        except Exception:
            candidate = ReasoningSynthesizer._extract_first_json_object(stripped)
            parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was not a JSON object")
        return parsed

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
