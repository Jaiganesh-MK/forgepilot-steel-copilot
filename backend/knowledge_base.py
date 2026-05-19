from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class KnowledgeSnippet:
    source: str
    title: str
    text: str
    score: float
    tags: tuple[str, ...] = ()

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "score": round(self.score, 3),
            "text": self.text[: config.RAG_MAX_SNIPPET_CHARS],
            "tags": list(self.tags),
        }


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(str(text or "")) if len(t) > 1]


def _chunk_markdown(path: Path, text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_title = path.stem.replace("_", " ").title()
    current_lines: list[str] = []
    current_tags: list[str] = []

    def flush() -> None:
        body = "\n".join(line.strip() for line in current_lines if line.strip()).strip()
        if not body:
            return
        # Soft split very large sections while preserving source/title.
        words = body.split()
        max_words = max(80, config.RAG_CHUNK_MAX_WORDS)
        overlap = max(0, min(config.RAG_CHUNK_OVERLAP_WORDS, max_words // 3))
        start = 0
        while start < len(words):
            part_words = words[start:start + max_words]
            part = " ".join(part_words).strip()
            if part:
                chunks.append({
                    "source": path.name,
                    "title": current_title,
                    "text": part,
                    "tags": tuple(current_tags),
                    "tokens": _tokenize(current_title + " " + part + " " + " ".join(current_tags)),
                })
            if start + max_words >= len(words):
                break
            start += max_words - overlap

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            current_title = heading.group(2).strip()
            current_lines = []
            current_tags = _tokenize(current_title)
        else:
            current_lines.append(line)
    flush()
    return chunks


@lru_cache(maxsize=1)
def _load_index() -> tuple[list[dict[str, Any]], dict[str, float]]:
    kb_dir = Path(config.KNOWLEDGE_BASE_DIR)
    docs: list[dict[str, Any]] = []
    if not kb_dir.exists():
        return docs, {}
    for path in sorted(kb_dir.glob("**/*.md")):
        try:
            docs.extend(_chunk_markdown(path, path.read_text(encoding="utf-8")))
        except Exception:
            continue

    # Lightweight IDF, no external vector DB or embeddings required for Streamlit Cloud.
    df: dict[str, int] = {}
    for doc in docs:
        for token in set(doc.get("tokens", [])):
            df[token] = df.get(token, 0) + 1
    total = max(len(docs), 1)
    idf = {token: math.log((1 + total) / (1 + freq)) + 1.0 for token, freq in df.items()}
    return docs, idf


def reset_cache() -> None:
    _load_index.cache_clear()


def status() -> dict[str, Any]:
    docs, _idf = _load_index()
    sources = sorted({str(d.get("source")) for d in docs})
    return {
        "rag_enabled": config.RAG_ENABLED,
        "rag_dir": str(config.KNOWLEDGE_BASE_DIR),
        "rag_chunk_count": len(docs),
        "rag_sources": sources,
        "rag_top_k": config.RAG_TOP_K,
    }


def build_agent_query(agent_name: str, decision_area: str, plant_state: dict[str, Any], trends: dict[str, Any] | None = None) -> str:
    terms = [agent_name, decision_area, "blast furnace operator advisory"]
    event = str(plant_state.get("event_label") or "")
    mode = str(plant_state.get("operating_mode") or "")
    risk = str(plant_state.get("plant_risk_level") or "")
    terms.extend([event, mode, risk])
    # Add state-dependent terms so the retriever pulls the right playbook.
    try:
        pressure_drop = float(plant_state.get("pressure_drop_kpa") or 0)
        permeability = float(plant_state.get("permeability_index") or 0)
        thermal = float(plant_state.get("thermal_state_index") or 0)
        hm_temp = float(plant_state.get("hot_metal_temp_c") or 0)
        si = float(plant_state.get("hot_metal_si_pct") or 0)
        gas = float(plant_state.get("gas_utilization_pct") or 0)
        hearth = float(plant_state.get("hearth_liquid_level_index") or 0)
    except Exception:
        pressure_drop = permeability = thermal = hm_temp = si = gas = hearth = 0
    if pressure_drop > 170 or 0 < permeability < 70:
        terms.extend(["permeability loss pressure drop gas flow restriction hanging slip wind reduction PCI reduction burden distribution"])
    if thermal < -0.55 or (hm_temp and hm_temp < 1485) or (si and si < 0.4):
        terms.extend(["cold thermal state hot metal temperature silicon coke support blast temperature oxygen enrichment PCI hold"])
    if thermal > 0.55 or hm_temp > 1510 or si > 0.62:
        terms.extend(["hot thermal state reduce heat input silicon high"])
    if gas and gas < 44:
        terms.extend(["low gas utilization top pressure efficiency gas distribution"])
    if hearth > 75:
        terms.extend(["hearth liquid level tapping priority cast house"])
    if trends:
        trend_text = " ".join(k for k, v in trends.items() if isinstance(v, (int, float)) and abs(v) > 0)
        terms.append(trend_text)
    return " ".join(str(t) for t in terms if str(t).strip())


def retrieve(query: str, agent_name: str | None = None, top_k: int | None = None) -> list[KnowledgeSnippet]:
    if not config.RAG_ENABLED:
        return []
    docs, idf = _load_index()
    if not docs:
        return []
    top_k = top_k or config.RAG_TOP_K
    query_tokens = _tokenize(query)
    if agent_name:
        query_tokens.extend(_tokenize(agent_name))
    if not query_tokens:
        return []
    q_tf: dict[str, int] = {}
    for token in query_tokens:
        q_tf[token] = q_tf.get(token, 0) + 1
    scored: list[KnowledgeSnippet] = []
    q_set = set(q_tf)
    for doc in docs:
        tokens = doc.get("tokens", [])
        if not tokens:
            continue
        d_tf: dict[str, int] = {}
        for token in tokens:
            d_tf[token] = d_tf.get(token, 0) + 1
        overlap = q_set.intersection(d_tf)
        if not overlap:
            continue
        score = 0.0
        for token in overlap:
            score += (1.0 + math.log(q_tf[token])) * (1.0 + math.log(d_tf[token])) * idf.get(token, 1.0)
        # Slightly boost chunks whose title/tags mention the agent's decision area.
        title_tags = " ".join([str(doc.get("title", "")), " ".join(doc.get("tags", ()))])
        if agent_name and any(tok in _tokenize(title_tags) for tok in _tokenize(agent_name)):
            score *= 1.2
        if score > 0:
            scored.append(KnowledgeSnippet(
                source=str(doc.get("source", "knowledge_base")),
                title=str(doc.get("title", "Untitled")),
                text=str(doc.get("text", "")),
                score=score,
                tags=tuple(doc.get("tags", ())),
            ))
    scored.sort(key=lambda item: item.score, reverse=True)
    # Keep source diversity when possible.
    selected: list[KnowledgeSnippet] = []
    seen_text: set[str] = set()
    for item in scored:
        key = item.text[:120]
        if key in seen_text:
            continue
        seen_text.add(key)
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected


def retrieve_for_agents(agent_requests: list[dict[str, Any]], plant_state: dict[str, Any], trends: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for request in agent_requests:
        agent_name = str(request.get("agent_name") or "")
        decision_area = str(request.get("decision_area") or "")
        playbook = request.get("playbook") if isinstance(request.get("playbook"), dict) else {}
        role = str(playbook.get("role") or "")
        watch = " ".join(str(x) for x in playbook.get("watch", []) if x)
        query = build_agent_query(agent_name, decision_area + " " + role + " " + watch, plant_state, trends)
        snippets = retrieve(query, agent_name=agent_name, top_k=config.RAG_TOP_K)
        result[agent_name] = [snippet.to_prompt_dict() for snippet in snippets]
    return result
