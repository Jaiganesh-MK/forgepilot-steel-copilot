from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .agent_core import PlantContext
from .safety_gate import plant_risk_level, plant_risk_score


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {k: json_safe(v) for k, v in row.to_dict().items()}


class DataStore:
    def __init__(self) -> None:
        self.df = self._load_dataset(config.DATASET_FILE)
        self.playbook_df = self._load_optional_csv(config.PLAYBOOK_FILE)
        self.dictionary_df = self._load_optional_csv(config.DATA_DICTIONARY_FILE)
        self.metadata = self._load_metadata(config.METADATA_FILE)
        self.current_index = len(self.df) - 1
        self._ensure_decision_log()

    @staticmethod
    def _load_dataset(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.reset_index(drop=True)

    @staticmethod
    def _load_optional_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_decision_log(self) -> None:
        if config.DECISION_LOG_PATH.exists():
            return
        with open(config.DECISION_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["logged_at_utc", "dataset_index", "recommendation_id", "operator_id", "decision", "modified_action", "notes"])

    def normalize_index(self, index: int | None = None) -> int:
        if index is None:
            return self.current_index
        return int(max(0, min(len(self.df) - 1, index)))

    def set_current_index(self, index: int) -> int:
        self.current_index = self.normalize_index(index)
        return self.current_index

    def step_index(self, index: int, direction: str = "next") -> int:
        return self.normalize_index(index - 1 if direction == "previous" else index + 1)

    def get_state(self, index: int | None = None) -> dict[str, Any]:
        idx = self.normalize_index(index)
        row = row_to_dict(self.df.iloc[idx])
        row["dataset_index"] = idx
        row["plant_risk_score"] = plant_risk_score(row)
        row["plant_risk_level"] = plant_risk_level(row)
        row["health_score"] = self._health_score(row)
        row["operating_summary"] = self._operating_summary(row)
        return row

    def get_history(self, index: int | None = None, window: int = 36) -> list[dict[str, Any]]:
        idx = self.normalize_index(index)
        start = max(0, idx - int(window) + 1)
        records: list[dict[str, Any]] = []
        for absolute_index, (_, row) in enumerate(self.df.iloc[start:idx + 1].iterrows(), start=start):
            record = row_to_dict(row)
            record["dataset_index"] = absolute_index
            records.append(record)
        return records

    def get_history_df(self, index: int | None = None, window: int = 36) -> pd.DataFrame:
        idx = self.normalize_index(index)
        start = max(0, idx - int(window) + 1)
        return self.df.iloc[start:idx + 1].copy()

    def get_playbook_records(self) -> list[dict[str, Any]]:
        return [dict((k, json_safe(v)) for k, v in record.items()) for record in self.playbook_df.to_dict(orient="records")]

    def get_data_dictionary_records(self) -> list[dict[str, Any]]:
        return [dict((k, json_safe(v)) for k, v in record.items()) for record in self.dictionary_df.to_dict(orient="records")]

    def get_context(self, index: int | None = None) -> PlantContext:
        idx = self.normalize_index(index)
        return PlantContext(
            index=idx,
            current=self.get_state(idx),
            history=self.get_history_df(idx, window=48),
            playbook=self.get_playbook_records(),
            similar_cases=self.get_similar_cases(idx, limit=5),
        )

    def get_similar_cases(self, index: int | None = None, limit: int = 5) -> list[dict[str, Any]]:
        idx = self.normalize_index(index)
        features = ["thermal_risk_score", "permeability_risk_score", "quality_risk_score", "thermal_state_index", "permeability_index", "pressure_drop_kpa", "hot_metal_si_pct", "hot_metal_temp_c"]
        available = [c for c in features if c in self.df.columns]
        if not available:
            return []
        matrix = self.df[available].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.fillna(matrix.median(numeric_only=True))
        std = matrix.std(numeric_only=True).replace(0, 1.0)
        normalized = (matrix - matrix.mean(numeric_only=True)) / std
        current_vector = normalized.iloc[idx]
        distances = ((normalized - current_vector) ** 2).sum(axis=1) ** 0.5
        distances.iloc[idx] = np.inf
        if "event_label" in self.df.columns:
            same_event = self.df["event_label"] == self.df.iloc[idx].get("event_label")
            distances = distances - same_event.astype(float) * 0.35
            distances.iloc[idx] = np.inf
        selected = distances.nsmallest(limit).index.tolist()
        cases: list[dict[str, Any]] = []
        for case_index in selected:
            row = row_to_dict(self.df.iloc[case_index])
            cases.append({
                "dataset_index": int(case_index),
                "timestamp": row.get("timestamp"),
                "event_label": row.get("event_label"),
                "thermal_state_index": row.get("thermal_state_index"),
                "permeability_index": row.get("permeability_index"),
                "pressure_drop_kpa": row.get("pressure_drop_kpa"),
                "hot_metal_si_pct": row.get("hot_metal_si_pct"),
                "operator_action_taken": row.get("operator_action_taken"),
                "operator_notes": row.get("operator_notes"),
                "similarity_distance": round(float(distances.loc[case_index]), 3),
            })
        return cases

    def append_feedback(self, dataset_index: int, recommendation_id: str, operator_id: str, decision: str, modified_action: str | None, notes: str | None) -> dict[str, Any]:
        record = {
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_index": self.normalize_index(dataset_index),
            "recommendation_id": recommendation_id,
            "operator_id": operator_id,
            "decision": decision,
            "modified_action": modified_action or "",
            "notes": notes or "",
        }
        with open(config.DECISION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writerow(record)
        return record

    def get_decision_log(self) -> list[dict[str, Any]]:
        if not config.DECISION_LOG_PATH.exists():
            return []
        df = pd.read_csv(config.DECISION_LOG_PATH)
        return [dict((k, json_safe(v)) for k, v in record.items()) for record in df.tail(100).to_dict(orient="records")]

    def metadata_payload(self) -> dict[str, Any]:
        first_ts = self.df["timestamp"].iloc[0].isoformat() if "timestamp" in self.df.columns else None
        last_ts = self.df["timestamp"].iloc[-1].isoformat() if "timestamp" in self.df.columns else None
        return {"row_count": int(len(self.df)), "current_index": int(self.current_index), "first_timestamp": first_ts, "last_timestamp": last_ts, "columns": list(self.df.columns), "metadata": self.metadata, "data_dictionary_rows": len(self.dictionary_df), "playbook_rows": len(self.playbook_df)}

    @staticmethod
    def _health_score(row: dict[str, Any]) -> int:
        sensor_penalty = 8 if str(row.get("sensor_quality_flag", "Good")).lower() != "good" else 0
        return int(round(max(0, min(100, 100 - plant_risk_score(row) - sensor_penalty))))

    @staticmethod
    def _operating_summary(row: dict[str, Any]) -> str:
        return f"Risk={row.get('plant_risk_level')}; scenario={row.get('event_label')}; operator decision required={row.get('operator_decision_required')}; base category={row.get('recommended_action_category')}."
