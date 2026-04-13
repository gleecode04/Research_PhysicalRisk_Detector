"""
Example message shapes for wristband → Kafka topics.

Assume producers publish JSON (UTF-8) so consumers can decode without Avro schema registry.
Adjust keys/fields to match your gateway / Empatica E4 integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MinuteBiomarkerEvent:
    """One row-equivalent after cloud aggregation (per-minute digital biomarkers)."""

    timestamp_unix_ms: int
    eda_scl_usiemens: float | None = None
    temperature_celsius: float | None = None
    pulse_rate_bpm: float | None = None
    participant_id: str | None = None
    missing_value_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "timestamp_unix": self.timestamp_unix_ms,
            "eda_scl_usiemens": self.eda_scl_usiemens,
            "temperature_celsius": self.temperature_celsius,
            "pulse_rate_bpm": self.pulse_rate_bpm,
        }
        if self.participant_id is not None:
            d["participant_id"] = self.participant_id
        if self.missing_value_reason is not None:
            d["missing_value_reason"] = self.missing_value_reason
        return d


@dataclass
class RawWristSampleEvent:
    """Higher-rate alternative: single fused sample (already mapped to model columns)."""

    timestamp_unix_ms: int
    eda_eda: float
    bvp_bvp: float
    temperature_temp: float
    meta: dict[str, Any] = field(default_factory=dict)
