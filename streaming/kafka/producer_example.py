#!/usr/bin/env python3
"""
Example Kafka **producer** (simulated wristband / gateway).

Install: ``pip install kafka-python`` (or use confluent-kafka-python in production).

Environment:
  KAFKA_BOOTSTRAP_SERVERS  default localhost:9092
  KAFKA_TOPIC_MINUTE       default wrist.minute_biomarkers
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from streaming.kafka.schemas import MinuteBiomarkerEvent


def main() -> None:
    try:
        from kafka import KafkaProducer
    except ImportError as e:
        raise SystemExit(
            "Install kafka-python: pip install kafka-python\n"
            "Or run your cluster's producer; this script is an example only."
        ) from e

    servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.environ.get("KAFKA_TOPIC_MINUTE", "wrist.minute_biomarkers")

    producer = KafkaProducer(
        bootstrap_servers=servers.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,
    )

    t0 = int(time.time() * 1000)
    for i in range(3):
        ev = MinuteBiomarkerEvent(
            timestamp_unix_ms=t0 + i * 60_000,
            eda_scl_usiemens=0.5 + 0.01 * i,
            temperature_celsius=36.5,
            pulse_rate_bpm=70.0 + i,
            participant_id="demo-597",
        )
        producer.send(topic, value=ev.to_dict())
        print("sent", ev.to_dict())
    producer.flush()
    producer.close()


if __name__ == "__main__":
    main()
