# Streaming (Kafka)

Example wiring for **wristband → cloud → Kafka → periodic ML analysis** (see `docs/DATAFLOW.md`).

## Layout

| File | Role |
|------|------|
| `kafka/schemas.py` | Dataclass examples for JSON payloads. |
| `kafka/producer_example.py` | Publishes synthetic minute biomarker events (dev only). |
| `kafka/consumer_windowed_predict.py` | Subscribes, buffers rows, runs `predict_frustration_final_df` every `ANALYSIS_INTERVAL_SEC` (default 300). |

## Quick dev loop

1. Start a Kafka broker (e.g. Docker `bitnami/kafka` or Redpanda).
2. Create topic `wrist.minute_biomarkers` (or set `KAFKA_TOPIC_MINUTE`).
3. `export VIP_MODEL_DIR=/path/to/VIP/Data/outputs/ml_results_final_ml`
4. `python streaming/kafka/consumer_windowed_predict.py` (from repo root)
5. In another shell: `python streaming/kafka/producer_example.py`

## Production notes

- Use **TLS + SASL** on `KafkaProducer` / `KafkaConsumer` (not shown in examples).
- Prefer **Avro / Protobuf + Schema Registry** if payloads evolve; this repo uses JSON for simplicity.
- Scale consumers with the same `group.id` for partition parallelism; ensure buffer and window semantics match your clinical use case.
