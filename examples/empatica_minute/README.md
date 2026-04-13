# Empatica minute CSV example

Contains **code-only** samples (Unity client + poller). **Minute CSVs** live under:

`Data/examples/empatica_minute/` (`eda.csv`, `temperature.csv`, `pulse-rate.csv`; `timestamp_unix` in ms).
- `EmpaticaLatestClient.cs` — Unity HTTP client sample
- `avro_stream_poller.py` — TA-style S3 / Avro poller (configure credentials via environment variables only)

Try prediction from repo root:

```bash
python empatica_minute_filter_predict.py --dir Data/examples/empatica_minute --json-out
```
