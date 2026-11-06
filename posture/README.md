# Integrated Psychosocial Dashboard + Live Wristband ML

This package combines the historical psychosocial dashboard with the live wristband stream and the machine-learning classifiers built from your uploaded data.

## Polar H10 EXG · 200 ms RMS

The sensor panel (synced from the `VIP_polarH` project) displays the **EXG 200 ms RMS**
signal. It reuses `polar-h10` for Bluetooth and the filter/RMS implementation extracted
from [polar-h10-visualizer](https://github.com/yhzhao343/polar-h10-visualizer):
`src/consts.ts` and `src/PolarH10VisualizerRow.ts`, commit
`771559bc6ba6e2f13b43f02e1e7b026b1ec7473e` (Yuhui Zhao, ISC). Defaults are a Fili
Butterworth highpass (`order: 4`, cutoff 25 Hz), 130 Hz EXG sampling, and a 200 ms /
26-sample RMS window over the filtered signal.

Open the dashboard in Chrome or Edge and click **Connect Polar H10**. Use localhost or
HTTPS and allow Bluetooth access. For firmware 4+, pair the H10 in your OS first. The
panel shows one RMS value and one RMS chart. Disconnect clears the session; missing
samples become unavailable after five seconds, and unexpected disconnections trigger
bounded reconnection attempts.

The integration source is `sensor/driver.js` (device library wrapper),
`sensor/controller.js` (EXG connection state machine), `sensor/data.js` (upstream
filter/RMS), and `sensor/panel.js` (RMS display). `assets/polar-panel.js` is the
pre-bundled build (esbuild output with `fili` and `polar-h10` inlined) that
`psychosocial_dashboard.html` loads directly — no `npm install`/build step is required
to run it as-is. Rebuild it only if you change the `sensor/*.js` source.
No Polar heart-rate, RR/HRV, acceleration, or raw waveform views are included.
Existing wristband data, psychological models, and historical dashboard views remain
independent and unchanged.

## What changed
- The dashboard now pulls live biomarker predictions from `avro_stream_poller_with_ml.py`.
- A **Live worker** is injected into the worker queue for both dashboard lanes:
  - **Frustration**
  - **Mental Demand**
- The dashboard auto-refreshes every 5 seconds.
- The detail panel now shows:
  - live Frustration label and confidence
  - live Mental Demand label and confidence
  - pulse rate from the stream
- Zone, summary, alerts, and worker queue all update to include the live worker.

## Run order
### 1) Start the live wristband poller
```bash
python avro_stream_poller_with_ml.py
```
That serves the live endpoint at:
```text
http://127.0.0.1:7001/latest
```
### Downlaoding dependencies
run pip install -r requirements.txt to download all relevant dependencies


### 2) Start the integrated dashboard server
In a second terminal:
```bash
python serve_dashboard.py
```
Then open:
```text
http://127.0.0.1:8000/psychosocial_dashboard.html
```

## Notes
- The dashboard still includes the original historical sessions.
- The live worker appears with `mode = Live` and updates as new minute-level data arrives.
- Frustration may show **Pending** until the second minute sample arrives, because that model needs two minute-level points.
- The current trained live models use **EDA + temperature + minute index**. Pulse rate is displayed in the dashboard and used for dashboard proxy scoring, but it is not part of the current trained classifier inputs.

## Files
- `serve_dashboard.py` — runs the integrated dashboard server
- `live_dashboard_backend.py` — polls the live stream and injects the live worker into dashboard data
- `psychosocial_dashboard.html` — updated dashboard front end
- `dashboard_data.json` — historical baseline dataset
- `avro_stream_poller_with_ml.py` — live wristband polling + ML prediction API
- `mock_poller.py` — demo-safe synthetic `/latest` source (no AWS needed); see `TTS_SETUP.md`
- `biomarker_runtime.py` — runtime predictor helper
- `models/` — trained model files and metadata
- `speak_biomarker_labels.py`, `construction_stress_speech.py`, `tts_utils.py`, `stress_state.py` — text-to-speech scripts for the construction-worker stress demo; see `TTS_SETUP.md`
- `REALTIME_RESEARCH.md` — research notes on real-time (vs. polling) options for EmbracePlus data
- `sensor/`, `assets/polar-panel.js` — Polar H10 EXG sensor panel, synced from `VIP_polarH`
