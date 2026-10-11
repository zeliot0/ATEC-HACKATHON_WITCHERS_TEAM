# C.R.A.T.E.R

**Constrained Reconnaissance, Autonomy & Telemetry in Extreme Regions**

ATEC'26 — ESPRIT IEEE Student Branch (AESS / MTT-S)
Solo submission — Salim

A fully software-emulated, end-to-end pipeline that takes a lunar rover from raw camera frame to a live, secured Mission Control dashboard, over a deliberately hostile, bandwidth-starved, lossy link — the exact scenario a rover inside a permanently shadowed crater actually faces (no GPS, no cellular, no cloud, regolith-absorbed radio).

Every stage in this repo is a real, runnable implementation — perception, tracking, localization, a binary telemetry protocol, forward error correction, authenticated encryption, a realistic packet-loss simulation, a fully offline AI advisor, and a real-time dashboard. Nothing here is mocked or hardcoded to look good in a demo.

---

## Table of contents

- [What this is](#what-this-is)
- [Architecture](#architecture)
- [Features in detail](#features-in-detail)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
- [Running it](#running-it)
- [The dashboard](#the-dashboard)
- [Security model](#security-model)
- [Testing & validation](#testing--validation)
- [Benchmark results](#benchmark-results-measured-not-estimated)
- [Design decisions, in one sentence each](#design-decisions-in-one-sentence-each)
- [Honest limitations & future work](#honest-limitations--future-work)
- [Q&A prep](#qa-prep--hard-questions-we-already-have-answers-for)

---

## What this is

A rover operating in a permanently shadowed lunar crater has to:

1. **Perceive** hazards (rocks, drops) in near-total darkness, across the full 360° around it, not just straight ahead.
2. **Transmit** what it sees home over a radio link that drops packets in bursts, adds latency and jitter, and is hard-capped in bandwidth.
3. **Protect** that transmission — the link is treated as actively hostile, not just unreliable.
4. **Present** all of this live, honestly, including when things are degrading.

C.R.A.T.E.R implements all four, as one connected pipeline, with a dashboard that shows the system working — and failing — in real time.

## Architecture

```
synthetic camera frame (full 360° via sector sweep)
    │
    ▼
┌─────────────┐  ┌──────────┐   pixel (px,py)   ┌──────────────┐   (x,y,unc_m)   ┌───────────────┐
│ STEREO DEPTH │  │  YOLOv8  │ ───────────────▶ │ LOCALIZATION │ ──────────────▶ │   TELEMETRY    │
│  disparity   │  │ (or depth│   + confidence    │  pinhole +   │  + covariance   │ ENCODER        │
│  two-pass CV │  │ fallback)│                   │  ground-plane │  ellipse        │ FULL/DELTA     │
└──────┬───────┘  └────┬─────┘                   │  intersection │                 │ CRC8 + XOR FEC │
       │               │                          └──────────────┘                 └───────┬────────┘
       └──────┬────────┘                                                                    │ bytes
              ▼                                                                             ▼
      ┌───────────────┐                                                           ┌───────────────────┐
      │   TRACKING    │                                                           │    ENCRYPTION      │
      │  Hungarian     │                                                           │ ChaCha20-Poly1305   │
      │  algorithm,    │                                                           │ AEAD, per-packet    │
      │ stable IDs     │                                                           │ nonce, HKDF-SHA256  │
      └───────────────┘                                                           └─────────┬──────────┘
                                                                                              │ ciphertext
                                                                                              ▼
                                                                                    ┌───────────────────┐
                                                                                    │   LINK EMULATOR    │
                                                                                    │ real UDP sockets   │
                                                                                    │ Gilbert-Elliott     │
                                                                                    │ burst loss, jitter, │
                                                                                    │ token-bucket cap    │
                                                                                    │ (live-tunable)      │
                                                                                    └─────────┬──────────┘
                                                                                              │ ciphertext
                                                                                              ▼
                                                                                    ┌───────────────────┐
                                                                                    │  RECEIVER           │
                                                                                    │  decrypt + verify   │
                                                                                    │  auth tag → decode  │
                                                                                    │  + FEC reconstruct  │
                                                                                    └─────────┬──────────┘
                                                                                              │ JSON over WS
                                                                                              ▼
                                                                                    ┌───────────────────┐
                                                                                    │ MISSION CONTROL     │
                                                                                    │ FastAPI + WebSocket │
                                                                                    │ 3 views: Mission /  │
                                                                                    │ Orbital / Security   │
                                                                                    └───────────────────┘
```

Every hazard detection is written to `trace_log.jsonl` at each hop (`detected → localized → transmitted → authenticated/auth_failed → decoded_full/delta → fec_reconstructed`), so one hazard's whole journey can be pulled up live and narrated with real timestamps — see `/api/trace` on the dashboard.

## Features in detail

### Perception — two independent detection paths
- **Stereo disparity**: `depth = (focal_length × baseline) / disparity`. Two synthetic camera views at a known baseline; closer objects shift more between them.
- **YOLOv8 object detection** (`src/yolo_detector.py`, `yolov8n.pt`): a trained neural detector running alongside the depth path. If the weights aren't available, the system **falls back to depth-only automatically** — graceful degradation, not a hard crash.
- **360° sector sweep**: the camera's field of view only covers one slice at a time. The rover rotates through fixed sectors on a schedule (`N_SECTORS`, `SECTOR_STEP_DEG`), and detections from every sector are merged into one persistent full-circle hazard map — the same way real rover mast cameras achieve full coverage.

### Tracking — stable hazard IDs
A **Hungarian-algorithm assignment** matches each frame's new detections to existing tracked hazards by position, minimizing total assignment distance. This is what lets the same physical rock keep the same ID across frames and across sectors, instead of being re-detected as "new" every time — which is also what makes cheap DELTA packets possible (see Telemetry below).

### Localization — position *and* confidence
- **Camera model**: a pinhole + ground-plane intersection turns a detected pixel into real-world rover-frame coordinates (`landing_frame_to_pixel`, round-trips to <1e-6 px error — see `tests/test_localization.py`).
- **Uncertainty (covariance ellipse)**: every position estimate carries a first-order uncertainty estimate (`distance² / height` amplification of 1-pixel angular error), visualized as an ellipse that shrinks as a hazard is tracked longer and from closer range. Explicitly *not* oversold as a full EKF/covariance treatment — see Limitations.

### Telemetry protocol — bandwidth-honest by design
- **FULL vs DELTA packets**: a FULL packet carries the whole hazard state; a DELTA packet, once a hazard is already tracked, carries only what changed.
- **Fixed-point integers**, not floats — smaller, deterministic packet sizes.
- **CRC8 checksum** on every packet, detecting corruption before it's trusted.
- **XOR-based Forward Error Correction**: a parity packet `P = D1 ⊕ D2 ⊕ ... ⊕ Dn` over each block lets the receiver reconstruct one lost packet per block with **zero retransmit round-trips** — essential when round-trip time is too slow or too costly to rely on.
- **Priority-weighted send order** with redundant sends for high-confidence, close-range hazards under bandwidth pressure.

### Lossy link simulation — a network that behaves like a network
- **Real UDP sockets** (`sender → emulator → receiver`), not a list with entries dropped in software.
- **Gilbert-Elliott burst-loss model**: the link toggles between a GOOD state (0% loss) and a BAD state (100% loss) as a two-state Markov chain, tuned so the long-run *average* loss matches the configured target — but losses arrive in realistic correlated bursts, not independent coin-flips.
- **Latency + jitter**: every surviving packet is delayed by a base latency plus random jitter.
- **Token-bucket bandwidth cap**: a hard byte-rate ceiling; packets that exceed the current budget are dropped, exactly like a saturated real downlink with no infinite buffer.
- All four parameters are **live-tunable from the dashboard** — loss can be cranked to 90% mid-demo and the system degrades honestly instead of hanging or crashing.

### Security layer — the link is treated as hostile
- **ChaCha20-Poly1305 AEAD** (`src/security.py`): one pass gives both confidentiality (ciphertext is unreadable to an eavesdropper) and authenticity (any tampering is detected via the authentication tag). Chosen over AES because it's fast in pure software, with no hardware acceleration dependency — relevant for constrained embedded hardware.
- **Per-packet nonce** — never reused with the same key.
- **HKDF-SHA256** key derivation — sender and receiver derive the same session key from a shared secret, without the key itself ever crossing the link.
- **Live "Simulate Attack" demo**: a dashboard button flips a real byte in a real ciphertext packet mid-flight. The receiver's authentication check fails and is logged (`auth_failed`) on screen in real time — not a scripted animation.

### Local AI advisor — fully offline, no leaked keys
`src/dashboard/local_advisor.py` originally called a cloud LLM API, which meant shipping an API key in a public repo — a real security liability. It was replaced with a fully local, offline retrieval system:
- **TF-IDF** (term frequency × inverse document frequency) scoring against a small hand-written mission knowledge base.
- **Cosine similarity** retrieval to find the most relevant advice for the current telemetry state.
- A **rule engine** that reasons over live numbers (loss rate, FEC recoveries, hazard count) rather than re-parsing prose.

No network call, no API key, nothing leaves the process. Deliberately less generative than an LLM, in exchange for zero attack surface and full explainability — a trade made on purpose, documented here rather than hidden.

### Mission Control dashboard — three focused views
FastAPI + WebSocket, real-time push (not polling) — the moment a hazard is decoded, every connected browser sees it.
- **Mission View** — live camera feed with detection overlays: what the rover actually sees.
- **Orbital View** — top-down hazard & position map: where everything is relative to the rover.
- **Security View** — encryption state and the live attack demo: is the link trustworthy right now.

A live `/api/pipeline_control` and `/api/link_config` let you change loss rate, latency, bandwidth, rover speed, FEC on/off, detector mode and the 360° sweep live, with the system reacting on its next frame.

## Project layout

```
src/
  perception.py          synthetic frame generator + stereo/two-pass CV hazard detector
  yolo_detector.py        YOLOv8 wrapper with graceful fallback to depth-only
  localization.py          pinhole + ground-plane projection, covariance-ellipse uncertainty
  telemetry.py             binary protocol: FULL/DELTA packets, CRC8, XOR FEC, priority encoder
  link_sim.py               UDP-based Gilbert-Elliott loss / jitter / bandwidth-capped emulator
  security.py                ChaCha20-Poly1305 AEAD, HKDF-SHA256 key derivation
  dashboard/
    server.py                FastAPI + WebSocket backend, REST control endpoints
    local_advisor.py          offline TF-IDF + cosine-similarity RAG advisor
    static/index.html         Mission Control UI (Mission / Orbital / Security views)
run_pipeline.py            wires everything end-to-end: 360° sweep, encryption, trace logging
benchmark.py                produces benchmark_results.json
tools/
  make_yolo_dataset.py      synthetic dataset generator for YOLO training
  train_yolo.py              YOLOv8 training script
datasets/hazards/           generated training data
tests/                      pytest unit tests: telemetry, localization, link_sim
trace_log.jsonl             per-hazard hop-by-hop event log, written at runtime
```

## Getting started

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.26
opencv-python-headless>=4.9
fastapi>=0.110
uvicorn[standard]>=0.29
websockets>=12.0
pytest>=8.0
scipy
cryptography>=42.0
```

## Running it

```bash
python run_pipeline.py
```

Then open **http://127.0.0.1:8000**. Ctrl+C to stop.

Use the dashboard's control panel to:
- Crank loss/latency/bandwidth live on the running system via "Link Emulator Control"
- Toggle FEC, the 360° sweep, and the detector mode (depth / YOLO / both)
- Fire the live attack demo from the **Security View** tab

Run the test suite and the benchmark separately:

```bash
python -m pytest tests/ -v
python benchmark.py        # writes benchmark_results.json
```

## The dashboard

| View | What it shows |
|---|---|
| **Mission View** | Live camera feed with hazard overlays, rover heading, frame counter |
| **Orbital View** | Top-down hazard map with covariance ellipses, link-health panel, pipeline stats |
| **Security View** | Encryption status, auth success/failure counters, live "Simulate Attack" button |

If the WebSocket connection drops, the UI shows "disconnected" and auto-retries every 1.5s; hazards already on screen age out and gray to "STALE" rather than freezing silently.

## Security model

The link between sender and receiver is treated as **actively hostile**, not just lossy:

1. Every telemetry packet is sealed with **ChaCha20-Poly1305** before it reaches the link emulator.
2. A fresh **nonce** is generated per packet.
3. Both sides derive the same session key via **HKDF-SHA256** from a shared secret that is never transmitted.
4. On receipt, the authentication tag is verified **before** the packet is decoded. A failed check is logged as `auth_failed` and the packet is discarded — it never reaches the telemetry decoder.
5. The dashboard's **Simulate Attack** button demonstrates this live: it tampers with a real in-flight ciphertext byte and shows the resulting authentication failure in real time.

This replaced an earlier design where the dashboard's AI advisor called a cloud LLM API with an embedded key — a liability that doesn't belong in a public repository. See [Local AI advisor](#local-ai-advisor--fully-offline-no-leaked-keys) above.

## Testing & validation

- **15 automated tests** (`pytest tests/`) covering telemetry encode/decode round-trips, localization/camera-model math (round-trips to <1e-6 px), and the link simulator's statistical loss behavior.
- **Manual Playwright (headless-browser) verification** of the dashboard UI — clicking through all three views to confirm correct rendering, not just reading the code. This caught and fixed a real view-switching CSS regression during development.
- The link-loss statistical test uses a wide sample (N=1200) and tolerance sized to the Gilbert-Elliott model's genuine run-to-run variance over a finite sample — the *stationary* average loss matches the configured target exactly; a short run's observed loss naturally varies around it. This was debugged deliberately, not silenced.

## Benchmark results (measured, not estimated — see `benchmark_results.json`)

**Perception** — 120 synthetic frames, 6 hazards each:
- **730.9 FPS** on CPU (classical CV, no GPU, no cloud call)
- **Precision 0.991, Recall 0.964** against pixel-level ground truth

**Telemetry compression** — vs. the equivalent JSON payload (77 bytes):
- FULL packet: **14 bytes → 5.5× smaller**
- DELTA packet (re-observed hazard): **9 bytes → 8.6× smaller**

**Link survival under induced loss** (400 FULL packets, FEC block size 4):

| Configured loss | Raw survival (no FEC) | Survival with FEC |
|---|---|---|
| 0% | 100.0% | 100.0% |
| 10% | 87.7% | **93.8%** |
| 30% | 70.8% | **78.7%** |
| 50% | 47.2% | 49.5% |
| 70% | 33.5% | 34.3% |

FEC buys a real, measurable survival gain in the 10–30% loss regime (its design point — recovering exactly one loss per block of 4); at very high loss, multiple losses per block become common and a single-parity code can no longer help — an honest, explainable limit, not a hidden one.

**Localization** — camera at 1.2 m height, 22° downward pitch:
- Center-pixel distance: **2.97 m measured vs. 2.97 m closed-form expected → 0.00 m error**
- Off-axis test points: 1.69 m – 3.50 m, all consistent with the pinhole + ground-plane geometry

## Design decisions, in one sentence each

| Component | Choice | Why |
|---|---|---|
| Perception | Classical CV + optional YOLOv8, not a single trained model | Deterministic, explainable fallback path when no real labeled lunar-hazard dataset exists to validate a model against |
| Input | Synthetic generated frames, not a real camera | The spec requires 100% software emulation — a live camera was never the expectation |
| Localization | Hand-built pinhole + ground-plane intersection (numpy), not a library | Every step is provable on a whiteboard and defensible line-by-line in Q&A |
| Uncertainty | First-order range² amplification of 1-pixel angular error | Cheap, honest, physically-motivated — explicitly not oversold as a full EKF treatment |
| Telemetry | Fixed-point binary struct, FULL/DELTA split | 5.5–8.6× smaller than the JSON equivalent; delta updates are nearly free once a hazard is already tracked |
| Resilience | XOR parity (FEC) over blocks of 4 FULL packets | Recovers from a single lost packet per block with zero retransmit round-trip |
| Security | ChaCha20-Poly1305 AEAD, not a bolt-on signature scheme | One pass gives confidentiality and tamper-detection together, fast in pure software |
| AI advisor | Local TF-IDF/cosine RAG, not a cloud LLM | No API key to leak, fully offline, fully explainable — a deliberate security trade |
| Link | Real UDP sockets with a single min-heap scheduler thread | Behaves like an actual network (real timing, real socket buffers); one scheduler thread instead of one `Timer` per packet so it doesn't buckle under bursts |
| Dashboard | FastAPI + WebSocket push | Real-time, no polling; live controls let a judge crank packet loss to 90% and watch the system degrade honestly |

## Honest limitations & future work

- **Full covariance/EKF uncertainty** instead of the current first-order scalar approximation — tracked explicitly as a simplification, not an oversight.
- **A stronger FEC code** (e.g. Reed-Solomon) to recover from more than one loss per block, trading more overhead for more resilience at high loss.
- **Real camera / real odometry integration** — everything here is software-emulated per the spec, but the interfaces (`CameraModel`, `RoverPose`, `detect_hazards(frame)`) are written so a real feed could be substituted without changing anything downstream.
- **The local AI advisor is retrieval + rules, not a generative model** — a deliberate trade for zero attack surface, stated here rather than implied otherwise.

## Q&A prep — hard questions we already have answers for

- **"Why no single trained model for perception?"** — No real labeled lunar-hazard dataset exists; a model trained on a synthetic proxy wouldn't generalize and couldn't be validated honestly. The classical CV path runs at 730 FPS with no GPU; YOLOv8 runs alongside it and the system falls back automatically if its weights are missing.
- **"How does the 360° sweep work — is it really detecting everywhere at once?"** — No. The camera has a narrow real FOV; the rover rotates through fixed sectors on a schedule and the tracker merges per-sector detections into one persistent full-circle map. It's an honest simulation of a narrow-FOV sensor achieving full coverage by turning, the way real rover mast cameras work.
- **"What if the link is one-way — no retransmit possible?"** — That's exactly what the FEC parity packets are for: one lost packet per block of 4 is recovered with zero round trips.
- **"Why does FEC stop helping at 50%+ loss?"** — A single XOR parity code can only correct one erasure per block; at high loss, multiple losses per block become likely. Shown directly in the benchmark table, not hidden.
- **"Is the AI advisor a real LLM?"** — No, and that's deliberate. It used to call a cloud API with an embedded key — a security liability for a public repo. It was replaced with a fully local TF-IDF + cosine-similarity retrieval system with zero network calls and zero credentials.
- **"Can you prove the encryption actually catches tampering, not just claim it?"** — Yes — the Security View's "Simulate Attack" button flips a real byte in a real in-flight ciphertext packet and the authentication failure is logged live, in front of you.
- **"How is uncertainty computed, and is it a real covariance?"** — It's a first-order linearization: ground error from a 1-pixel angular error, amplified by `distance² / height` for shallow viewing angles. Explicitly not a full EKF/covariance treatment — listed as future work.
- **"Why synthetic camera frames instead of real video?"** — The spec requires a 100% software-emulated format; there was never a physical rover or camera to integrate.
- **"What happens if the dashboard loses the WebSocket connection?"** — It shows "disconnected" and auto-retries every 1.5s; hazards already on screen age out and gray to "STALE" rather than freezing silently.

---

Built solo for ATEC'26 by Salim, ESPRIT.
