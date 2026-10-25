# KaawayAn Dashboard

KaawayAn Dashboard is the live base-station interface and ingest runtime for the KaawayAn acoustic monitoring project. It handles multi-node USB serial ingestion, decoupled neural network drone classification, SQLite storage with WAL concurrency, and a Streamlit visualization dashboard.

Localization algorithms, wire protocol framing, bearing calculation, and multi-node triangulation math are provided by the authoritative `kaawayan` package in this repository.

---

## Architecture

```
[Node 1: XIAO ESP32-S3] ──▶ 48 kHz Stereo USB CDC ──┐
  (2x INMP441 + BME280)                             │
                                                    ▼
[Node 2: DevKit N16R8]  ──▶ 48 kHz Stereo USB CDC ──▶ [kaawayan.SerialSource + FrameAssembler]
  (2x INMP441 + BME280)                             │
                                                    ├──▶ [ClassifierWorker Thread (Bounded Queue)]
                                                    │      - Decimate 3:1 (48 kHz ➔ 16 kHz)
                                                    │      - PyTorch CRNN DroneClassifier
                                                    │      - Output logged to detection_events
                                                    │
                                                    └──▶ [kaawayan.live.LivePipeline]
                                                           - Speed of sound c(T) compensation
                                                           - GCC-PHAT bearing per node
                                                           - Paired-window 2D least-squares fusion
                                                           - Rolling median smoothing
                                                           - Rejection (<30° crossing, behind)
                                                                   │
                                                                   ▼
                                                          [SQLite database/events.db (WAL)]
                                                                   │
                                                                   ▼
                                                          [Streamlit Dashboard (mode=ro)]
                                                    (Overview, 2D Floor Plan, Detections, Env)
```

---

## Directory Structure

```
KaawayAn Dashboard/
├── kaawayan/                # Protocol, bearing, fusion, live pipeline, calibration, and evaluation
├── config/
│   ├── site.toml             # Surveyed node poses and deployment arena (placeholders until surveyed)
│   └── acceptance.toml       # Editable acceptance thresholds
├── config.py                 # Classifier thresholds and UI display settings
├── main.py                   # Live ingest entry point (--demo, --record, --no-classifier)
├── live_adapter.py           # Ingest coordinator & ClassifierWorker thread
├── demo_source.py            # Hardware-free synthetic two-node v1 frame generator
├── tools/
│   └── bundle.py             # Release packager (zip with MANIFEST.txt)
├── database/
│   └── database.py           # SQLite connection, additive migrations, WAL mode
├── dashboard/
│   ├── dashboard.py          # Streamlit console (Overview, Localization, Detections, Env)
│   ├── db.py                 # Read-only query layer (mode=ro)
│   └── ui_assets/            # HTML templates
├── model/
│   ├── model.py              # CRNN DroneClassifier architecture
│   └── drone_classifier_aug_mixed.pt # Neural network weights
├── esp32/
│   ├── esp32.ino             # Firmware sketch for Node 1
│   ├── esp32_devkit_j1.ino   # Firmware sketch for Node 2
│   └── protocol.md           # Firmware documentation
├── tests/
│   ├── test_integration.py   # Accuracy, noise-only, BME, migration, and demo tests
│   ├── test_localization_e2e.py # End-to-end mathematical and fusion tests
│   └── test_bundle.py        # Packaging and pattern exclusion tests
├── CONTEXT.md                # Dashboard scope and database contracts
├── pyproject.toml            # Root package, extras, and test/coverage config
├── requirements-lock.txt     # Pinned core and test dependencies
├── PROGRESS.md               # Phase-by-phase completion log
└── build.md                  # Run guide and setup instructions
```

---

## Quick Start

### 1. Install the system and dependencies
```bash
python -m pip install -e ".[live,dash,test]"
```

Add `[classifier]` on platforms with matching PyTorch and torchaudio wheels.
Python 3.11 or newer is required; Python 3.13 is tested.

### 2. Run the offline demo
```bash
# Terminal 1
python main.py --demo --no-classifier

# Terminal 2
streamlit run dashboard/dashboard.py
```

### 3. Run Tests
```bash
python -m pytest
```

The root test suite includes both the `kaawayan/` library and the dashboard,
with a 60% coverage floor. Simulation does not verify physical hardware.
See [docs/HARDWARE_BRINGUP.md](docs/HARDWARE_BRINGUP.md) for deployment steps.

---

## Hardware Status

All pipeline components and localization accuracy metrics are verified in simulation. Real-hardware deployment, node survey verification, and outdoor acoustic validation remain UNTESTED ON HARDWARE.
