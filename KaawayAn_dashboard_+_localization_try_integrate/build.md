# KaawayAn Build and Run Guide

## Install from a clean checkout

Use Python 3.11 or newer (tested on Python 3.13):

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[live,dash,test]"
```

The optional classifier requires platform-matched PyTorch and torchaudio
wheels. Install those for the target platform, then install `.[classifier]`.
Without the model, explicitly set `--no-classifier`.

## Hardware-free demo

```bash
# Terminal 1
python main.py --demo --no-classifier

# Terminal 2
streamlit run dashboard/dashboard.py
```

Demo mode routes events to `database/demo.db`; live mode uses
`database/events.db`. Databases are created on first ingest and are not shipped
in source control. To record a synthetic trial, add `--record`.

## Physical deployment

1. Survey node positions, headings, microphone spacing, acoustic offsets, and
   USB serial identities. Update `config/site.toml`; never leave placeholders.
2. Flash the matching firmware and connect nodes over native USB CDC.
3. Run preflight; it must pass with the real configuration and attached nodes:

   ```bash
   python main.py --selftest
   ```

4. Start ingest and dashboard together:

   ```powershell
   scripts/start_station.ps1
   ```

   On Linux/macOS, run `bash scripts/start_station.sh`.
5. Follow [docs/HARDWARE_BRINGUP.md](docs/HARDWARE_BRINGUP.md) for acceptance.
   Simulation reports are explicitly labeled and do not verify hardware.

## Tests and release bundles

Run the complete library, dashboard, and integration suite from the repository
root:

```bash
python -m pytest
```

Create a source release bundle with manifest (add `--with-model` only when
shipping the model weights):

```bash
python tools/bundle.py [--with-model] [--out dist]
```

The project CI runs this suite on Windows and Linux with Python 3.11 and 3.13.
