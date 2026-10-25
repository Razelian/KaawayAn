# Hardware bring-up and acceptance

## Before connecting the station

1. Install Python 3.11 or 3.13, then from the repository root run:

   ```powershell
   python -m pip install -e ".[live,dash,test]"
   ```

   For classifier inference also install the platform-matched PyTorch and
   torchaudio wheels, then use `python -m pip install -e ".[classifier]"`.
2. Flash the matching firmware sketch in `esp32/`. Select the correct ESP32
   board, enable **USB CDC On Boot**, and use native USB upload. Pin maps live
   beside each sketch; do not assume the board pin mappings match.
3. Record each node's actual USB serial identity, survey its position and
   heading, measure microphone spacing and acoustic offset, and update
   `config/site.toml`. Never copy placeholder/example geometry into a real
   deployment without measurement.
4. Keep `RAMP_TEST` disabled for acoustic testing. Enable it only during the
   loss test, then flash normal firmware again.

## Commands

From the repository root:

```powershell
python -m kaawayan.acceptance --config config/site.toml --out reports/ --simulate
python main.py --selftest
python -m kaawayan.acceptance --config config/site.toml --out reports/
```

The first command validates the software/simulation path and labels all
evidence **SIMULATED**. It is not physical acceptance. `--selftest` checks
deployment prerequisites and must fail while configuration placeholders or
hardware checks remain unresolved. Run the acceptance command without
`--simulate` only with the real surveyed config and connected nodes; it must
not issue a PASS based on simulation.

Reports should contain the configuration SHA-256 and current git revision.
Save both generated `.md` and `.json` files with the deployment record. A
simulated report is never a substitute for a real hardware report.

## Physical checks and evidence

Run the ramp integrity check on each node for at least 60 seconds with ramp
firmware enabled. Any missing/corrupt samples fail the check. Then capture
both nodes together for five minutes, check sequence gaps and audio cadence,
compare inter-node timing with `pair_tolerance_s`, inspect both microphone
channels for liveness/clipping/order, and verify plausible ENV readings.
Perform centerline clap calibration and known-position localization trials.
Do not auto-write suggested `tau0_s` or heading values: review the report and
make a measured, reviewed config change.

For classifier acceptance, supply an approved reference clip. A synthetic
smoke input only checks that model inference can execute; it is not accuracy
evidence. Follow the outdoor drone/non-drone trial plan and preserve raw
recordings for review.

## If a check fails

Paste back:

- the full `python main.py --selftest` output;
- the Markdown and JSON acceptance report;
- the exact config SHA-256 and `git rev-parse --short HEAD` output;
- node ID, firmware version, board, OS, USB serial identity (redact if needed),
  and the failed check name;
- logs from `logs/station.log` and the relevant trial `meta.json`/CSV/WAV
  artifacts.

Do not change measured pose/calibration values to make a report pass.

## Start automatically

- Windows: run `scripts/start_station.ps1` interactively first. For boot start,
  create a Task Scheduler task triggered **At startup**, configured to run as
  the station account with the repository as its working directory. Set the
  action to `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
  <repo>\scripts\start_station.ps1`; enable restart-on-failure and ensure the
  account can access the USB devices.
- Linux: install the repository at `/opt/kaawayan`, create the `kaawayan`
  service user, adjust paths and permissions in
  `deploy/kaawayan-station.service`, then run
  `sudo systemctl enable --now kaawayan-station.service`.
