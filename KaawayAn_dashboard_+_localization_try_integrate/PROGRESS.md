Done: Phase 0 – verified 331 passed (4 skipped) in LOCALIZATION; pyproject.toml present in LOCALIZATION\laptop and kaawayan installed editable into live venv; DASHBOARD\config\site.toml established as deployment config with CLI pointer notes.
Next: Phase 1 – live adapter in DASHBOARD\main.py replacing DASHBOARD\localization\, staging\receiver.py, and event_handler\handler.py; separate classifier worker, bounded queue, --demo, --record, and requirements files.
Open: Real node positions, headings, and USB serials in config\site.toml are placeholders until surveyed on hardware; SerialSource UNTESTED ON HARDWARE.
Done: Phase 1 – main.py wired to kaawayan.LivePipeline, SerialSource, FrameAssembler, TrialRecorder, and ClassifierWorker; DASHBOARD\localization\, staging\receiver.py, and event_handler\handler.py removed; requirements files created; all 331 LOCALIZATION tests and 11 DASHBOARD tests pass.
Next: Phase 2 – database schema additive migrations, WAL, busy_timeout, mode=ro read-only connections, and faithful logging of status (ok|uncertain|degraded|none) and bearings.
Open: Real node positions, headings, and USB serials in config\site.toml are placeholders until surveyed on hardware; SerialSource UNTESTED ON HARDWARE.
Done: Phase 2 – database schema additive migrations in place (t_wall, drone_recent, dead_nodes, bearings table); WAL + busy_timeout enabled; dashboard read-only mode=ro connections; LivePipeline faithfully stored and "none" row logged on gate-fail groups.
Next: Phase 3 – dashboard fixes: gating position dot on status=="ok" and age<3s; node poses and arena bounds from site.toml; bearing rays from bearings table; node online status from t_wall; offline/fallback rendering.
Open: Real node positions, headings, and USB serials in config\site.toml are placeholders until surveyed on hardware; SerialSource UNTESTED ON HARDWARE.
Done: Phase 3 – dashboard updated to gate position dot strictly on status=="ok" and age<3s; node poses and arena bounds loaded dynamically from site.toml; bearing rays drawn per-node from bearings table; node online status drawn from t_wall; empty DB and fallback handling added.
Next: Phase 4 – comprehensive test suite in DASHBOARD\tests covering simulated source accuracy (median error <= 0.05m), noise-only zero "ok" fixes, 16000-sample classifier window and non-blocking queue, invalid BME temperature rejection, old-schema migration, and demo mode.
Open: Real node positions, headings, and USB serials in config\site.toml are placeholders until surveyed on hardware; SerialSource UNTESTED ON HARDWARE.
Done: Phase 4 – DASHBOARD\tests suite verified: clean simulated source median error 0.0075 m <= 0.05 m; noise-only produced zero "ok" rows; classifier window exactly 16000 samples and bounded queue non-blocking; BME status != 0 temperature update rejected; old-schema DB migrated and read; demo mode verified end-to-end; 17 tests passed in DASHBOARD, 331 passed (4 skipped) in LOCALIZATION.
Next: Phase 5 – bundle command in DASHBOARD\tools\bundle.py with inclusions, exclusions, manifest, size warning, and test.
Open: Accuracy verified in simulation only; all pipeline components UNTESTED ON HARDWARE; real node positions, headings, and USB serials in config\site.toml remain placeholders until hardware survey.
Done: Phase 5 – tools/bundle.py implemented (stdlib only, Windows-safe) supporting --with-model and --with-localization; MANIFEST.txt and 2MB warning included; 2 bundle tests added and passed (19 DASHBOARD tests total, 331 LOCALIZATION tests passing).
Next: Phase 6 – documentation updates: DASHBOARD\CONTEXT.md, build.md rewrite, and README.md cleanup.
Open: Real node positions, headings, and USB serials in config\site.toml remain placeholders until hardware survey; SerialSource and physical pipeline UNTESTED ON HARDWARE.
Done: Phase 6 – documentation completed: DASHBOARD\CONTEXT.md created with DB contracts; build.md rewritten as a single concise guide; README.md purged of obsolete architecture claims and non-hardware claims.
Next: Complete integration handoff and deployment testing on physical hardware.
Open: Real node positions, headings, and USB serials in config\site.toml are placeholders until surveyed on hardware; SerialSource and physical pipeline UNTESTED ON HARDWARE.

## Production-readiness catch-up — 2026-10-21

The previous phase notes above describe the original dashboard/library split.
The current checkout has since been changed to a consolidated root layout.

- Baseline: dashboard `python -m pytest -q` had 34 passed and 2 failures caused
  by remote CDN/font/image references in runtime templates. The standalone
  library suite had 331 passed and 4 skipped.
- Added the library package, scripts, golden test fixtures, and library tests
  under root `kaawayan/`; root editable install is available as `pip install -e .`.
- Removed tracked generated environment/database/WAV assets from the index and
  added ignore rules. Database files are created by the runtime, not shipped.
- Removed remote assets from runtime dashboard templates; raw templates,
  screenshots, and design notes now live under `docs/design_reference/`.
- Added stricter site config validation, classifier test skips, full root test
  discovery with a 60% library coverage floor, root test dependencies, and CI
  for Linux/Windows on Python 3.11/3.13.
- Updated both firmware sketches with the compile-time ramp generator and a
  transmit-buffer availability guard; hardware behavior is unverified.
- Added startup firmware format checks, disk-space/serial/frame preflight
  diagnostics, rotating logs, SIGTERM cleanup, launchers, revised protocol
  documentation, and a report generator.
- `python -m kaawayan.acceptance --config config/site.toml --out reports --simulate`
  produced **SIMULATED PASS** for nine software/synthetic checks. The real-device
  capture workflow is not yet implemented. No hardware acceptance is claimed.
- The full unified Python 3.13 suite now passes: **375 passed, 4 skipped**,
  with 91.68% library coverage against a 60% floor. The four skips require a
  raw frame dump from real firmware hardware. The release-bundle test also
  verifies the golden protocol fixture is included and that archive entries
  are unique.
- `python main.py --selftest --demo` passed and explicitly reported simulation
  preflight; this does not verify serial discovery or physical sensor behavior.
- Rebuilt the release ZIP and validated it in a fresh Python 3.13 virtual
  environment using `pip install -e ".[dash,test]"`. The packaged demo
  self-test passed; the complete packaged test suite reported **374 passed,
  5 skipped** with 91.68% coverage. Skips were one optional-classifier test
  (no torch in the clean environment) and four tests requiring a real
  firmware frame dump.
- `python -m compileall -q main.py live_adapter.py kaawayan dashboard database
  tools tests`, `git diff --check`, and a source/docs/tools search for the
  former absolute `C:\Users\Admin\...` paths completed without errors or
  matches. Linux CI, firmware compilation, and physical acceptance remain
  unverified.

Still open: actual sensor survey and serial identities, BME/USB/I2S behavior,
firmware compilation and flashing, real STARTUP/ENV/audio acceptance capture,
calibration, field localization, outdoor classifier validation, and Windows /
Linux clean-machine installation verification.
