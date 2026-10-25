# Changelog

## Unreleased

- Baseline (2026-10-21): the sibling `kaawayan` suite passed 331 tests with 4
  skips. The dashboard suite passed 34 tests and failed 2 because its saved
  design templates still contained remote assets, violating the offline UI
  contract and causing the rendered-page audit to fail.
- Started consolidating the library and tests into this repository so a root
  install and test run can cover the full software stack.
- Merged the library and tests into `kaawayan/`, added the root editable
  package and coverage-gated test configuration, and removed saved design
  templates' remote dependencies from dashboard runtime assets.
- Added deployment config validation, ramp-test firmware support, startup
  firmware-format checking, log rotation/SIGTERM cleanup, preflight checks,
  launch scripts, hardware bring-up documentation, and simulation report
  generation. Real-hardware acceptance capture remains unimplemented.
- Replaced the obsolete firmware protocol guide with the v1 20-byte-header,
  CRC-32, STARTUP/AUDIO/ENV contract and added cross-platform CI.
- Fixed BME280 probe placement and preserved the specific failure status if
  sensor initialization fails after a successful chip-ID check.
- Included the golden protocol fixture in release bundles and prevented the
  generated manifest from duplicating a repository file.
- Verified the unified Python 3.13 suite: 375 passed, 4 skipped; the skips all
  require a raw firmware frame dump from physical hardware. Coverage was
  91.68%, above the configured 60% floor.
- Verified `main.py --selftest --demo` and the nine-stage simulated acceptance
  run. Both explicitly label results as simulated; no hardware is verified.
- Rebuilt the release ZIP, installed it editable in a fresh Python 3.13
  virtual environment with `[dash,test]`, passed demo self-test, and ran the
  packaged suite: 374 passed, 5 skipped (the optional classifier test and
  four tests requiring raw frames from real hardware).
