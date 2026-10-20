Done: slice 1 – protocol.py (encode/decode/stream-parser + typed helpers), config.py (tomllib load + SHA-256 + validation), config/site.toml, tests/golden/frames.hex; 37 tests pass.
Next: slice 2 – bearing.py + sim.py (synthetic delay generation, GCC-PHAT, sign test per §5).
Open: mic_spacing_m is provisional (T3 not yet done); USB serial values in site.toml are placeholders.
Done: slice 1b – _STARTUP_FMT unsigned (mic_flags=0x80 test), placeholder HDR_FMT line removed, decode_frame rejects version!=1 (CRC-valid path), parse_stream skips bad-version frame, AudioPayload.to_numpy() (shape 512×2 + value equality), config validations (node_id uniqueness, mic_spacing_m>0, band_hz [lo<hi], window_s>0, hop_s>0), per-node defaults; site.toml node positions updated; 51 tests pass.
Next: slice 2 – bearing.py + sim.py (synthetic delays, GCC-PHAT, sign test per §5).
Open: mic_spacing_m still provisional; usb_serial/positions/headings in site.toml are placeholders until surveyed.
Done: slice 2 – bearing.py (speed_of_sound, gcc_phat with PHAT+band mask+parabolic interp+confidence, tau_to_theta, estimate_bearing), sim.py (make_stereo: FFT phase-shift delay, echo, deterministic rng), laptop/scripts/band_diagnostic.py (tone error table + noise/echo sweep table); 71 tests pass (20 new).
Next: slice 3 – firmware v0 (counter-ramp output, then real I2S mics) + record.py (WAV writer + gap log from seq/CRC).
Open: ±60° bearing error at 20 dB SNR is ~0.66° (endfire compression, cos(60°)=0.5); test tolerates 1.0° at ±60°, still well inside T3 (≤8°). Band ambiguity above 1.7 kHz confirmed in diagnostic table (a).
Done: slice 2b – fixed confidence formula (corr_peak*n/(2*sum|G_k|), now ~1.0 for pure delay, ~0.02 noise), fixed PMR to use all lags (noise<3, signal>10), added beta parameter to gcc_phat/estimate_bearing (last kwarg, existing callers unaffected), added phat_beta to config.py (default 1.0, ConfigError if outside [0,1]) and config/site.toml (0.5, provisional), added band_diagnostic tables (c) harmonic sweep and (d) white+echo sweep with SHA-256/git header; 74 tests pass (3 new in test_config.py, all existing tests preserved).
Next: slice 6a – fusion.py (list API, least squares, rejection) + geosim.py.
Open: Table (c) best cell is beta=0, band=[300,1700], SNR=10dB (median |err|=0.079°); table (d) best cell is beta=0.25, band=[300,6000], SNR=10dB (median |err|=0.019°). Slice 4 must confirm with real hardware before setting phat_beta and band_hz in site.toml.
Done: slice 6a catch-up – fixed TestFuseExactTheta target (1,5)→(1,1.5) so crossing≈67°>30°; crossing_deg now None (not 0.0) when too_few_lines (new test); pair_windows + RollingMedianXY already existed and pass; 2-node geosim: median 0.005–0.008 m, p95 0.010–0.021 m; 3-node geosim: median 0.006 m, p95 0.008–0.017 m; 142 passed in 7.98s.
Next: slice 3a – record.py (FrameAssembler, NodeRecorder, SerialSource) + tests/test_record.py.
Open: pair_windows test_input_order_independent uses random.shuffle – seed-independent by design; geosim errors are simulation-only (no real hardware yet).
Done: slice 3a – record.py (FrameAssembler: buffers across feed(), advances 1 byte on CRC failure, keeps partial SYNC tail; NodeRecorder: WAV+env.csv+gaps.log+meta.json, zero-fill gaps, flagged when >100ms; SerialSource: pyserial wrapper); 20 new tests; 162 total passed in 8.84s.
Next: slice 4t – windows.py (iter_windows, band_energy_db) + bench_table.py.
Open: SerialSource UNTESTED ON HARDWARE; close() assembler kwarg is optional (callers must pass it for CRC counter echo); no wall-clock timestamps in meta.json by design.
Done: R1 (bench_table fixed), R2 (TrialRecorder, per-node env, reset/dup handling, crash-safe WAV), slice 5 calibrate.py, slice 6b evaluate.py; 216 tests pass
Next: slice 7 live.py fix + capture.py
Open: all numbers are simulation-only; SerialSource UNTESTED ON HARDWARE.
Done: slice 7 live.py (rewrite: absolute sample index timestamps, alive/dead + startup grace, final-group rule, flush, smoother wired, replay_trial interleaved with cfg_path); R4 corrected 3 tests in test_live.py to match startup-grace spec; 239 passed
Next: R5 slice 8 capture.py
Open: __main__ runner and reconnect handling (plan L2) UNTESTED / OUT OF SCOPE; startup grace means the first ~2*window_s of a run reports dead_node_ids==[] for a node that is really dead.

Done: R3b bookkeeping – PROGRESS.md entry for slice 8 / R6 / R7 / R8 / ramp contract; R9 rawdump.py (split_raw_frames, select_golden, CLI) + test_rawdump.py (24 tests: garbage skip, corrupt skip, raw bytes exact, consecutive-seq, force, CLI monkeypatch, 4-check golden); R10 SerialSource hardening (unopened Serial, dtr/rts before open, 1 MiB buffer, serial_number attr) + 11 tests via fake serial module; CONTEXT.md golden-frames line updated; 331 passed 4 skipped.
Next: firmware (slice 3b, hardware only), then real-hardware serial capture and ramp verification.
Open: SerialSource / rawdump CLI UNTESTED ON HARDWARE; capture.py __main__ UNTESTED ON HARDWARE; startup grace means first ~2*window_s dead_node_ids==[] for a truly dead node.
