# KaawayAn: project context (contracts only)

Read this first in every session. It holds the decisions that must not drift.
The full System Plan is NOT needed unless a task names a requirement ID
(F1..F11, L1..L25, T1..T10). Do not guess; if a contract is missing here, ask.

## 1. Scope
- 2 nodes: node 1 = XIAO ESP32-S3, node 2 = ESP32-S3 devkit N16R8. Each node: 2 INMP441
  mics on one stereo I2S bus + 1 BME280. The node does NO processing. It only streams raw
  audio plus temperature/humidity over native USB CDC. Wi-Fi and Bluetooth are never
  initialised.
- Laptop (Python): ingest -> WAV + logs -> bearing per node -> fusion -> position.
- Goal for now: DIRECTIONALITY and LOCALIZATION of one source (speaker first, then a
  hovering DJI Neo 2), horizontal plane only. No height, no range from a single node.
- OUT OF SCOPE for now: dashboard, learned classifier, SQLite. Dashboard integration comes
  later, and another person is building the dashboard.
- Gate (is this window usable?) = GCC-PHAT confidence + a band-energy threshold. No ML.
- Two nodes today. A third node must need only a config entry and a new USB serial.

## 2. Rules for the AI (token and quality guardrails)
1. One task per session: one module plus its test. List the files you will touch first.
2. Write tests first, then make them pass. Do not refactor files outside the task.
3. Pinned deps only: python>=3.11, pyserial, numpy, scipy, pytest, and matplotlib for
   OFFLINE plots only. Config is TOML (stdlib tomllib). Ask before adding any package.
4. Fusion takes a LIST of (node_id, bearing, confidence), never two named arguments.
   Everything is keyed by node_id; nothing may assume exactly two nodes.
5. Little-endian everywhere. SI units in code (metres, seconds, m/s); degrees only in
   config, CSV and UI.
6. Keep answers short: show diffs or new files, no long explanations unless asked.
7. Every script that produces results must write the config hash and git commit with them.
8. At the end of a session, append 3 lines to PROGRESS.md: done, next, open problems.
9. Any metric that feeds a threshold (confidence, ratios, gate values) needs ABSOLUTE tests on
   known inputs (perfect delay -> about 1, noise only -> near 0), not only relative comparisons.

## 3. Audio and timing
- 48 kHz, I2S standard (Philips) mode, 32-bit slots, stereo, ESP32 is master.
- 24-bit mic data converted to int16 by keeping the top 16 bits of each 32-bit sample.
- First ~100 ms after I2S start is discarded on the node.
- Bearing uses both channels at 48 kHz: generalized cross-correlation, spectrum weighted
  X/|X|^beta (`phat_beta`: 1 = classic PHAT, 0 = plain cross-correlation), band-limited
  (`band_hz`), lag search limited to +/-(d/c) plus a small margin, parabolic sub-sample peak
  interpolation. Code default beta = 1.0; site.toml provisional 0.5. Beta and band are chosen
  TOGETHER from data in slice 4 (sweep beta x band).
- confidence = corr_peak * n / (2 * sum|G_k|) over the band bins, in [0,1]: 1 for a pure delay,
  about 0.02 for noise only (n = 48000). peak_median_ratio = peak / median(|corr|) over ALL
  lags (noise about 1.5, a real peak 5 to 40). Gate thresholds come from data, not by hand.
- Windows: 1 s with 0.5 s hop (speaker and hover phases); 0.25 to 0.5 s only in the
  flight stretch.
- Speed of sound: c = 331.3 * sqrt(1 + T/273.15) m/s, T in Celsius from the BME280.
- The two nodes need no shared clock. Their windows are paired by the laptop monotonic
  receive time (within `pair_tolerance_s`), never by node timestamps.
- Calibration facts: 1 sample = 20.8 us = 7.15 mm of path = about 4 deg at broadside for
  d = 10 cm, so sub-sample interpolation and the tau0 offset decide accuracy.
- Known risk 1: a tonal source above c/(2d) (about 1.7 kHz at 10 cm) causes phase ambiguity.
- Known risk 2 (simulated, 2026-10-20): full PHAT fails on tonal/harmonic sources because empty
  noise bins get the same weight as the tone bins. Harmonic source f0=180 Hz, theta=30, d=10 cm:
  beta=1 median error 60 deg (band 300-1500) and ~30 deg (300-6000); beta=0.5 0.3 to 3.4 deg;
  beta=0 0.03 to 0.4 deg (SNR 20 to 0 dB). Table (d) (echo from the same direction in both
  channels) shows beta barely matters there; real reverberation is untested.
  Simulation only: real Neo 2 audio decides.

## 4. Wire protocol (version 1, little-endian, no padding)
Frame = header (20 bytes) + payload (len bytes) + CRC-32 (4 bytes).

Header, struct format `<2sBBHIQH`:

| Field   | Size | Meaning |
|---------|------|---------|
| sync    | 2    | 0xBA 0xAB |
| version | 1    | 1 |
| type    | 1    | 1 STARTUP, 2 AUDIO, 3 ENV |
| node_id | 2    | from the node's config |
| seq     | 4    | counter per node AND per type, 0 at boot |
| t_us    | 8    | node esp_timer microseconds at frame assembly (diagnostic only) |
| len     | 2    | payload bytes |

CRC-32 (IEEE, same as Python `zlib.crc32`) over version..end of payload (sync excluded).
Mandatory test vector on firmware and laptop: crc32(b"123456789") == 0xCBF43926.
The ESP-IDF ROM CRC has init/inversion quirks, so verify it against this vector.

Payloads:
- AUDIO (2): 512 stereo pairs, interleaved int16, left then right. Left = M1 (L/R tied to
  GND), right = M2 (L/R tied to 3V3). Payload 2048 B, frame 2072 B every 10.67 ms
  (~194 kB/s). Sample index = seq * 512.
- ENV (3), once per second, `<hHIIB` (13 B): temp_c_x100 i16, rh_x100 u16,
  overrun_count u32, usb_drop_count u32, bme_status u8.
  bme_status codes: 0 ok, 1 no response from the BME280 (I2C NACK), 2 chip id is not 0x60
  (0x58 = BMP280: temperature valid, humidity 0), 3 read error or timeout. When status != 0
  the node still sends ENV frames (temp and rh = 0 unless noted) so the laptop keeps its counters.
- STARTUP (1), `<BBBBIIHBB` (16 B): board_type u8 (1 XIAO, 2 devkit), fw_major u8,
  fw_minor u8, fw_patch u8, git_short u32, sample_rate_hz u32, samples_per_ch u16,
  channels u8, mic_flags u8 (bit0 M1 responded at boot, bit1 M2). Sent at boot and
  repeated every 10 s so a late or reconnecting laptop still gets it.

Ingest rules: scan for sync, reject len > 4096, verify CRC; on failure advance ONE byte and
rescan. Identify nodes by USB serial number (never by port name); warn if it disagrees
with header node_id. Log gaps per node and per type from seq.
Golden test: `tests/golden/frames.hex` holds known frames that both the C and Python sides
must parse identically. Those frames are made by the Python encoder, so they only prove Python
agrees with itself; frames dumped from real firmware are added as `tests/golden/frames_fw.hex`
(generate with `python -m kaawayan.rawdump`).

## 4b. Ramp test mode (firmware v0)
Audio is synthetic. For a node, let n = seq*512 + i be the global stereo sample index since
boot (i = 0..511 inside AUDIO frame seq). Then L = int16 wrap of (n mod 65536),
R = bitwise NOT of L as int16 (R = -L-1). The pair (0,0) can never occur in ramp data, so
recorder zero-fill (0,0) is distinguishable from data. Ramp mode is announced by mic_flags
bit 7 (0x80) in STARTUP (bits 0,1 keep their meaning).

## 5. Coordinate and bearing convention (convention_version = 1)
- World frame: right-handed 2D, metres, origin at a surveyed floor mark. Angles in degrees,
  counterclockwise from +x.
- Node heading psi: direction of the plate normal, out through the ports toward the area.
- Build rule: looking out along the heading, M1 (left channel) is on the LEFT, M2 (right
  channel) on the RIGHT.
- Bearing theta: source angle relative to the node front, positive counterclockwise (toward
  M1). World azimuth of the source = psi + theta.
- Fusion line for a node at p: `p + t*(cos(psi+theta), sin(psi+theta))`; t < 0 means behind.
- Delay: tau = t(M2) - t(M1); positive when sound reaches the left mic (M1) first.
  theta = asin(c*tau/d), d = port-to-port spacing.
- Pinning test (synthetic): delaying the RIGHT channel by +tau relative to the left must
  give a POSITIVE theta.

## 6. Config (one TOML file; its SHA-256 is stored with every trial)
Global: `convention_version`, `mic_spacing_m` (provisional until T3 locks it),
`band_hz = [lo, hi]`, `phat_beta` (0..1, default 1.0), `window_s`, `hop_s`, `pair_tolerance_s`, `gate_min_confidence`,
`gate_min_band_energy_db`, `min_crossing_deg = 30`.
Per node (`[[nodes]]`): `node_id`, `usb_serial`, `board` (xiao | devkit), `x_m`, `y_m`,
`heading_deg`, `tau0_s` (calibration offset), `channel_swap` (bool), `mic_model`.

## 7. Storage: one folder per trial
- `node{id}.wav`: stereo, 48 kHz, int16, per node.
- `node{id}_env.csv`: one per node; columns t_laptop_s, temp_c, rh_pct, overrun_count,
  usb_drop_count, bme_status (one row per ENV frame).
- `meta.json`: ONE per trial: trial_id, kind, config_hash, git_commit, flagged, and
  nodes: {id: {fw_version, mic_flags, total_gap_ms, resets, crc_failures,
  resync_bytes_skipped, t_first_audio_laptop_s, n_audio_frames}}, plus caller extras
  (true_theta_deg per node, true_xy_m, distance_m, site, notes).
- `bearings.csv` columns: t_laptop_s, node_id, window_idx, theta_deg, confidence,
  peak_median_ratio, gate_ok.
- `positions.csv`: t_laptop_s, x_m, y_m, crossing_deg, n_nodes, status
  (ok | uncertain | degraded).
- `gaps.log`: stream gaps from seq and CRC failures; a trial with total gaps over 100 ms is
  flagged.
Replaying a saved trial must give identical output twice.

## 8. Fusion rules
- Each node's bearing becomes a line from its surveyed position and heading.
- Position = intersection of the lines (least squares if more than two).
- Drop a line when its bearing is uncertain (gate not ok). Reject the result when lines
  cross under `min_crossing_deg` or the crossing is behind a node; report status
  "uncertain".
- Smooth with a rolling median. Test with synthetic bearings and synthetic timestamps,
  including a missing node.
- Expected error is about 1.4 * R * sigma_theta / sin(gamma), gamma = crossing angle.
- crossing_deg = largest pairwise acute angle (0..90) between the kept lines. Position =
  confidence-weighted least squares of perpendicular distances to the lines (works for n >= 2).
  Any kept line with t < 0 at the solution -> status "uncertain".
- A stereo pair cannot tell front from back: a source BEHIND a node gives the same tau as its
  front mirror. Fusion can only expose it through t < 0 or a bad crossing. Tests document this
  limitation; do not try to fix it inside fusion.

## 9. Calibration (redo at each site and each session)
- tau0 per node: clap or tone straight ahead (centerline) should give zero delay; the
  residual is stored in config.
- Heading per node: fit from a speaker at a known position. 1 deg of heading error is
  1 deg of bearing error.
- Positions and headings are surveyed and stored in config.

## 10. Test targets (placeholders)
- T1 bench tone: frequency within 0.5%, no undetected frame loss in 10 min.
- T2 clap: residual delay <= 0.25 sample; gain mismatch <= 1 dB (diagnostic only).
- T3 bearing at 1.5 m: median error <= 5 deg within +/-30 deg, <= 8 deg at +/-60 deg.
  Report an error table by beta x band (slice 4).
- T4 on the bamboo mast: no more than about 2 deg worse than bare.
- T7 speaker position: median error <= 0.75 m. T8 Neo 2 hover: <= 1.5 m.
Report median and 95th percentile, never only the best case.

## 11. Pins (committed hardware; one firmware config for both boards)
GPIO 7 BCLK, GPIO 8 WS, GPIO 9 DOUT, GPIO 5 SDA, GPIO 6 SCL. Never use GPIO 0, 3, 45, 46,
19, 20, and on the devkit also GPIO 26-37. 100 kOhm pull-down on DOUT. BME280 at 0x76,
forced mode, once per second. PSRAM disabled on both boards.

## 12. Repo layout
```
kaawayan/
  CONTEXT.md  PROGRESS.md  requirements.txt  config/site.toml
  firmware/          ESP-IDF 5.x, version pinned; one codebase, two boards
  laptop/kaawayan/   protocol.py config.py bearing.py sim.py geosim.py fusion.py
                     record.py calibrate.py evaluate.py live.py windows.py
  laptop/tests/      test_*.py (golden frames in tests/golden/)
  trials/            one folder per trial, as in section 7
  laptop/scripts/  band_diagnostic.py bench_table.py (offline diagnostics, not tests)
```

## 13. Slices (build order)
1  protocol.py + config loader + golden frames [done]
2  bearing.py + sim.py (synthetic delays, sign test) [done]
2b bearing fix: phat_beta weighting, absolute confidence and peak_median_ratio, harmonic-source
   tests, diagnostic sweep beta x band [done]
6a fusion.py (list API, least squares, rejection) + geosim.py (source x,y -> per-node stereo ->
   bearing -> fused). Pure software, may run in parallel with 3. Cases: 2 nodes, 3 nodes, one
   node missing, low-confidence node, shallow crossing, behind a node.
3a record.py (software: WAV + gap log + FrameAssembler + NodeRecorder) [done]
3b firmware v0 (counter ramp, then real mics) — hardware, not in this batch
   Non-code step before 4: phone recording of the Neo 2 and the speaker (plan 1C/1D).
4t bench spike (windows.py + bench_table.py): clap/tone/clips at 0, +/-30, +/-60 deg;
   error table by beta x band; set band_hz, phat_beta and gate thresholds from data [done]
5  calibrate.py (tau0, heading fit) [done]
6b pair_windows (laptop receive time) + rolling median + evaluate.py (T7 speaker harness) [done]
7  live.py: one JSON line per fused result (future dashboard hand-off) [in progress]
8  capture.py (record CLI + throughput report)