# KaawayAn USB CDC Protocol v1

This document describes the wire format implemented by `kaawayan/protocol.py`.
The protocol parser and acoustic/localization math live only in `kaawayan/`.
The firmware sketches are `esp32/esp32.ino` (XIAO ESP32-S3) and
`esp32/esp32_devkit_j1.ino` (ESP32 DevKit).

## Transport

- One native USB CDC connection per sensor node; the node ID is carried in
  every frame.
- The host opens the configured serial identity using `pyserial`. The baud
  argument is ignored by native USB CDC; `921600` in firmware is not a USB
  link-rate guarantee.
- Binary frames may be split or coalesced arbitrarily by USB. The receiver
  incrementally scans for sync and verifies each complete frame's CRC.
- There is no retransmission. `seq` gaps, CRC errors, node restarts, and
  firmware counters must be monitored by the host.

## Common frame layout

All integer fields are little-endian. A frame has a 20-byte header, payload,
and 4-byte CRC:

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 2 | sync | `BA AB` |
| 2 | 1 | version | `1` |
| 3 | 1 | type | `1=STARTUP`, `2=AUDIO`, `3=ENV` |
| 4 | 2 | node_id | Unsigned node identifier |
| 6 | 4 | seq | Unsigned per-node, per-frame-type sequence counter |
| 10 | 8 | t_us | Node monotonic microseconds; diagnostic, not a shared clock |
| 18 | 2 | len | Payload bytes |
| 20 | len | payload | Frame-type-specific bytes |
| 20+len | 4 | CRC-32 | IEEE/zlib CRC over bytes 2 through 19+len; sync excluded |

`kaawayan.protocol.HDR_SIZE` is 20 bytes and `MAX_PAYLOAD` is 4096 bytes.
Frames with unsupported versions, unknown types, invalid lengths, truncation,
or CRC mismatch are rejected. The stream assembler scans forward to find the
next valid sync/frame.

## STARTUP (type 1)

The payload is 16 bytes with format `<BBBBIIHBB`:

| Field | Type | Meaning |
|---|---|---|
| board_type | u8 | `1=XIAO`, `2=DevKit` |
| fw_major, fw_minor, fw_patch | 3 × u8 | Firmware semantic version |
| git_short | u32 | Optional firmware revision identifier |
| sample_rate_hz | u32 | Expected to be 48000 |
| samples_per_ch | u16 | Expected to be 512 |
| channels | u8 | Expected to be 2 |
| mic_flags | u8 | Firmware-reported microphone/ramp capability flags |

The host uses the declared version and audio format for diagnostics and
acceptance checks. It must not infer surveyed node pose or acoustic
calibration from STARTUP.

## AUDIO (type 2)

The current firmware sends 512 interleaved stereo pairs per frame:

- Sample rate: 48,000 pairs/second.
- Payload: 512 × (`left int16`, `right int16`) = 2048 bytes.
- Each sample is signed little-endian 16-bit PCM.
- Nominal cadence: 512 / 48000 seconds = 10.667 ms per frame.
- Channel mapping is the installed left/right microphone slot order; confirm
  it during hardware acceptance. The node setting and `channel_swap` must
  reflect the measured wiring.

In compile-time ramp test mode (when implemented/enabled for acceptance),
sample index `n = seq * 512 + i`, left is signed-int16 wrap of `n mod 65536`,
and right is bitwise complement of left. Ramp-test results are acceptance
evidence only; never use ramp samples as acoustic recordings.

## ENV (type 3)

The payload is 13 bytes with format `<hHIIB`:

| Field | Type | Meaning |
|---|---|---|
| temp_c_x100 | i16 | Temperature in degrees Celsius × 100 |
| rh_x100 | u16 | Relative humidity in percent × 100 |
| overrun_count | u32 | Cumulative I2S/audio read overrun count |
| usb_drop_count | u32 | Cumulative USB transmission drop count |
| bme_status | u8 | `0=valid`, `1=NACK/uninitialized`, `2=wrong chip ID`, `3=timeout` |

Firmware normally emits ENV approximately once per second. The host may use
temperature for speed-of-sound compensation only when `bme_status == 0` and
the reading is plausible.

## Timing and loss interpretation

`t_us` is local to each node and cannot be compared directly across nodes
without clock synchronization. Host receive timestamps are used for pairing.
`pair_tolerance_s` in `config/site.toml` controls paired-window tolerance;
measure actual inter-node timing during acceptance. Sequence numbers are
independent for STARTUP, AUDIO, and ENV and may restart after a reboot.

## Firmware setup and pin maps

Use the Arduino ESP32-S3/ESP32 board package and Adafruit BME280 library.
Enable **USB CDC On Boot** and select the native USB upload mode supported by
the board. Use a partition scheme with sufficient application space; no
filesystem partition is required by these sketches. The upload port and
runtime USB serial identity are distinct: record the runtime identity into
`config/site.toml` only after surveying the actual deployed nodes.

The pin definitions at the top of each `.ino` file are authoritative. The
XIAO sketch maps I2S BCLK/WS/DIN to GPIO 7/8/9 and I2C SDA/SCL to GPIO 5/6.
Check the DevKit sketch's pin macros and board wiring before powering it;
do not reuse XIAO pin assignments by assumption. Both microphone channels
share the I2S data input and use opposite L/R slot selection. Verify
power, shared clock/data wiring, BME280 address, channel order, sample rate,
and clipping with the hardware acceptance procedure before deployment.

## Source of truth and checks

- Wire encoders, decoders, CRC, and payload structures: `kaawayan/protocol.py`
- Stream assembly, startup/audio/ENV recording: `kaawayan/record.py`
- Site parameters and physical node poses: `config/site.toml`
- Firmware implementation: the two sketches in `esp32/`
- Hardware requirements and report interpretation: `docs/HARDWARE_BRINGUP.md`
