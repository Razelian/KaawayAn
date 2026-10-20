# KaawayAn Firmware

Firmware for KaawayAn acoustic nodes on ESP32-S3. Streams raw stereo 48 kHz audio and 1 Hz environmental data over USB CDC.

## Toolchain & Pinning
- **ESP-IDF Version**: Requires ESP-IDF >= 5.1 (due to `esp_task_wdt_reconfigure`). Exact version on host: unknown (`idf.py` not found in PATH during host generation).
- **Target Chip**: ESP32-S3 (dual-core Xtensa LX7, 240 MHz).
- **Toolchain Components**:
  - `esp_tinyusb` pinned in `components/usb_out/idf_component.yml` to `~1.4.5`.

## Board Configurations
One codebase supports both boards. Per-board settings are selected via sdkconfig defaults:
- **Node 1**: Seeed Studio XIAO ESP32-S3 (8 MB flash, no PSRAM).
- **Node 2**: ESP32-S3 DevKit N16R8 (16 MB flash, PSRAM disabled).

### Dedicated Pinouts (Both Boards)
- **GPIO 7**: I2S BCLK
- **GPIO 8**: I2S WS
- **GPIO 9**: I2S DIN (labeled DOUT on INMP441 microphones; 100 kΩ pull-down)
- **GPIO 5**: I2C SDA (BME280)
- **GPIO 6**: I2C SCL (BME280)
- **Prohibited Pins**: NEVER use GPIO 0, 3, 45, 46, 19, 20, and on devkit also 26–37.

---

## Build and Flash Commands

Always run builds inside the project virtual environment or with ESP-IDF environment activated (`get_idf` or `export.sh` / `Export.ps1`).

### 1. Default USB Serial/JTAG Backend

#### Seeed Studio XIAO ESP32-S3 (Node 1)
##### Bash:
```bash
cd firmware
idf.py -B build_xiao -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.xiao" set-target esp32s3
idf.py -B build_xiao build
idf.py -B build_xiao -p /dev/ttyACM0 flash monitor
```
##### PowerShell:
```powershell
Set-Location firmware
idf.py -B build_xiao -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.xiao" set-target esp32s3
idf.py -B build_xiao build
idf.py -B build_xiao -p COM3 flash monitor
```

#### ESP32-S3 DevKit N16R8 (Node 2)
##### Bash:
```bash
cd firmware
idf.py -B build_devkit -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.devkit" set-target esp32s3
idf.py -B build_devkit build
idf.py -B build_devkit -p /dev/ttyACM1 flash monitor
```
##### PowerShell:
```powershell
Set-Location firmware
idf.py -B build_devkit -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.devkit" set-target esp32s3
idf.py -B build_devkit build
idf.py -B build_devkit -p COM4 flash monitor
```

---

### 2. TinyUSB CDC Backend (Separate Build Dirs)

#### Seeed Studio XIAO ESP32-S3 (Node 1, TinyUSB)
##### Bash:
```bash
cd firmware
idf.py -B build_xiao_tusb -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.xiao;sdkconfig.defaults.tinyusb" set-target esp32s3
idf.py -B build_xiao_tusb build
idf.py -B build_xiao_tusb -p /dev/ttyACM0 flash monitor
```
##### PowerShell:
```powershell
Set-Location firmware
idf.py -B build_xiao_tusb -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.xiao;sdkconfig.defaults.tinyusb" set-target esp32s3
idf.py -B build_xiao_tusb build
idf.py -B build_xiao_tusb -p COM3 flash monitor
```

#### ESP32-S3 DevKit N16R8 (Node 2, TinyUSB)
##### Bash:
```bash
cd firmware
idf.py -B build_devkit_tusb -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.devkit;sdkconfig.defaults.tinyusb" set-target esp32s3
idf.py -B build_devkit_tusb build
idf.py -B build_devkit_tusb -p /dev/ttyACM1 flash monitor
```
##### PowerShell:
```powershell
Set-Location firmware
idf.py -B build_devkit_tusb -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.devkit;sdkconfig.defaults.tinyusb" set-target esp32s3
idf.py -B build_devkit_tusb build
idf.py -B build_devkit_tusb -p COM4 flash monitor
```

> [!WARNING]
> `sdkconfig.defaults*` files are only read when the target build directory's `sdkconfig` does not exist yet.
> After editing any defaults file, delete that build directory (e.g. `rm -r build_xiao` or `Remove-Item -Recurse -Force build_xiao`) or run `idf.py -B build_xiao fullclean` before rebuilding.

---

### Verify The Board Configuration

After the first build, verify the configuration for each board:

#### Bash:
```bash
grep -E "KAAWAYAN_(BOARD|NODE_ID|RAMP_MODE|USB_BACKEND)" build_xiao/sdkconfig
grep -E "KAAWAYAN_(BOARD|NODE_ID|RAMP_MODE|USB_BACKEND)" build_devkit/sdkconfig
```

#### PowerShell:
```powershell
Select-String -Path build_xiao\sdkconfig -Pattern "KAAWAYAN_(BOARD|NODE_ID|RAMP_MODE|USB_BACKEND)"
Select-String -Path build_devkit\sdkconfig -Pattern "KAAWAYAN_(BOARD|NODE_ID|RAMP_MODE|USB_BACKEND)"
```

**Expected output:**
- XIAO (Node 1): `CONFIG_KAAWAYAN_BOARD_XIAO=y`, `CONFIG_KAAWAYAN_NODE_ID=1`, `CONFIG_KAAWAYAN_RAMP_MODE=y`, `CONFIG_KAAWAYAN_USB_BACKEND_JTAG=y`.
- DevKit (Node 2): `CONFIG_KAAWAYAN_BOARD_DEVKIT=y`, `CONFIG_KAAWAYAN_NODE_ID=2`, `CONFIG_KAAWAYAN_RAMP_MODE=y`, `CONFIG_KAAWAYAN_USB_BACKEND_JTAG=y`.

The first captured STARTUP frame must show `board_type 1` / `node_id 1` on the XIAO and `board_type 2` / `node_id 2` on the devkit.

---

## Identifying USB Serial Numbers

The laptop identifies nodes strictly by USB serial number:
- The **USB Serial/JTAG backend** serial number comes from the ESP32-S3 hardware (unique per chip, derived from silicon ROM/eFuse, may be formatted with colons by the host OS).
- The **TinyUSB CDC backend** serial number is the uppercase hex eFuse MAC string set programmatically in `usb_out_tinyusb.c`.

To discover connected serial numbers, run from `laptop/` (or repository root):
```bash
python -m kaawayan.capture --list-ports
```

Copy the reported serial numbers into `config/site.toml` under `[[nodes]]`:
```toml
[[nodes]]
node_id = 1
usb_serial = "378396343534"

[[nodes]]
node_id = 2
usb_serial = "378396343538"
```

> [!NOTE]
> **Known limitation**: On the JTAG backend, a write that times out part-way can emit a truncated frame. The laptop parser (`record.py`) automatically resyncs on CRC failure and logs the sequence gap, so it appears as a detected gap in `gaps.log` and never as silent corruption.

---

## Human Verification Procedure (Ramp T1 Test)

### CLI Reference

`kaawayan.capture` CLI usage:
```
usage: python -m kaawayan.capture [-h] [--config CONFIG] [--out OUT]
                                  [--kind {clap,tone,speaker,hover}]
                                  [--seconds SECONDS]
                                  [--true-theta TRUE_THETA]
                                  [--true-xy TRUE_XY]
                                  [--distance-m DISTANCE_M] [--site SITE]
                                  [--notes NOTES] [--list-ports]
```

`scripts.verify_ramp` CLI usage:
```
usage: python -m scripts.verify_ramp [-h] [--node ID] trial_dir
```

> [!IMPORTANT]
> `python -m kaawayan.capture` connects to and records **all configured nodes** in `config/site.toml` concurrently in a single run. `scripts.verify_ramp` checks the trial directory for all nodes.

### Step-by-Step Execution

1. Flash Node 1 (XIAO) and Node 2 (DevKit).
2. Query serials (from `laptop/` or repository root):
   ```bash
   python -m kaawayan.capture --list-ports
   ```
3. Put the serial numbers into `config/site.toml`.
4. Capture a 600 s (10-minute) trial for the connected nodes (from `laptop/`):
   ```bash
   python -m kaawayan.capture --seconds 600 --kind tone --notes "ramp T1 jtag"
   ```
   *(For TinyUSB, use `--notes "ramp T1 tinyusb"`).*
5. Run the ramp integrity checker on the generated trial directory (e.g. from `laptop/`):
   ```bash
   python -m scripts.verify_ramp ../trials/<trial_id>
   ```
   *(Or from repo root: `python laptop/scripts/verify_ramp.py trials/<trial_id>`)*.
6. Check `trials/<trial_id>/node{id}_env.csv` to inspect `overrun_count` and `usb_drop_count`.

---

## F4 Backend Comparison Decision Matrix

Compare USB Serial/JTAG vs TinyUSB CDC backends over a 10-minute capture on both boards. Choose the backend with zero UNDETECTED_LOSS and the lowest drops.

Record the ESP-IDF version and `git rev-parse --short HEAD` with every row.

| Board | Backend | Total Gap (ms) | Resets | Overrun Count | USB Drop Count | `verify_ramp.py` Status | Chosen |
|---|---|---|---|---|---|---|---|
| Node 1 (XIAO) | USB Serial/JTAG | | | | | | [ ] |
| Node 1 (XIAO) | TinyUSB CDC | | | | | | [ ] |
| Node 2 (DevKit) | USB Serial/JTAG | | | | | | [ ] |
| Node 2 (DevKit) | TinyUSB CDC | | | | | | [ ] |
