# KaawayAn Firmware

Firmware for KaawayAn acoustic nodes on ESP32-S3. Streams raw stereo 48 kHz audio and 1 Hz environmental data over USB CDC.

## Toolchain & Pinning
- **ESP-IDF Version**: ESP-IDF 5.x (Exact version on host: unknown; `idf.py` not found in PATH during host generation).
- **Target Chip**: ESP32-S3 (dual-core Xtensa LX7, 240 MHz).
- **Toolchain Components**:
  - `esp_tinyusb` pinned in `main/idf_component.yml` to `~1.4.5`.

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

### 1. Seeed Studio XIAO ESP32-S3 (Node 1)

#### Bash:
```bash
cd firmware
idf.py -B build_xiao -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.xiao" set-target esp32s3
idf.py -B build_xiao build
idf.py -B build_xiao -p /dev/ttyACM0 flash monitor
```

#### PowerShell:
```powershell
Set-Location firmware
idf.py -B build_xiao -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.xiao" set-target esp32s3
idf.py -B build_xiao build
idf.py -B build_xiao -p COM3 flash monitor
```

---

### 2. ESP32-S3 DevKit N16R8 (Node 2)

#### Bash:
```bash
cd firmware
idf.py -B build_devkit -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.devkit" set-target esp32s3
idf.py -B build_devkit build
idf.py -B build_devkit -p /dev/ttyACM1 flash monitor
```

#### PowerShell:
```powershell
Set-Location firmware
idf.py -B build_devkit -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.devkit" set-target esp32s3
idf.py -B build_devkit build
idf.py -B build_devkit -p COM4 flash monitor
```

---

## Identifying USB Serial Numbers

The laptop identifies nodes strictly by USB serial number, which is derived from the hardware eFuse MAC address.

Run:
```bash
python -m kaawayan.capture --list-ports
```

Copy the reported serial numbers into `config/site.toml` under `[[nodes]]`:
```toml
[[nodes]]
node_id = 1
usb_serial = "378396343534"  # example MAC serial

[[nodes]]
node_id = 2
usb_serial = "378396343538"
```

---

## Human Verification Procedure (Ramp T1 Test)

1. Flash Node 1 (XIAO) and Node 2 (DevKit).
2. Query serials: `python -m kaawayan.capture --list-ports`.
3. Put the serials into `config/site.toml`.
4. Capture a 600 s (10-minute) trial for each board:
   ```bash
   python -m kaawayan.capture --duration 600 --kind tone --notes "ramp T1 jtag"
   ```
5. Run the ramp integrity checker:
   ```bash
   python laptop/scripts/verify_ramp.py trials/<trial_id>/node1.wav
   python laptop/scripts/verify_ramp.py trials/<trial_id>/node2.wav
   ```
6. Check `trials/<trial_id>/node{id}_env.csv` to verify `overrun_count` and `usb_drop_count`.

---

## F4 Backend Comparison Decision Matrix

Compare USB Serial/JTAG vs TinyUSB CDC backends over a 10-minute capture on both boards. Choose the backend with zero UNDETECTED_LOSS and the lowest drops.

| Board | Backend | Total Gap (ms) | Resets | Overrun Count | USB Drop Count | `verify_ramp.py` Status | Chosen |
|---|---|---|---|---|---|---|---|
| Node 1 (XIAO) | USB Serial/JTAG | | | | | | [ ] |
| Node 1 (XIAO) | TinyUSB CDC | | | | | | [ ] |
| Node 2 (DevKit) | USB Serial/JTAG | | | | | | [ ] |
| Node 2 (DevKit) | TinyUSB CDC | | | | | | [ ] |
