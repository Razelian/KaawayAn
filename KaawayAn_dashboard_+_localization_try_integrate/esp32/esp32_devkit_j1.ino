// KaawayAn Node 2: ESP32-S3 DevKit N16R8 + 2x INMP441 (48 kHz stereo I2S) + BME280 (I2C) -> USB CDC.
// Protocol: KaawayAn v1 Framed Protocol (SYNC 0xBA 0xAB, CRC-32).
// IDE Settings:
//   Board: "ESP32S3 Dev Module"
//   USB CDC On Boot: Enabled
//   Upload Mode: USB-OTG CDC (Native USB)
//
// Wiring (Dedicated Pinout):
//   3V3  -> mic VDD x2 + BME VCC
//   GND  -> mic GND x2 + BME GND
//   SCK  -> GPIO 7 -> both mics' SCK / BCLK
//   WS   -> GPIO 8 -> both mics' WS / LRCL
//   DOUT -> GPIO 9 -> both mics' SD / DOUT (with 100 kΩ pull-down to GND)
//   M1 L/R -> GND (Left channel slot)
//   M2 L/R -> 3V3 (Right channel slot)
//   BME280: SDA -> GPIO 5, SCL -> GPIO 6, CSB -> 3V3, SDO -> GND (Addr 0x76)

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <driver/i2s_std.h>
#include <esp_timer.h>
#include <esp_task_wdt.h>

#define NODE_ID         2        // 1 for Node 1 (XIAO), 2 for Node 2 (DevKit)
#define BOARD_TYPE      2        // 1=XIAO, 2=DevKit
#define SERIAL_BAUD     921600
#define SAMPLES_PER_CH  512      // 512 stereo pairs = 10.67 ms per frame
#define I2S_SCK         7        // GPIO 7
#define I2S_WS          8        // GPIO 8
#define I2S_SD          9        // GPIO 9
#define I2C_SDA         5        // GPIO 5
#define I2C_SCL         6        // GPIO 6
#define BME_ADDR        0x76
#ifndef RAMP_TEST
#define RAMP_TEST       0
#endif

// Fast IEEE 802.3 CRC-32
uint8_t probe_bme280() {
    Wire.beginTransmission(BME_ADDR);
    if (Wire.endTransmission() != 0) return 1;
    Wire.beginTransmission(BME_ADDR);
    Wire.write(0xD0);
    if (Wire.endTransmission(false) != 0) return 1;
    if (Wire.requestFrom((uint8_t)BME_ADDR, (uint8_t)1) != 1) return 3;
    uint8_t chip_id = Wire.read();
    return chip_id == 0x60 ? 0 : 2;
}

uint32_t crc32_ieee(const uint8_t *data, size_t length) {
    uint32_t crc = ~0;
    while (length--) {
        crc ^= *data++;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return ~crc;
}

#pragma pack(push, 1)
struct FrameHeader {
    uint8_t  sync[2];       // 0xBA, 0xAB
    uint8_t  version;       // 1
    uint8_t  type;          // 1=STARTUP, 2=AUDIO, 3=ENV
    uint16_t node_id;
    uint32_t seq;
    uint64_t t_us;
    uint16_t len;
};

struct EnvPayload {
    int16_t  temp_c_x100;
    uint16_t rh_x100;
    uint32_t overrun_count;
    uint32_t usb_drop_count;
    uint8_t  bme_status;    // 0 ok, 1 NACK / uninit, 2 wrong chip ID, 3 timeout
};

struct StartupPayload {
    uint8_t  board_type;    // 1=XIAO, 2=DevKit
    uint8_t  fw_major;
    uint8_t  fw_minor;
    uint8_t  fw_patch;
    uint32_t git_short;
    uint32_t sample_rate_hz;
    uint16_t samples_per_ch;
    uint8_t  channels;
    uint8_t  mic_flags;
};
#pragma pack(pop)

static i2s_chan_handle_t rx_handle = NULL;
static int32_t i2s_raw_buf[2 * SAMPLES_PER_CH];      // 512 stereo pairs in 32-bit
static int16_t audio_payload[2 * SAMPLES_PER_CH];   // 512 stereo pairs interleaved int16 (2048 bytes)

static uint8_t frame_buf[sizeof(FrameHeader) + 2048 + 4];

static Adafruit_BME280 bme;
static bool bme_ok = false;
static uint8_t bme_status_code = 1;
static uint32_t audio_seq = 0;
static uint32_t env_seq = 0;
static uint32_t last_env_ms = 0;

static uint32_t overrun_count = 0;
static uint32_t usb_drop_count = 0;

void send_frame(uint8_t type, uint32_t seq, const uint8_t *payload, uint16_t payload_len) {
    FrameHeader *h = (FrameHeader *)frame_buf;
    h->sync[0] = 0xBA;
    h->sync[1] = 0xAB;
    h->version = 1;
    h->type = type;
    h->node_id = NODE_ID;
    h->seq = seq;
    h->t_us = esp_timer_get_time();
    h->len = payload_len;

    memcpy(frame_buf + sizeof(FrameHeader), payload, payload_len);

    // CRC over version..end of payload (sync excluded)
    uint32_t crc = crc32_ieee(frame_buf + 2, (sizeof(FrameHeader) - 2) + payload_len);
    memcpy(frame_buf + sizeof(FrameHeader) + payload_len, &crc, 4);

    size_t total_len = sizeof(FrameHeader) + payload_len + 4;
    if (Serial.availableForWrite() < (int)total_len) {
        usb_drop_count++;
        return;
    }
    size_t written = Serial.write(frame_buf, total_len);
    if (written < total_len) {
        usb_drop_count++;
    }
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    Serial.setTxTimeoutMs(50); // Sensible TX timeout to prevent hung serial writes
    delay(1500); // Wait for USB enumeration

    // Enable task watchdog timer (3 seconds timeout)
    #if defined(ESP_IDF_VERSION_MAJOR) && (ESP_IDF_VERSION_MAJOR >= 5)
        esp_task_wdt_config_t twdt_config = {
            .timeout_ms = 3000,
            .idle_core_mask = 0,
            .trigger_panic = true,
        };
        esp_task_wdt_reconfigure(&twdt_config);
        esp_task_wdt_add(NULL);
    #else
        esp_task_wdt_init(3, true);
        esp_task_wdt_add(NULL);
    #endif

    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setTimeOut(50);
    bme_status_code = probe_bme280();
    bme_ok = bme_status_code == 0 && bme.begin(BME_ADDR);
    if (bme_status_code == 0 && !bme_ok) bme_status_code = 1;

    // I2S Master Receive Configuration @ 48 kHz Stereo
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_frame_num = SAMPLES_PER_CH;
    chan_cfg.dma_desc_num = 4;
    i2s_new_channel(&chan_cfg, NULL, &rx_handle);

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(48000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)I2S_SCK,
            .ws   = (gpio_num_t)I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)I2S_SD,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    i2s_channel_init_std_mode(rx_handle, &std_cfg);
    i2s_channel_enable(rx_handle);

    // Discard initial 100 ms startup transient
    size_t discarded = 0;
    for (int i = 0; i < 10; i++) {
        i2s_channel_read(rx_handle, i2s_raw_buf, sizeof(i2s_raw_buf), &discarded, pdMS_TO_TICKS(50));
    }

    // Send startup handshake frame
    StartupPayload start = {BOARD_TYPE, 1, 0, 0, 0x1234, 48000, SAMPLES_PER_CH, 2,
                            (uint8_t)(0x03 | (RAMP_TEST ? 0x80 : 0x00))};
    send_frame(1, 0, (const uint8_t *)&start, sizeof(StartupPayload));
}

void loop() {
    esp_task_wdt_reset();

    size_t bytes_read = 0;
    esp_err_t err = i2s_channel_read(rx_handle, i2s_raw_buf, sizeof(i2s_raw_buf), &bytes_read, pdMS_TO_TICKS(100));

    if (err == ESP_OK && bytes_read == sizeof(i2s_raw_buf)) {
        // Keep stereo interleaved: L (M1) then R (M2).
        for (int i = 0; i < 2 * SAMPLES_PER_CH; i++) {
#if RAMP_TEST
            uint16_t left = (uint16_t)((audio_seq * SAMPLES_PER_CH + i / 2) & 0xFFFF);
            audio_payload[i] = (i % 2 == 0) ? (int16_t)left : (int16_t)(~left);
#else
            // Convert 32-bit slot samples to int16 (keeping top 16 bits).
            audio_payload[i] = (int16_t)(i2s_raw_buf[i] >> 16);
#endif
        }
        send_frame(2, audio_seq++, (const uint8_t *)audio_payload, sizeof(audio_payload));
    } else {
        // Count real overruns and I2S channel read errors
        overrun_count++;
    }

    // Send BME280 environment telemetry once per second
    uint32_t now = millis();
    if (now - last_env_ms >= 1000) {
        last_env_ms = now;
        EnvPayload env;
        if (bme_ok) {
            float t = bme.readTemperature();
            float rh = bme.readHumidity();
            if (isnan(t) || isnan(rh)) {
                env.temp_c_x100 = 0;
                env.rh_x100 = 0;
                bme_status_code = 3;
            } else {
                env.temp_c_x100 = (int16_t)(t * 100.0f);
                env.rh_x100 = (uint16_t)(rh * 100.0f);
                bme_status_code = 0;
            }
        } else {
            env.temp_c_x100 = 0;
            env.rh_x100 = 0;
        }
        env.bme_status = bme_status_code;
        env.overrun_count = overrun_count;
        env.usb_drop_count = usb_drop_count;
        send_frame(3, env_seq++, (const uint8_t *)&env, sizeof(EnvPayload));
    }
}
