#include "capture.h"
#include "board_config.h"
#include "kaawayan_proto.h"
#include <string.h>
#include "esp_timer.h"
#include "esp_log.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_task_wdt.h"

#define TAG "CAPTURE"

volatile uint32_t g_overrun_count = 0;
volatile uint8_t  g_mic_flags = 0;
RingbufHandle_t   g_audio_ring = NULL;

static i2s_chan_handle_t s_rx_handle = NULL;
static uint32_t s_audio_seq = 0;

static esp_err_t init_i2s_channel(void)
{
    if (s_rx_handle != NULL) {
        i2s_channel_disable(s_rx_handle);
        i2s_del_channel(s_rx_handle);
        s_rx_handle = NULL;
    }

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    /* Allocate DMA buffers: 4 buffers of 512 stereo samples (4096 bytes each) */
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = KAAWAYAN_SAMPLES_PER_CH;

    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &s_rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel failed: %d", ret);
        return ret;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(KAAWAYAN_SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)KAAWAYAN_PIN_I2S_BCLK,
            .ws   = (gpio_num_t)KAAWAYAN_PIN_I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)KAAWAYAN_PIN_I2S_DIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;

    ret = i2s_channel_init_std_mode(s_rx_handle, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_init_std_mode failed: %d", ret);
        return ret;
    }

    ret = i2s_channel_enable(s_rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_enable failed: %d", ret);
        return ret;
    }

    return ESP_OK;
}

void capture_init(void)
{
    /* Ring buffer holding at least 32 audio frames */
    /* 32 * 2100 bytes = 67,200 bytes */
    g_audio_ring = xRingbufferCreate(32 * 2100, RINGBUF_TYPE_NOSPLIT);
    if (g_audio_ring == NULL) {
        ESP_LOGE(TAG, "Failed to allocate audio ring buffer");
    }

    init_i2s_channel();

    /* Buffer for 512 stereo samples in 32-bit slots: 512 * 2 * 4 = 4096 bytes */
    static int32_t raw_slots[KAAWAYAN_SAMPLES_PER_CH * 2];
    size_t bytes_read = 0;

    /* Discard the first 10 frames (~107 ms, F11) */
    for (int i = 0; i < 10; i++) {
        if (s_rx_handle != NULL) {
            i2s_channel_read(s_rx_handle, raw_slots, sizeof(raw_slots), &bytes_read, pdMS_TO_TICKS(100));
        }
    }

    /* Mic health check at boot (F7): evaluate ~50 ms (5 frames) of raw I2S data */
    int32_t min_l = 0x7FFFFFFF, max_l = (int32_t)0x80000000;
    int32_t min_r = 0x7FFFFFFF, max_r = (int32_t)0x80000000;
    for (int f = 0; f < 5; f++) {
        if (s_rx_handle != NULL &&
            i2s_channel_read(s_rx_handle, raw_slots, sizeof(raw_slots), &bytes_read, pdMS_TO_TICKS(100)) == ESP_OK) {
            for (int i = 0; i < KAAWAYAN_SAMPLES_PER_CH; i++) {
                int32_t sl = raw_slots[2 * i];
                int32_t sr = raw_slots[2 * i + 1];
                if (sl < min_l) min_l = sl;
                if (sl > max_l) max_l = sl;
                if (sr < min_r) min_r = sr;
                if (sr > max_r) max_r = sr;
            }
        }
    }

    uint8_t flags = 0;
    if (max_l > min_l) {
        flags |= 0x01; /* Bit 0: M1 alive */
    }
    if (max_r > min_r) {
        flags |= 0x02; /* Bit 1: M2 alive */
    }

#if defined(CONFIG_KAAWAYAN_RAMP_MODE)
    flags |= 0x80; /* Bit 7: ramp mode active */
#endif

    g_mic_flags = flags;
}

void capture_task(void *pvParameters)
{
    (void)pvParameters;

    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));

    /* DMA buffer for 512 stereo samples in 32-bit slots (4096 bytes) */
    static int32_t raw_slots[KAAWAYAN_SAMPLES_PER_CH * 2];
    /* Converted 16-bit stereo payload (2048 bytes) */
    static int16_t audio_payload[KAAWAYAN_SAMPLES_PER_CH * 2];
    /* Encoded audio frame (2072 bytes) */
    static uint8_t frame_buf[KAAWAYAN_AUDIO_FRAME_BYTES];

    int consecutive_i2s_errors = 0;

    while (1) {
        esp_task_wdt_reset();

        /* Guard against NULL handle if initialization failed or disconnected */
        if (s_rx_handle == NULL) {
            g_overrun_count++;
            s_audio_seq++;
            vTaskDelay(pdMS_TO_TICKS(100));
            init_i2s_channel();
            continue;
        }

        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(s_rx_handle, raw_slots, sizeof(raw_slots),
                                         &bytes_read, pdMS_TO_TICKS(100));

        if (err != ESP_OK || bytes_read != sizeof(raw_slots)) {
            consecutive_i2s_errors++;
            g_overrun_count++;
            s_audio_seq++; /* Always advance seq */

            if (consecutive_i2s_errors >= 3) {
                ESP_LOGW(TAG, "I2S read failed 3x in a row, reinstalling channel...");
                esp_err_t reinit_err = init_i2s_channel();
                consecutive_i2s_errors = 0;
                if (reinit_err != ESP_OK) {
                    vTaskDelay(pdMS_TO_TICKS(100));
                }
            }
            continue;
        }

        consecutive_i2s_errors = 0;

        /* Convert 24-bit mic data in 32-bit slots to int16 (top 16 bits) */
        for (int i = 0; i < KAAWAYAN_SAMPLES_PER_CH; i++) {
            audio_payload[2 * i]     = (int16_t)(raw_slots[2 * i] >> 16);
            audio_payload[2 * i + 1] = (int16_t)(raw_slots[2 * i + 1] >> 16);
        }

#if defined(CONFIG_KAAWAYAN_RAMP_MODE)
        /* Overwrite with deterministic counter ramp */
        kp_ramp_fill(s_audio_seq, audio_payload);
#endif

        uint64_t t_us = (uint64_t)esp_timer_get_time();
        uint32_t current_seq = s_audio_seq++; /* Assign seq and always advance */

        size_t frame_len = kp_frame_encode(2, KAAWAYAN_NODE_ID, current_seq, t_us,
                                           (const uint8_t *)audio_payload,
                                           (uint16_t)sizeof(audio_payload),
                                           frame_buf, sizeof(frame_buf));

        if (frame_len == 0) {
            continue;
        }

        /* Push whole frame to FreeRTOS ring buffer. If full, drop frame and count overrun. */
        BaseType_t sent = xRingbufferSend(g_audio_ring, frame_buf, frame_len, 0);
        if (sent != pdTRUE) {
            g_overrun_count++;
        }
    }
}
