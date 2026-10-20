#include "health.h"
#include "board_config.h"
#include "kaawayan_proto.h"
#include "capture.h"
#include "usb_task.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#if defined(CONFIG_ESP_TASK_WDT)
#include "esp_task_wdt.h"
#endif

#if __has_include("bme280.h")
#include "bme280.h"
#define HAVE_BME280 1
#else
#define HAVE_BME280 0
#endif

#define TAG "HEALTH"

static uint32_t s_startup_seq = 0;
static uint32_t s_env_seq = 0;

static void send_startup_frame(void)
{
    ctrl_frame_t frame;
    uint8_t payload[16];

    kp_startup_payload(KAAWAYAN_BOARD_TYPE,
                       KAAWAYAN_FW_MAJOR,
                       KAAWAYAN_FW_MINOR,
                       KAAWAYAN_FW_PATCH,
                       KAAWAYAN_GIT_SHORT,
                       KAAWAYAN_SAMPLE_RATE_HZ,
                       KAAWAYAN_SAMPLES_PER_CH,
                       KAAWAYAN_CHANNELS,
                       g_mic_flags,
                       payload);

    uint64_t t_us = (uint64_t)esp_timer_get_time();
    size_t sz = kp_frame_encode(1, KAAWAYAN_NODE_ID, s_startup_seq++, t_us,
                                payload, sizeof(payload), frame.data, sizeof(frame.data));
    frame.len = sz;
    if (sz > 0 && g_ctrl_queue != NULL) {
        xQueueSend(g_ctrl_queue, &frame, 0);
    }
}

static void send_env_frame(void)
{
    ctrl_frame_t frame;
    uint8_t payload[13];

    int16_t temp_c_x100 = 0;
    uint16_t rh_x100 = 0;
    uint8_t bme_status = 1; /* 1 = no response / NACK */

#if HAVE_BME280
    bme280_read(&temp_c_x100, &rh_x100, &bme_status);
#endif

    kp_env_payload(temp_c_x100, rh_x100, g_overrun_count, g_usb_drop_count, bme_status, payload);

    uint64_t t_us = (uint64_t)esp_timer_get_time();
    size_t sz = kp_frame_encode(3, KAAWAYAN_NODE_ID, s_env_seq++, t_us,
                                payload, sizeof(payload), frame.data, sizeof(frame.data));
    frame.len = sz;
    if (sz > 0 && g_ctrl_queue != NULL) {
        xQueueSend(g_ctrl_queue, &frame, 0);
    }
}

void health_init(void)
{
#if HAVE_BME280
    bme280_init(KAAWAYAN_PIN_I2C_SDA, KAAWAYAN_PIN_I2C_SCL);
#endif

    /* Emit initial STARTUP frame immediately at boot */
    send_startup_frame();
}

void health_task(void *pvParameters)
{
    (void)pvParameters;

#if defined(CONFIG_ESP_TASK_WDT)
    esp_task_wdt_add(NULL);
#endif

    uint32_t ticks_sec = 0;

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        ticks_sec++;

#if defined(CONFIG_ESP_TASK_WDT)
        esp_task_wdt_reset();
#endif

        /* Emit ENV frame every 1 second */
        send_env_frame();

        /* Emit STARTUP frame every 10 seconds */
        if (ticks_sec >= 10) {
            ticks_sec = 0;
            send_startup_frame();
        }
    }
}
