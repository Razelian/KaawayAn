#include "board_config.h"
#include "capture.h"
#include "usb_task.h"
#include "health.h"
#include "esp_log.h"
#if defined(CONFIG_ESP_TASK_WDT)
#include "esp_task_wdt.h"
#endif

#define TAG "MAIN"

void app_main(void)
{
    ESP_LOGI(TAG, "Starting KaawayAn Node ID %d (Board Type %d, FW %d.%d.%d)...",
             KAAWAYAN_NODE_ID, KAAWAYAN_BOARD_TYPE,
             KAAWAYAN_FW_MAJOR, KAAWAYAN_FW_MINOR, KAAWAYAN_FW_PATCH);

#if defined(CONFIG_ESP_TASK_WDT)
    esp_task_wdt_config_t twdt_config = {
        .timeout_ms = 3000,
        .idle_core_mask = 0,
        .trigger_panic = true,
    };
    esp_task_wdt_init(&twdt_config);
#endif

    /* 1. Initialize USB output and control queue */
    usb_task_init();

    /* 2. Initialize I2S capture and test mic health */
    capture_init();

    /* 3. Initialize health monitoring and send boot STARTUP frame */
    health_init();

    /* 4. Launch FreeRTOS tasks pinned to dedicated cores:
     *    - Capture task: Core 0, high priority
     *    - USB task:     Core 1, high priority
     *    - Health task:  Core 0, normal priority
     */
    xTaskCreatePinnedToCore(capture_task, "capture_task", 4096, NULL,
                           configMAX_PRIORITIES - 2, NULL, 0);

    xTaskCreatePinnedToCore(usb_worker_task, "usb_worker_task", 4096, NULL,
                           configMAX_PRIORITIES - 3, NULL, 1);

    xTaskCreatePinnedToCore(health_task, "health_task", 3072, NULL,
                           5, NULL, 0);

    ESP_LOGI(TAG, "KaawayAn tasks launched successfully.");
}
