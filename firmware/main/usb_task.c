#include "usb_task.h"
#include "usb_out.h"
#include "capture.h"
#include "esp_log.h"
#if defined(CONFIG_ESP_TASK_WDT)
#include "esp_task_wdt.h"
#endif

#define TAG "USB_TASK"

volatile uint32_t g_usb_drop_count = 0;
QueueHandle_t     g_ctrl_queue = NULL;

void usb_task_init(void)
{
    g_ctrl_queue = xQueueCreate(8, sizeof(ctrl_frame_t));
    if (g_ctrl_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create control frame queue");
    }

    usb_out_init();
}

void usb_worker_task(void *pvParameters)
{
    (void)pvParameters;

#if defined(CONFIG_ESP_TASK_WDT)
    esp_task_wdt_add(NULL);
#endif

    ctrl_frame_t ctrl;

    while (1) {
#if defined(CONFIG_ESP_TASK_WDT)
        esp_task_wdt_reset();
#endif

        /* 1. Prioritize control frames (STARTUP, ENV) */
        if (xQueueReceive(g_ctrl_queue, &ctrl, 0) == pdTRUE) {
            size_t written = usb_out_write(ctrl.data, ctrl.len, 50);
            if (written < ctrl.len) {
                g_usb_drop_count++;
            }
            continue;
        }

        /* 2. Audio frames from ring buffer */
        if (g_audio_ring == NULL) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        size_t item_size = 0;
        uint8_t *item = (uint8_t *)xRingbufferReceive(g_audio_ring, &item_size, pdMS_TO_TICKS(10));
        if (item != NULL) {
            size_t written = usb_out_write(item, item_size, 50);
            if (written < item_size) {
                /* If host does not drain or write times out, drop whole frame */
                g_usb_drop_count++;
            }
            vRingbufferReturnItem(g_audio_ring, item);
        }
    }
}
