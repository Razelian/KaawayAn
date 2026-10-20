#ifndef USB_TASK_H
#define USB_TASK_H

#include <stdint.h>
#include <stddef.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t data[64];
    size_t  len;
} ctrl_frame_t;

extern volatile uint32_t g_usb_drop_count;
extern QueueHandle_t     g_ctrl_queue;

/**
 * @brief Initialize USB task and its queue.
 */
void usb_task_init(void);

/**
 * @brief FreeRTOS task entry point for USB transmission.
 */
void usb_worker_task(void *pvParameters);

#ifdef __cplusplus
}
#endif

#endif /* USB_TASK_H */
