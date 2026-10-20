#ifndef CAPTURE_H
#define CAPTURE_H

#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

extern volatile uint32_t g_overrun_count;
extern volatile uint8_t  g_mic_flags;
extern RingbufHandle_t   g_audio_ring;

/**
 * @brief Initialize I2S peripheral and test microphones at boot.
 */
void capture_init(void);

/**
 * @brief FreeRTOS task entry point for I2S audio capture.
 */
void capture_task(void *pvParameters);

#ifdef __cplusplus
}
#endif

#endif /* CAPTURE_H */
