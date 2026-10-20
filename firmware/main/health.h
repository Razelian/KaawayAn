#ifndef HEALTH_H
#define HEALTH_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize health monitoring (I2C for BME280, initial STARTUP frame).
 */
void health_init(void);

/**
 * @brief Periodic health task (sends ENV frame every 1s, STARTUP every 10s).
 */
void health_task(void *pvParameters);

#ifdef __cplusplus
}
#endif

#endif /* HEALTH_H */
