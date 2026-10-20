#ifndef BME280_H
#define BME280_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Calibration parameters read from BME280 NVM */
typedef struct {
    uint16_t dig_T1;
    int16_t  dig_T2;
    int16_t  dig_T3;
    uint8_t  dig_H1;
    int16_t  dig_H2;
    uint8_t  dig_H3;
    int16_t  dig_H4;
    int16_t  dig_H5;
    int8_t   dig_H6;
} bme280_calib_t;

/**
 * @brief Bosch integer compensation for BME280 / BMP280. Pure C, host-testable.
 *
 * @param adc_T Raw 20-bit temperature reading.
 * @param adc_H Raw 16-bit humidity reading.
 * @param calib Pointer to calibration structure.
 * @param is_bmp280 If true (chip id 0x58), humidity is forced to 0.
 * @param temp_c_x100 Output temperature in 0.01 °C (e.g. 2508 = 25.08 °C).
 * @param rh_x100 Output relative humidity in 0.01 %RH (e.g. 3631 = 36.31 %RH).
 */
void bme280_compensate(int32_t adc_T, int32_t adc_H, const bme280_calib_t *calib,
                       bool is_bmp280, int16_t *temp_c_x100, uint16_t *rh_x100);

#ifndef BME280_HOST_TEST
/**
 * @brief Initialize I2C bus and probe BME280 at address 0x76.
 */
void bme280_init(int sda_pin, int scl_pin);

/**
 * @brief Trigger forced mode measurement, read sensor, and compensate values.
 *
 * bme_status codes per CONTEXT §4:
 *   0: OK
 *   1: No response from BME280 (I2C NACK / missing sensor)
 *   2: Chip ID is not 0x60 (0x58 BMP280: temp valid, rh = 0; or unrecognized ID)
 *   3: Read error or timeout
 */
void bme280_read(int16_t *temp_c_x100, uint16_t *rh_x100, uint8_t *bme_status);
#endif

#ifdef __cplusplus
}
#endif

#endif /* BME280_H */
