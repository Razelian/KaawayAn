#include "bme280.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
 * Note on test vectors:
 * - The TEMPERATURE vector (adc_T=519888 -> 2508) is the Bosch datasheet example (BST-BME280-DS002 Section 4.2.3).
 * - The HUMIDITY vector (adc_H=25600 -> 3631) is a regression value computed by this code, NOT from Bosch.
 */

int main(void)
{
    printf("Running BME280 pure-C host tests...\n");

    /* 1. Calibration parsing test (step FX1.4) */
    const uint8_t raw_t[6] = {0x70, 0x6B, 0x43, 0x67, 0x18, 0xFC};
    const uint8_t raw_a1 = 75;
    const uint8_t raw_h[7] = {0x71, 0x01, 0x00, 0xF0, 0x1F, 0x80, 0x1E};

    bme280_calib_t parsed;
    bme280_parse_calib(raw_t, raw_a1, raw_h, &parsed);

    assert(parsed.dig_T1 == 27504);
    assert(parsed.dig_T2 == 26435);
    assert(parsed.dig_T3 == -1000);
    assert(parsed.dig_H1 == 75);
    assert(parsed.dig_H2 == 369);
    assert(parsed.dig_H3 == 0);
    assert(parsed.dig_H4 == -241);
    assert(parsed.dig_H5 == -2047);
    assert(parsed.dig_H6 == 30);

    /* 2. Existing compensation tests */
    bme280_calib_t calib = {
        .dig_T1 = 27504,
        .dig_T2 = 26435,
        .dig_T3 = -1000,
        .dig_H1 = 75,
        .dig_H2 = 369,
        .dig_H3 = 0,
        .dig_H4 = 301,
        .dig_H5 = 0,
        .dig_H6 = 30,
    };

    int32_t adc_T = 519888;
    int32_t adc_H = 25600;

    int16_t temp_c_x100 = 0;
    uint16_t rh_x100 = 0;

    bme280_compensate(adc_T, adc_H, &calib, false, &temp_c_x100, &rh_x100);
    printf("BME280 compensated: temp = %d (25.08 C), rh = %d (36.31 %%)\n",
           temp_c_x100, rh_x100);
    assert(temp_c_x100 == 2508);
    assert(rh_x100 == 3631);

    /* 3. BMP280 fallback test (chip id 0x58): temp valid, rh = 0 */
    temp_c_x100 = 0;
    rh_x100 = 9999;
    bme280_compensate(adc_T, adc_H, &calib, true, &temp_c_x100, &rh_x100);
    printf("BMP280 fallback: temp = %d, rh = %d\n", temp_c_x100, rh_x100);
    assert(temp_c_x100 == 2508);
    assert(rh_x100 == 0);

    /* 4. Range clamping test */
    bme280_compensate(adc_T, 0, &calib, false, &temp_c_x100, &rh_x100);
    assert(temp_c_x100 == 2508);
    assert(rh_x100 == 0);

    printf("All BME280 host tests passed!\n");
    return 0;
}
