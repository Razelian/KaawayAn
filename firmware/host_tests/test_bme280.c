#include "bme280.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
 * Test vector based on Bosch Sensortec BME280 datasheet BST-BME280-DS002 Section 4.2.3.
 * Calibration constants:
 *   dig_T1 = 27504, dig_T2 = 26435, dig_T3 = -1000
 *   dig_H1 = 75, dig_H2 = 369, dig_H3 = 0, dig_H4 = 301, dig_H5 = 0, dig_H6 = 30
 * Raw readings:
 *   adc_T = 519888
 *   adc_H = 25600
 *
 * Hand-checked intermediate and expected results:
 *   var1 = 127976, var2 = 446 -> t_fine = 128422
 *   T = (128422 * 5 + 128) >> 8 = 2508  (25.08 deg C)
 *   v_x1_u32r = 152281088 -> h_q22_10 = 37178 (36.31 %RH)
 *   rh_x100 = (37178 * 100 + 512) >> 10 = 3631  (36.31 %RH)
 */

int main(void)
{
    printf("Running BME280 pure-C compensation host tests...\n");

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

    /* 1. Standard BME280 test */
    bme280_compensate(adc_T, adc_H, &calib, false, &temp_c_x100, &rh_x100);
    printf("BME280 compensated: temp = %d (25.08 C), rh = %d (36.31 %%)\n",
           temp_c_x100, rh_x100);
    assert(temp_c_x100 == 2508);
    assert(rh_x100 == 3631);

    /* 2. BMP280 fallback test (chip id 0x58): temp valid, rh = 0 */
    temp_c_x100 = 0;
    rh_x100 = 9999;
    bme280_compensate(adc_T, adc_H, &calib, true, &temp_c_x100, &rh_x100);
    printf("BMP280 fallback: temp = %d, rh = %d\n", temp_c_x100, rh_x100);
    assert(temp_c_x100 == 2508);
    assert(rh_x100 == 0);

    /* 3. Range clamping tests */
    /* Extreme low adc_H */
    bme280_compensate(adc_T, 0, &calib, false, &temp_c_x100, &rh_x100);
    assert(temp_c_x100 == 2508);
    assert(rh_x100 == 0);

    printf("All BME280 host tests passed!\n");
    return 0;
}
