#include "bme280.h"

/* -------------------------------------------------------------------------
 * Pure-C Bosch integer compensation (BST-BME280-DS002 Section 4.2.3)
 * ------------------------------------------------------------------------- */
void bme280_compensate(int32_t adc_T, int32_t adc_H, const bme280_calib_t *calib,
                       bool is_bmp280, int16_t *temp_c_x100, uint16_t *rh_x100)
{
    if (calib == NULL || temp_c_x100 == NULL || rh_x100 == NULL) {
        return;
    }

    /* 1. Temperature compensation (calculates t_fine) */
    int32_t var1 = ((((adc_T >> 3) - ((int32_t)calib->dig_T1 << 1))) * ((int32_t)calib->dig_T2)) >> 11;
    int32_t var2 = (((((adc_T >> 4) - ((int32_t)calib->dig_T1)) *
                      ((adc_T >> 4) - ((int32_t)calib->dig_T1))) >> 12) *
                    ((int32_t)calib->dig_T3)) >> 14;

    int32_t t_fine = var1 + var2;
    int32_t T = (t_fine * 5 + 128) >> 8;
    *temp_c_x100 = (int16_t)T;

    /* If sensor is BMP280 (chip id 0x58), humidity is absent */
    if (is_bmp280) {
        *rh_x100 = 0;
        return;
    }

    /* 2. Humidity compensation (uses t_fine) */
    int32_t v_x1_u32r = (t_fine - ((int32_t)76800));

    v_x1_u32r = (((((adc_H << 14) - (((int32_t)calib->dig_H4) << 20) -
                    (((int32_t)calib->dig_H5) * v_x1_u32r)) + ((int32_t)16384)) >> 15) *
                 (((((((v_x1_u32r * ((int32_t)calib->dig_H6)) >> 10) *
                     (((v_x1_u32r * ((int32_t)calib->dig_H3)) >> 11) + ((int32_t)32768))) >> 10) +
                    ((int32_t)2097152)) * ((int32_t)calib->dig_H2) + 8192) >> 14));

    v_x1_u32r = (v_x1_u32r - (((((v_x1_u32r >> 15) * (v_x1_u32r >> 15)) >> 7) *
                               ((int32_t)calib->dig_H1)) >> 4));

    if (v_x1_u32r < 0) {
        v_x1_u32r = 0;
    } else if (v_x1_u32r > 419430400) {
        v_x1_u32r = 419430400;
    }

    uint32_t h_q22_10 = (uint32_t)(v_x1_u32r >> 12);
    /* Convert from Q22.10 (val / 1024 %RH) to hundredths of a percent (%RH x 100) */
    uint32_t rh = (h_q22_10 * 100u + 512u) >> 10;
    if (rh > 10000u) {
        rh = 10000u;
    }
    *rh_x100 = (uint16_t)rh;
}

#ifndef BME280_HOST_TEST

#include "driver/i2c.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define TAG "BME280"
#define BME280_I2C_ADDR 0x76
#define I2C_PORT        I2C_NUM_0

static bme280_calib_t s_calib;
static bool s_is_bmp280 = false;
static uint8_t s_status = 1; /* Default: no response */
static bool s_initialized = false;

static esp_err_t i2c_read_reg(uint8_t reg, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_PORT, BME280_I2C_ADDR, &reg, 1,
                                        data, len, pdMS_TO_TICKS(20));
}

static esp_err_t i2c_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = {reg, val};
    return i2c_master_write_to_device(I2C_PORT, BME280_I2C_ADDR, buf, 2,
                                      pdMS_TO_TICKS(20));
}

void bme280_init(int sda_pin, int scl_pin)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = sda_pin,
        .scl_io_num = scl_pin,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 100000,
    };
    i2c_param_config(I2C_PORT, &conf);
    i2c_driver_install(I2C_PORT, conf.mode, 0, 0, 0);

    /* Read chip ID at register 0xD0 */
    uint8_t chip_id = 0;
    esp_err_t err = i2c_read_reg(0xD0, &chip_id, 1);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "No response from BME280 at 0x76 (err %d)", err);
        s_status = 1;
        return;
    }

    if (chip_id == 0x60) {
        ESP_LOGI(TAG, "Found BME280 (chip ID 0x60)");
        s_status = 0;
        s_is_bmp280 = false;
    } else if (chip_id == 0x58) {
        ESP_LOGW(TAG, "Found BMP280 (chip ID 0x58, temperature only)");
        s_status = 2;
        s_is_bmp280 = true;
    } else {
        ESP_LOGW(TAG, "Unknown sensor chip ID 0x%02X", chip_id);
        s_status = 2;
        return;
    }

    /* Read temperature calibration parameters (0x88..0x8D) */
    uint8_t cal_t[6];
    if (i2c_read_reg(0x88, cal_t, 6) != ESP_OK) {
        s_status = 3;
        return;
    }
    s_calib.dig_T1 = (uint16_t)(cal_t[0] | (cal_t[1] << 8));
    s_calib.dig_T2 = (int16_t)(cal_t[2] | (cal_t[3] << 8));
    s_calib.dig_T3 = (int16_t)(cal_t[4] | (cal_t[5] << 8));

    if (!s_is_bmp280) {
        /* Read humidity calibration parameters: 0xA1, 0xE1..0xE7 */
        uint8_t h1 = 0;
        uint8_t cal_h[7];
        if (i2c_read_reg(0xA1, &h1, 1) != ESP_OK ||
            i2c_read_reg(0xE1, cal_h, 7) != ESP_OK) {
            s_status = 3;
            return;
        }
        s_calib.dig_H1 = h1;
        s_calib.dig_H2 = (int16_t)(cal_h[0] | (cal_h[1] << 8));
        s_calib.dig_H3 = cal_h[2];
        s_calib.dig_H4 = (int16_t)((cal_h[3] << 4) | (cal_h[4] & 0x0F));
        s_calib.dig_H5 = (int16_t)((cal_h[5] << 4) | (cal_h[4] >> 4));
        s_calib.dig_H6 = (int8_t)cal_h[6];
    }

    s_initialized = true;
}

void bme280_read(int16_t *temp_c_x100, uint16_t *rh_x100, uint8_t *bme_status)
{
    *temp_c_x100 = 0;
    *rh_x100 = 0;
    *bme_status = s_status;

    if (!s_initialized || s_status == 1) {
        /* If sensor failed probe, keep status 1 (or 2) and return 0s quickly */
        return;
    }

    /* 1. Trigger forced mode measurement:
     *    ctrl_hum (0xF2) = 0x01 (osrs_h = x1)
     *    ctrl_meas (0xF4) = 0x25 (osrs_t = x1, osrs_p = 0, mode = 01 forced)
     */
    if (!s_is_bmp280) {
        i2c_write_reg(0xF2, 0x01);
    }
    if (i2c_write_reg(0xF4, 0x21) != ESP_OK) { /* 0x21: osrs_t=x1, osrs_p=skip, forced mode */
        *bme_status = 3;
        return;
    }

    /* Measurement takes ~8 ms; delay 10 ms */
    vTaskDelay(pdMS_TO_TICKS(10));

    /* Read raw data registers: 0xFA..0xFC (temperature), 0xFD..0xFE (humidity) */
    uint8_t data[5];
    size_t len = s_is_bmp280 ? 3 : 5;
    if (i2c_read_reg(0xFA, data, len) != ESP_OK) {
        *bme_status = 3;
        return;
    }

    int32_t adc_T = (int32_t)(((uint32_t)data[0] << 12) |
                              ((uint32_t)data[1] << 4) |
                              ((uint32_t)data[2] >> 4));
    int32_t adc_H = 0;
    if (!s_is_bmp280) {
        adc_H = (int32_t)(((uint32_t)data[3] << 8) | (uint32_t)data[4]);
    }

    bme280_compensate(adc_T, adc_H, &s_calib, s_is_bmp280, temp_c_x100, rh_x100);
    *bme_status = s_is_bmp280 ? 2 : 0;
}

#endif /* !BME280_HOST_TEST */
