#include "usb_out.h"
#include "sdkconfig.h"

#if defined(CONFIG_KAAWAYAN_USB_BACKEND_TINYUSB)

#include <stdio.h>
#include <string.h>
#include "tinyusb.h"
#include "tusb_cdc_acm.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define TAG "USB_TINYUSB"

static char s_serial_str[13];
static char s_lang[] = {0x09, 0x04};
static const char *s_strings[] = {
    s_lang,
    "KaawayAn",
    "KaawayAn Node",
    s_serial_str,
    "KaawayAn CDC",
};

void usb_out_init(void)
{
    /* 1. Retrieve eFuse MAC to create unique serial number string */
    uint8_t mac[6] = {0};
    esp_efuse_mac_get_default(mac);
    snprintf(s_serial_str, sizeof(s_serial_str), "%02X%02X%02X%02X%02X%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    ESP_LOGI(TAG, "Initializing TinyUSB CDC with unique eFuse MAC serial: %s", s_serial_str);

    const tinyusb_config_t tusb_cfg = {
        .device_descriptor = NULL,
        .string_descriptor = s_strings,
        .string_descriptor_count = sizeof(s_strings) / sizeof(s_strings[0]),
        .external_phy = false,
        .configuration_descriptor = NULL,
    };

    ESP_ERROR_CHECK(tinyusb_driver_install(&tusb_cfg));

    tinyusb_config_cdcacm_t acm_cfg = {
        .usb_dev = TINYUSB_USBDEV_0,
        .cdc_port = TINYUSB_CDC_ACM_0,
        .rx_unread_buf_sz = 512,
        .callback_rx = NULL,
        .callback_rx_wanted_char = NULL,
        .callback_line_state_changed = NULL,
        .callback_line_coding_changed = NULL,
    };

    ESP_ERROR_CHECK(tusb_cdc_acm_init(&acm_cfg));
    ESP_LOGI(TAG, "TinyUSB initialized successfully.");
}

size_t usb_out_write(const uint8_t *p, size_t n, uint32_t timeout_ms)
{
    if (p == NULL || n == 0) {
        return 0;
    }

    if (!tusb_cdc_acm_initialized(TINYUSB_CDC_ACM_0)) {
        return 0;
    }

    int64_t deadline = esp_timer_get_time() + ((int64_t)timeout_ms * 1000LL);
    size_t sent = 0;

    while (sent < n) {
        size_t q = tinyusb_cdcacm_write_queue(TINYUSB_CDC_ACM_0, p + sent, n - sent);
        sent += q;
        if (sent >= n) {
            break;
        }
        if (esp_timer_get_time() >= deadline) {
            break;
        }
        tinyusb_cdcacm_write_flush(TINYUSB_CDC_ACM_0, pdMS_TO_TICKS(2)); /* ticks, not ms */
    }

    if (sent > 0) {
        tinyusb_cdcacm_write_flush(TINYUSB_CDC_ACM_0, pdMS_TO_TICKS(2));
    }

    return sent;
}

#endif /* CONFIG_KAAWAYAN_USB_BACKEND_TINYUSB */
