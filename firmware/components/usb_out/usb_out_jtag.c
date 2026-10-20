#include "usb_out.h"
#include "sdkconfig.h"

#if !defined(CONFIG_KAAWAYAN_USB_BACKEND_TINYUSB)

#include "driver/usb_serial_jtag.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void usb_out_init(void)
{
    usb_serial_jtag_driver_config_t jtag_config = {
        .tx_buffer_size = 4096,
        .rx_buffer_size = 512,
    };
    usb_serial_jtag_driver_install(&jtag_config);
}

size_t usb_out_write(const uint8_t *p, size_t n, uint32_t timeout_ms)
{
    if (p == NULL || n == 0) {
        return 0;
    }
    int written = usb_serial_jtag_write_bytes(p, n, pdMS_TO_TICKS(timeout_ms));
    if (written < 0) {
        return 0;
    }
    return (size_t)written;
}

#endif /* !CONFIG_KAAWAYAN_USB_BACKEND_TINYUSB */
