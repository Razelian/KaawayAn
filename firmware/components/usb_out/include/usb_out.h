#ifndef USB_OUT_H
#define USB_OUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the selected USB backend (Serial/JTAG or TinyUSB).
 */
void usb_out_init(void);

/**
 * @brief Write bytes to USB output.
 * 
 * @param p Pointer to byte buffer.
 * @param n Number of bytes to write.
 * @param timeout_ms Timeout in milliseconds.
 * @return size_t Number of bytes actually written.
 */
size_t usb_out_write(const uint8_t *p, size_t n, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* USB_OUT_H */
