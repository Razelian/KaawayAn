idf_component_register(SRCS "usb_out_jtag.c" "usb_out_tinyusb.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES driver freertos esp_ringbuf esp_hw_support esp_timer esp_tinyusb esp_driver_usb_serial_jtag)