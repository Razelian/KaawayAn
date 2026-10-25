idf_component_register(SRCS "main.c" "capture.c" "health.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES driver freertos esp_ringbuf esp_driver_i2s esp_driver_gpio esp_timer esp_hw_support kaawayan_proto usb_out bme280)