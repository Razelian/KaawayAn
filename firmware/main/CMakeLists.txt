idf_component_register(SRCS "main.c" "capture.c" "usb_task.c" "health.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp_timer esp_system driver freertos esp_ringbuf esp_common kaawayan_proto usb_out bme280)
