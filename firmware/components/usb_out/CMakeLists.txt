set(srcs "usb_out_jtag.c" "usb_out_tinyusb.c")
set(reqs "driver" "esp_hw_support")

if(CONFIG_KAAWAYAN_USB_BACKEND_TINYUSB)
    list(APPEND reqs "esp_tinyusb")
endif()

idf_component_register(SRCS ${srcs}
                       INCLUDE_DIRS "include" "."
                       REQUIRES ${reqs})
