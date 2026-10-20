#ifndef KAAWAYAN_PROTO_H
#define KAAWAYAN_PROTO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * CRC-32 (IEEE 802.3, same as Python zlib.crc32)
 * Test vector: kp_crc32("123456789", 9) == 0xCBF43926
 * ----------------------------------------------------------------------- */
uint32_t kp_crc32(const uint8_t *data, size_t len);

/* -----------------------------------------------------------------------
 * Frame encoder
 * Returns total bytes written into out[], or 0 on error.
 * Errors: out_cap too small, payload len > 4096.
 * ----------------------------------------------------------------------- */
size_t kp_frame_encode(uint8_t type, uint16_t node_id, uint32_t seq,
                       uint64_t t_us, const uint8_t *payload, uint16_t len,
                       uint8_t *out, size_t out_cap);

/* -----------------------------------------------------------------------
 * Payload builders – caller provides correctly-sized buffer:
 *   STARTUP : 16 bytes  (<BBBBIIHBB>)
 *   ENV     : 13 bytes  (<hHIIB>)
 * ----------------------------------------------------------------------- */
void kp_startup_payload(uint8_t board_type, uint8_t fw_major, uint8_t fw_minor,
                        uint8_t fw_patch, uint32_t git_short,
                        uint32_t sample_rate_hz, uint16_t samples_per_ch,
                        uint8_t channels, uint8_t mic_flags, uint8_t *out);

void kp_env_payload(int16_t temp_c_x100, uint16_t rh_x100,
                    uint32_t overrun_count, uint32_t usb_drop_count,
                    uint8_t bme_status, uint8_t *out);

/* -----------------------------------------------------------------------
 * Ramp test-mode audio fill
 * Fills stereo512[0..1023] with interleaved L,R pairs for AUDIO frame seq.
 * n = seq*512 + i;  L = (int16_t)(uint16_t)(n & 0xFFFF);  R = ~L
 * ----------------------------------------------------------------------- */
void kp_ramp_fill(uint32_t seq, int16_t *stereo512);

#ifdef __cplusplus
}
#endif

#endif /* KAAWAYAN_PROTO_H */
