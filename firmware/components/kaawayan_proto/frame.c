/*
 * frame.c – KaawayAn wire-protocol frame encoder (host-portable, no ESP-IDF)
 *
 * Frame layout (little-endian):
 *   [0]    = 0xBA  (sync[0])
 *   [1]    = 0xAB  (sync[1])
 *   [2]    = 0x01  (version)
 *   [3]    = type
 *   [4-5]  = node_id  (LE uint16)
 *   [6-9]  = seq      (LE uint32)
 *   [10-17]= t_us     (LE uint64)
 *   [18-19]= len      (LE uint16)
 *   [20 .. 20+len-1] = payload
 *   [20+len .. 20+len+3] = CRC-32 LE  (over bytes[2..20+len-1])
 */
#include "kaawayan_proto.h"
#include <string.h>

#define KP_SYNC0   0xBAu
#define KP_SYNC1   0xABu
#define KP_VERSION 0x01u
#define KP_HDR_LEN 20u
#define KP_CRC_LEN 4u
#define KP_MAX_PAYLOAD 4096u

size_t kp_frame_encode(uint8_t type, uint16_t node_id, uint32_t seq,
                       uint64_t t_us, const uint8_t *payload, uint16_t len,
                       uint8_t *out, size_t out_cap)
{
    if ((size_t)len > KP_MAX_PAYLOAD)
        return 0;

    size_t total = KP_HDR_LEN + len + KP_CRC_LEN;
    if (out_cap < total)
        return 0;

    /* --- header --- */
    out[0] = KP_SYNC0;
    out[1] = KP_SYNC1;
    out[2] = KP_VERSION;
    out[3] = type;
    out[4] = (uint8_t)(node_id & 0xFF);
    out[5] = (uint8_t)(node_id >> 8);
    out[6]  = (uint8_t)(seq & 0xFF);
    out[7]  = (uint8_t)((seq >> 8)  & 0xFF);
    out[8]  = (uint8_t)((seq >> 16) & 0xFF);
    out[9]  = (uint8_t)((seq >> 24) & 0xFF);
    out[10] = (uint8_t)(t_us & 0xFF);
    out[11] = (uint8_t)((t_us >> 8)  & 0xFF);
    out[12] = (uint8_t)((t_us >> 16) & 0xFF);
    out[13] = (uint8_t)((t_us >> 24) & 0xFF);
    out[14] = (uint8_t)((t_us >> 32) & 0xFF);
    out[15] = (uint8_t)((t_us >> 40) & 0xFF);
    out[16] = (uint8_t)((t_us >> 48) & 0xFF);
    out[17] = (uint8_t)((t_us >> 56) & 0xFF);
    out[18] = (uint8_t)(len & 0xFF);
    out[19] = (uint8_t)(len >> 8);

    /* --- payload --- */
    if (len > 0 && payload != NULL)
        memcpy(out + KP_HDR_LEN, payload, len);

    /* --- CRC over bytes[2 .. 20+len-1] --- */
    uint32_t crc = kp_crc32(out + 2, KP_HDR_LEN - 2 + len);
    out[KP_HDR_LEN + len + 0] = (uint8_t)(crc & 0xFF);
    out[KP_HDR_LEN + len + 1] = (uint8_t)((crc >> 8)  & 0xFF);
    out[KP_HDR_LEN + len + 2] = (uint8_t)((crc >> 16) & 0xFF);
    out[KP_HDR_LEN + len + 3] = (uint8_t)((crc >> 24) & 0xFF);

    return total;
}

/* -----------------------------------------------------------------------
 * Payload builders
 * ----------------------------------------------------------------------- */

/* STARTUP – <BBBBIIHBB> = 16 bytes */
void kp_startup_payload(uint8_t board_type, uint8_t fw_major, uint8_t fw_minor,
                        uint8_t fw_patch, uint32_t git_short,
                        uint32_t sample_rate_hz, uint16_t samples_per_ch,
                        uint8_t channels, uint8_t mic_flags, uint8_t *out)
{
    out[0] = board_type;
    out[1] = fw_major;
    out[2] = fw_minor;
    out[3] = fw_patch;
    out[4] = (uint8_t)(git_short & 0xFF);
    out[5] = (uint8_t)((git_short >> 8)  & 0xFF);
    out[6] = (uint8_t)((git_short >> 16) & 0xFF);
    out[7] = (uint8_t)((git_short >> 24) & 0xFF);
    out[8]  = (uint8_t)(sample_rate_hz & 0xFF);
    out[9]  = (uint8_t)((sample_rate_hz >> 8)  & 0xFF);
    out[10] = (uint8_t)((sample_rate_hz >> 16) & 0xFF);
    out[11] = (uint8_t)((sample_rate_hz >> 24) & 0xFF);
    out[12] = (uint8_t)(samples_per_ch & 0xFF);
    out[13] = (uint8_t)(samples_per_ch >> 8);
    out[14] = channels;
    out[15] = mic_flags;
}

/* ENV – <hHIIB> = 13 bytes */
void kp_env_payload(int16_t temp_c_x100, uint16_t rh_x100,
                    uint32_t overrun_count, uint32_t usb_drop_count,
                    uint8_t bme_status, uint8_t *out)
{
    /* temp_c_x100 signed i16 */
    out[0] = (uint8_t)((uint16_t)temp_c_x100 & 0xFF);
    out[1] = (uint8_t)((uint16_t)temp_c_x100 >> 8);
    /* rh_x100 u16 */
    out[2] = (uint8_t)(rh_x100 & 0xFF);
    out[3] = (uint8_t)(rh_x100 >> 8);
    /* overrun_count u32 */
    out[4] = (uint8_t)(overrun_count & 0xFF);
    out[5] = (uint8_t)((overrun_count >> 8)  & 0xFF);
    out[6] = (uint8_t)((overrun_count >> 16) & 0xFF);
    out[7] = (uint8_t)((overrun_count >> 24) & 0xFF);
    /* usb_drop_count u32 */
    out[8]  = (uint8_t)(usb_drop_count & 0xFF);
    out[9]  = (uint8_t)((usb_drop_count >> 8)  & 0xFF);
    out[10] = (uint8_t)((usb_drop_count >> 16) & 0xFF);
    out[11] = (uint8_t)((usb_drop_count >> 24) & 0xFF);
    /* bme_status u8 */
    out[12] = bme_status;
}
