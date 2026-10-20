#include "kaawayan_proto.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int hex_val(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static size_t parse_hex(const char *hex_str, uint8_t *out, size_t out_cap)
{
    size_t len = 0;
    while (*hex_str && *(hex_str + 1)) {
        while (*hex_str == ' ' || *hex_str == '\r' || *hex_str == '\n') {
            hex_str++;
        }
        if (!*hex_str) break;
        int hi = hex_val(*hex_str++);
        int lo = hex_val(*hex_str++);
        if (hi < 0 || lo < 0) break;
        if (len >= out_cap) return 0;
        out[len++] = (uint8_t)((hi << 4) | lo);
    }
    return len;
}

int main(void)
{
    printf("Running protocol core host tests...\n");

    /* 1. CRC test vector */
    const char *vector = "123456789";
    uint32_t crc = kp_crc32((const uint8_t *)vector, 9);
    printf("CRC vector result: 0x%08X (expected 0xCBF43926)\n", crc);
    assert(crc == 0xCBF43926u);

    /* 2. Negative temperature ENV payload test (step FX1.5d) */
    uint8_t neg_env[KP_ENV_PAYLOAD_LEN];
    kp_env_payload(-1234, 6543, 7, 9, 2, neg_env);
    const uint8_t expected_neg[KP_ENV_PAYLOAD_LEN] = {
        0x2E, 0xFB, 0x8F, 0x19, 0x07, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x02
    };
    assert(memcmp(neg_env, expected_neg, KP_ENV_PAYLOAD_LEN) == 0);

    /* 3. Ramp test (basic, wrap, non-zero, and sequence limit) */
    int16_t stereo[1024];
    kp_ramp_fill(0, stereo);
    assert(stereo[0] == 0);
    assert(stereo[1] == -1);

    kp_ramp_fill(127, stereo);
    assert(stereo[1022] == -1);
    assert(stereo[1023] == 0);

    kp_ramp_fill(128, stereo);
    assert(stereo[0] == 0);
    assert(stereo[1] == -1);

    for (uint32_t s = 0; s <= 300; s++) {
        kp_ramp_fill(s, stereo);
        for (int i = 0; i < 512; i++) {
            int16_t l = stereo[2 * i];
            int16_t r = stereo[2 * i + 1];
            assert(!(l == 0 && r == 0));
        }
    }

    /* Ramp at the sequence limit (step FX1.5e) */
    int16_t ramp_lim[1024];
    kp_ramp_fill(0xFFFFFFFFu, ramp_lim);
    assert(ramp_lim[0] == -512);
    assert(ramp_lim[1] == 511);
    assert(ramp_lim[1022] == -1);
    assert(ramp_lim[1023] == 0);

    /* 4. Encode with len 4097 fails */
    uint8_t big_payload[4097];
    uint8_t frame_buf[5000];
    size_t enc_sz = kp_frame_encode(1, 1, 0, 0, big_payload, 4097, frame_buf, sizeof(frame_buf));
    assert(enc_sz == 0);

    /* 5. Golden frames from laptop/tests/golden/frames.hex */
    const char *paths[] = {
        "laptop/tests/golden/frames.hex",
        "../../laptop/tests/golden/frames.hex",
        "../laptop/tests/golden/frames.hex",
        NULL
    };
    FILE *f = NULL;
    for (int i = 0; paths[i] != NULL; i++) {
        f = fopen(paths[i], "r");
        if (f) break;
    }
    assert(f != NULL);

    char line[16384];
    int golden_count = 0;
    while (fgets(line, sizeof(line), f)) {
        char *colon = strchr(line, ':');
        if (!colon) continue;
        colon++; /* start of hex */

        uint8_t raw[8192];
        size_t raw_len = parse_hex(colon, raw, sizeof(raw));
        assert(raw_len >= 24);

        uint8_t type = raw[3];
        uint16_t node_id = (uint16_t)((uint16_t)raw[4] | ((uint16_t)raw[5] << 8));
        uint32_t seq = (uint32_t)raw[6] |
                       ((uint32_t)raw[7] << 8) |
                       ((uint32_t)raw[8] << 16) |
                       ((uint32_t)raw[9] << 24);
        uint64_t t_us = 0;
        for (int b = 0; b < 8; b++) {
            t_us |= ((uint64_t)raw[10 + b]) << (8 * b);
        }
        uint16_t payload_len = (uint16_t)((uint16_t)raw[18] | ((uint16_t)raw[19] << 8));
        assert(raw_len == (size_t)(20u + payload_len + 4u));

        const uint8_t *payload = raw + 20;

        /* Verify payload builders for STARTUP and ENV golden lines (step FX1.5c) */
        if (type == 1) { /* STARTUP */
            assert(payload_len == KP_STARTUP_PAYLOAD_LEN);
            uint8_t b_type = payload[0];
            uint8_t maj = payload[1];
            uint8_t min = payload[2];
            uint8_t pat = payload[3];
            uint32_t git = (uint32_t)payload[4] |
                           ((uint32_t)payload[5] << 8) |
                           ((uint32_t)payload[6] << 16) |
                           ((uint32_t)payload[7] << 24);
            uint32_t rate = (uint32_t)payload[8] |
                            ((uint32_t)payload[9] << 8) |
                            ((uint32_t)payload[10] << 16) |
                            ((uint32_t)payload[11] << 24);
            uint16_t spc = (uint16_t)((uint16_t)payload[12] | ((uint16_t)payload[13] << 8));
            uint8_t ch = payload[14];
            uint8_t mf = payload[15];

            uint8_t rebuilt[KP_STARTUP_PAYLOAD_LEN];
            kp_startup_payload(b_type, maj, min, pat, git, rate, spc, ch, mf, rebuilt);
            assert(memcmp(payload, rebuilt, KP_STARTUP_PAYLOAD_LEN) == 0);
        } else if (type == 3) { /* ENV */
            assert(payload_len == KP_ENV_PAYLOAD_LEN);
            int16_t t_c = (int16_t)((uint16_t)payload[0] | ((uint16_t)payload[1] << 8));
            uint16_t rh = (uint16_t)((uint16_t)payload[2] | ((uint16_t)payload[3] << 8));
            uint32_t overrun = (uint32_t)payload[4] |
                               ((uint32_t)payload[5] << 8) |
                               ((uint32_t)payload[6] << 16) |
                               ((uint32_t)payload[7] << 24);
            uint32_t drops = (uint32_t)payload[8] |
                             ((uint32_t)payload[9] << 8) |
                             ((uint32_t)payload[10] << 16) |
                             ((uint32_t)payload[11] << 24);
            uint8_t status = payload[12];

            uint8_t rebuilt[KP_ENV_PAYLOAD_LEN];
            kp_env_payload(t_c, rh, overrun, drops, status, rebuilt);
            assert(memcmp(payload, rebuilt, KP_ENV_PAYLOAD_LEN) == 0);
        }

        uint8_t reencoded[8192];
        size_t re_len = kp_frame_encode(type, node_id, seq, t_us, payload, payload_len, reencoded, sizeof(reencoded));
        assert(re_len == raw_len);
        assert(memcmp(raw, reencoded, raw_len) == 0);
        golden_count++;
    }
    fclose(f);
    printf("Golden frames verified: %d\n", golden_count);
    assert(golden_count >= 3);

    printf("All protocol core host tests passed!\n");
    return 0;
}
