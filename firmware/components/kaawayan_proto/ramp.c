/*
 * ramp.c – Ramp test-mode audio fill
 *
 * Global sample index n = seq * 512 + i.
 * L = (int16_t)(uint16_t)(n & 0xFFFFu)
 * R = ~L (as int16_t)
 */
#include "kaawayan_proto.h"

void kp_ramp_fill(uint32_t seq, int16_t *stereo512)
{
    if (stereo512 == NULL) {
        return;
    }
    for (uint32_t i = 0; i < 512; i++) {
        uint64_t n = ((uint64_t)seq * 512ull) + (uint64_t)i;
        int16_t l = (int16_t)(uint16_t)(n & 0xFFFFu);
        int16_t r = (int16_t)(~l);
        stereo512[2u * i]     = l;
        stereo512[2u * i + 1] = r;
    }
}
