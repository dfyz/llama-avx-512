#include "avx512_modern.h"

#include <immintrin.h>

#include "common-inl.h"
#include "quantize_avx2-inl.h"

inline static __m512 blk_2_scales(__m512i x) {
    const __m512i prm = _mm512_set_epi32(
        5, 5, 5, 5,
        5, 5, 5, 5,
        0, 0, 0, 0,
        0, 0, 0, 0
    );
    return _mm512_permutexvar_ps(prm, (__m512)x);
}

inline static __m512i blk_2_bytes(__m512i x) {
    const __m512i prm = _mm512_set_epi8(
        39, 39, 38, 38, 37, 37, 36, 36, 35, 35, 34, 34, 33, 33, 32, 32,
        31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
        19, 19, 18, 18, 17, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12,
        11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4
    );
    const __m512i y = _mm512_permutexvar_epi8(prm, x);

    const __m512i prm2 = _mm512_set1_epi64(0x3c302c201c100c00);
    const __m512i z = _mm512_multishift_epi64_epi8(prm2, y);

    return _mm512_and_si512(_mm512_set1_epi8(0xF), z);
}

inline static void ggml_vec_dot_q4_0(const int n, float * __restrict__ s, const void * __restrict__ x, const void * __restrict__ y) {
    const int nb = n / QK;

    const size_t bs = sizeof(float) + QK/2;

    const uint8_t * __restrict__ pd0 = ((const uint8_t *)x + 0*bs);
    const uint8_t * __restrict__ pd1 = ((const uint8_t *)y + 0*bs);

    float sumf = 0.0;

    // Initialize accumulator with zeros
    __m512 acc = _mm512_setzero_ps();

    const __mmask64 blk_2_mask = 0xFF'FF'FF'FF'FFUL;
    const __m512i off = _mm512_set1_epi8(8);

    for (int i = 0; i < nb; i += 2) {
        __m512i blk0 = _mm512_maskz_loadu_epi8(blk_2_mask, pd0 + i*bs);
        __m512i blk1 = _mm512_maskz_loadu_epi8(blk_2_mask, pd1 + i*bs);

        const __m512 scales = _mm512_mul_ps(
            blk_2_scales(blk0),
            blk_2_scales(blk1)
        );

        const __m512i bx = blk_2_bytes(blk0);
        __m512i by = blk_2_bytes(blk1);
        by = _mm512_sub_epi8(by, off);

        const __m512i aa = _mm512_dpbusds_epi32(
            _mm512_setzero_epi32(),
            bx, by
        );
        const __m512i bb = _mm512_dpbusds_epi32(
            _mm512_setzero_epi32(),
            off, by
        );
        const __m512i diff = _mm512_sub_epi32(aa, bb);

        acc = _mm512_fmadd_ps(
            scales,
            _mm512_cvtepi32_ps(diff),
            acc
        );
    }

    // Horizontal sum of all lanes of the accumulator
    *s = _mm512_reduce_add_ps( acc );
}

#define MAIN_FUNC_NAME MatMulAvx512Modern

#include "main_loop-inl.h"