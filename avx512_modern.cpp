#include "avx512_modern.h"

#include <immintrin.h>

#include "common-inl.h"
#include "quantize_avx2-inl.h"

inline static __m512i blk_2_bytes(__m512i x) {
    const __m512i prm = _mm512_set_epi8(
        39, 38, 39, 38, 37, 36, 37, 36, 35, 34, 35, 34, 33, 32, 33, 32,
        31, 30, 31, 30, 29, 28, 29, 28, 27, 26, 27, 26, 25, 24, 25, 24,
        19, 18, 19, 18, 17, 16, 17, 16, 15, 14, 15, 14, 13, 12, 13, 12,
        11, 10, 11, 10, 9, 8, 9, 8, 7, 6, 7, 6, 5, 4, 5, 4
    );
    const __m512i y = _mm512_permutexvar_epi8(prm, x);

    const __mmask32 shift_mask = 0xaaaaaaaa;
    const __m512i z = _mm512_mask_srai_epi16(y, shift_mask, y, 4);

    return _mm512_and_si512(_mm512_set1_epi8(0xF), z);
}

inline static void ggml_vec_dot_q4_0(const int n, float * __restrict__ s, const void * __restrict__ vx, const void * __restrict__ vy) {
    const int nb = n / QK;

    const auto * __restrict__ x = reinterpret_cast<const block_q4_0*>(vx);
    const auto * __restrict__ y = reinterpret_cast<const block_q4_0*>(vy);

    float sumf = 0.0;

    // Initialize accumulator with zeros
    __m512 acc = _mm512_setzero_ps();

    const __m512i off = _mm512_set1_epi8(8);
    const __mmask16 scale_mul_mask = 0x21;

    const __m512i prm = _mm512_set_epi32(
        5, 5, 5, 5,
        5, 5, 5, 5,
        0, 0, 0, 0,
        0, 0, 0, 0
    );

    for (int i = 0; i < nb; i += 2) {
        __m512i blk0 = _mm512_loadu_si512(&x[i]);
        __m512i blk1 = _mm512_loadu_si512(&y[i]);

        __m512 scales = _mm512_maskz_mul_ps(scale_mul_mask, (__m512)blk0, (__m512)blk1);
        scales = _mm512_permutexvar_ps(prm, scales);

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
