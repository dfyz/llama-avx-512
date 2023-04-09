#include "avx512_modern.h"

#include <immintrin.h>

#include "common-inl.h"
#include "quantize_avx2-inl.h"

inline static __m512i blk_2_bytes(__m512i x) {
#ifdef __AVX512VBMI__
    const __m512i prm = _mm512_set_epi8(
        39, 38, 39, 38, 37, 36, 37, 36, 35, 34, 35, 34, 33, 32, 33, 32,
        31, 30, 31, 30, 29, 28, 29, 28, 27, 26, 27, 26, 25, 24, 25, 24,
        19, 18, 19, 18, 17, 16, 17, 16, 15, 14, 15, 14, 13, 12, 13, 12,
        11, 10, 11, 10, 9, 8, 9, 8, 7, 6, 7, 6, 5, 4, 5, 4
    );
    const __m512i y = _mm512_permutexvar_epi8(prm, x);
#else
    const __m512i prm = _mm512_set_epi16(
        19, 19, 18, 18, 17, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12,
         9,  9,  8,  8,  7,  7,  6,  6,  5,  5,  4,  4,  3,  3,  2,  2
    );
    const __m512i y = _mm512_permutexvar_epi16(prm, x);
#endif

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

#ifdef __clang__
        __m512i scales;
        asm(
            "vmulps %1, %2, %0%{%3%}"
            : "=v" (scales)
            : "vm" (blk0), "v" (blk1), "Yk" (scale_mul_mask)
        );
#else
        __m512 scales = _mm512_maskz_mul_ps(scale_mul_mask, (__m512)blk0, (__m512)blk1);
#endif
        scales = _mm512_permutexvar_ps(prm, scales);

        const __m512i bx = blk_2_bytes(blk0);
        const __m512i by = blk_2_bytes(blk1);
        const __m512i by8 = _mm512_sub_epi8(by, off);

#ifdef __AVX512VNNI__
        const __m512i dot_init = _mm512_set1_epi32(4 * 64);
        const __m512i neg8 = _mm512_set1_epi8(-8);
        const __m512i aa = _mm512_dpbusds_epi32(dot_init, by, neg8);
        const __m512i prod = _mm512_dpbusds_epi32(aa, bx, by8);
#else
        const __m512i aa = _mm512_maddubs_epi16(bx, by8);
        const __m512i bb = _mm512_maddubs_epi16(off, by8);
        const __m512i diff = _mm512_sub_epi16(aa, bb);
        const __m512i prod = _mm512_madd_epi16(diff, _mm512_set1_epi16(1));
#endif

        acc = _mm512_fmadd_ps(
            scales,
            _mm512_cvtepi32_ps(prod),
            acc
        );
    }

    // Horizontal sum of all lanes of the accumulator
    *s = _mm512_reduce_add_ps( acc );
}

#define MAIN_FUNC_NAME MatMulAvx512Modern

#include "main_loop-inl.h"
