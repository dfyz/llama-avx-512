#include "avx2.h"

#include <immintrin.h>

#include "common-inl.h"
#include "quantize_avx2-inl.h"

inline static void ggml_vec_dot_q4_0(const int n, float * __restrict__ s, const void * __restrict__ x, const void * __restrict__ y) {
    const int nb = n / QK;

    const size_t bs = sizeof(float) + QK/2;

    const uint8_t * __restrict__ pd0 = ((const uint8_t *)x + 0*bs);
    const uint8_t * __restrict__ pd1 = ((const uint8_t *)y + 0*bs);

    const uint8_t * __restrict__ pb0 = ((const uint8_t *)x + 0*bs + sizeof(float));
    const uint8_t * __restrict__ pb1 = ((const uint8_t *)y + 0*bs + sizeof(float));

    float sumf = 0.0;

    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        const float * d0_0 = (const float *) (pd0 + i*bs);
        const float * d1_0 = (const float *) (pd1 + i*bs);

        const uint8_t * __restrict__ ptr0 = pb0 + i*bs;
        const uint8_t * __restrict__ ptr1 = pb1 + i*bs;

        // Compute combined scale for the block
        const __m256 scale = _mm256_mul_ps( _mm256_broadcast_ss( d0_0 ), _mm256_broadcast_ss( d1_0 ) );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        __m256i bx = bytesFromNibbles( ptr0 );
        __m256i by = bytesFromNibbles( ptr1 );

        const __m256i off = _mm256_set1_epi8( 8 );
        by = _mm256_sub_epi8( by, off );
        // These weird multiplication instructions compute a0*b0 + a1*b1 for uint8_t a, int8_t b
        __m256i aa = _mm256_dpbusd_epi32( _mm256_setzero_ps(), bx, by);
        __m256i bb = _mm256_dpbusd_epi32( _mm256_setzero_ps(), off, by );
        __m256i cc = _mm256_sub_epi32(aa, bb);

        // Competes for the same ports as _mm256_maddubs_epi16, needs the constant vector with ones,
        // and takes 3-5 cycles of latency
        // However, that's 1 instruction instead of 3.
        // __m256i i32 = _mm256_madd_epi16( p16, _mm256_set1_epi16( 1 ) );

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps( cc );
        // Apply the scale, and accumulate
        acc = _mm256_fmadd_ps( scale, p, acc );
    }

    // Return horizontal sum of the acc vector
    __m128 res = _mm256_extractf128_ps( acc, 1 );
    res = _mm_add_ps( res, _mm256_castps256_ps128( acc ) );
    res = _mm_add_ps( res, _mm_movehl_ps( res, res ) );
    res = _mm_add_ss( res, _mm_movehdup_ps( res ) );

    sumf = _mm_cvtss_f32( res );

    *s = sumf;
}

#define MAIN_FUNC_NAME MatMulAvx2

#include "main_loop-inl.h"
