#include "avx2.h"

#include <immintrin.h>

#include "common-inl.h"
#include "quantize_avx2-inl.h"

inline static void ggml_vec_dot_q4_0(const int n, float * __restrict__ s, const void * __restrict__ vx, const void * __restrict__ vy) {
    const int nb = n / QK;

    const auto * __restrict__ x = reinterpret_cast<const block_q4_0*>(vx);
    const auto * __restrict__ y = reinterpret_cast<const block_q4_0*>(vy);

    float sumf = 0.0;

        // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        const __m256 d = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        __m256i bx = bytesFromNibbles( x[i].qs );
        __m256i by = bytesFromNibbles( y[i].qs );

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        bx = _mm256_sub_epi8( bx, off );
        by = _mm256_sub_epi8( by, off );

        // Sign-extend first 16 signed bytes into int16_t
        __m256i x16 = _mm256_cvtepi8_epi16( _mm256_castsi256_si128( bx ) );
        __m256i y16 = _mm256_cvtepi8_epi16( _mm256_castsi256_si128( by ) );
        // Compute products of int16_t integers, add pairwise
        __m256i i32 = _mm256_madd_epi16( x16, y16 );

        // Sign-extend last 16 signed bytes into int16_t vectors
        x16 = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( bx, 1 ) );
        y16 = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( by, 1 ) );
        // Accumulate products of int16_t integers
        i32 = _mm256_add_epi32( i32, _mm256_madd_epi16( x16, y16 ) );

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps( i32 );
        // Apply the scale, and accumulate
        acc = _mm256_fmadd_ps( d, p, acc );
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
