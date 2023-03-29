#include "avx512.h"

#include <immintrin.h>

#include "common-inl.h"
#include "quantize_avx2-inl.h"

inline static void ggml_vec_dot_q4_0(const int n, float * __restrict__ s, const void * __restrict__ vx, const void * __restrict__ vy) {
    const int nb = n / QK;

    const auto * __restrict__ x = reinterpret_cast<const block_q4_0*>(vx);
    const auto * __restrict__ y = reinterpret_cast<const block_q4_0*>(vy);

    float sumf = 0.0;

    // Initialize accumulator with zeros
    __m512 acc = _mm512_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        __m512 d = _mm512_set1_ps( x[i].d * y[i].d );

        __m256i bx = bytesFromNibbles( x[i].qs );
        __m256i by = bytesFromNibbles( y[i].qs );

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        bx = _mm256_sub_epi8( bx, off );
        by = _mm256_sub_epi8( by, off );

        // Sign-extend 16 signed bytes into int16_t
        __m512i x32 = _mm512_cvtepi8_epi16( bx );
        __m512i y32 = _mm512_cvtepi8_epi16( by );
        // Compute products of int16_t integers, add pairwise
        __m512i i64 = _mm512_madd_epi16( x32, y32 );

        // Convert int32_t to float
        __m512 p = _mm512_cvtepi32_ps( i64 );
        // Apply the scale, and accumulate
        acc = _mm512_fmadd_ps( d, p, acc );
    }

    // Horizontal sum of all lanes of the accumulator
    *s = _mm512_reduce_add_ps( acc );
}

#define MAIN_FUNC_NAME MatMulAvx512NoUnroll

#include "main_loop-inl.h"
