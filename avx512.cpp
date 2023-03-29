#include "avx512.h"

#include <immintrin.h>

#include "common-inl.h"
#include "quantize_avx2-inl.h"

static inline __m512 dot_q4_0_oneblock_avx512(
    __m512 acc,
    const block_q4_0 * __restrict__ x,
    const block_q4_0 * __restrict__ y,
    int i
) {
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
    return _mm512_fmadd_ps( d, p, acc );
}

inline static void ggml_vec_dot_q4_0(const int n, float * __restrict__ s, const void * __restrict__ vx, const void * __restrict__ vy) {
    const int nb = n / QK;

    const auto * __restrict__ x = reinterpret_cast<const block_q4_0*>(vx);
    const auto * __restrict__ y = reinterpret_cast<const block_q4_0*>(vy);

    float sumf = 0.0;

    // Initialize accumulator with zeros
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    const int superblock_size = 8;
    const int superblock_count = nb / superblock_size;
    const int remainder = nb % superblock_size;

    for (int superblock_ix = 0; superblock_ix < superblock_count; superblock_ix += 1) {
        int i = superblock_ix * superblock_size;

        acc0 = dot_q4_0_oneblock_avx512( acc0, x, y, i+0 );
        acc1 = dot_q4_0_oneblock_avx512( acc1, x, y, i+1 );
        acc0 = dot_q4_0_oneblock_avx512( acc0, x, y, i+2 );
        acc1 = dot_q4_0_oneblock_avx512( acc1, x, y, i+3 );
        acc0 = dot_q4_0_oneblock_avx512( acc0, x, y, i+4 );
        acc1 = dot_q4_0_oneblock_avx512( acc1, x, y, i+5 );
        acc0 = dot_q4_0_oneblock_avx512( acc0, x, y, i+6 );
        acc1 = dot_q4_0_oneblock_avx512( acc1, x, y, i+7 );
    }

    // Remainders
    for (int i = superblock_count * superblock_size; i < nb; ++i) {
        acc0 = dot_q4_0_oneblock_avx512( acc0, x, y, i );
    }

    // Horizontal sum of all lanes of the accumulator
    sumf = _mm512_reduce_add_ps( acc0 ) + _mm512_reduce_add_ps( acc1 );

    *s = sumf;
}

#define MAIN_FUNC_NAME MatMulAvx512

#include "main_loop-inl.h"
