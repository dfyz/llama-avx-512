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

    /* Prepare the constants we will need during execution */
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    const __m256i offset_8 = _mm256_set1_epi16( 8 );

#define UNROLL_COUNT 8
    // Main loop
    for (int i = 0; i < nb; i+=UNROLL_COUNT) {

        // This loop will be unrolled by the compiler
        for (int u=0;u<UNROLL_COUNT;u++)  {
            /* Compute combined scale for the block */
            const __m256 scale = _mm256_mul_ps(
                    _mm256_broadcast_ss( &x[i+u].d ),
                    _mm256_broadcast_ss( &y[i+u].d ) );

            /* get input from x
               Input: 32 Nibbles (16 bytes) at *x[i+u]
               Output: 2 vectors with 16 values of type int16_t (x_high_q, x_low_q) */

            /* Load 16 bytes from memory */
            const __m128i tmp_x = _mm_loadu_si128( ( const __m128i* ) x[i+u].qs);
            /* Expand bytes into uint16_t values */
            const __m256i bytes_x = _mm256_cvtepu8_epi16(tmp_x);
            /* Unpack values into individual bytes */
            __m256i x_low_q = _mm256_and_si256( lowMask, bytes_x );
            const __m256i pre_shift_x_high_q = _mm256_andnot_si256( lowMask, bytes_x );
            __m256i x_high_q = _mm256_srli_epi16( pre_shift_x_high_q, 4 );
            /* Now we have two vectors with bytes in [ 0 .. 15 ] interval.  Offset them into [ -8 .. +7 ] interval.  */
            x_high_q = _mm256_sub_epi16( x_high_q, offset_8 );
            x_low_q = _mm256_sub_epi16( x_low_q, offset_8 );

            /* get input from y
               Input: 32 Nibbles (16 bytes) at *y[i+u]
               Output: 2 vectors with 16 values of type int16_t (y_high_q, y_low_q) */

            /* Load 16 bytes from memory */
            const __m128i tmp_y = _mm_loadu_si128( (const __m128i* ) y[i+u].qs);
            /* Expand bytes into uint16_t values */
            const __m256i bytes_y = _mm256_cvtepu8_epi16(tmp_y);
            /* Unpack values into individual bytes */
            const __m256i pre_shift_y_high_q = _mm256_andnot_si256( lowMask, bytes_y );
            __m256i y_high_q = _mm256_srli_epi16( pre_shift_y_high_q, 4 );
            __m256i y_low_q = _mm256_and_si256( lowMask, bytes_y );
            /* Now we have two vectors with bytes in [ 0 .. 15 ] interval.  Offset them into [ -8 .. +7 ] interval.  */
            y_high_q = _mm256_sub_epi16( y_high_q, offset_8 );
            y_low_q = _mm256_sub_epi16( y_low_q, offset_8 );

            /* Compute products of int16_t integers, add pairwise, store as int32_t */
            __m256i xy_high_q = _mm256_madd_epi16( x_high_q, y_high_q );
            __m256i xy_low_q = _mm256_madd_epi16( x_low_q, y_low_q );

            /* Accumulate the products of int32_t integers -> we now have a vector of 8 int_32t */
            __m256i xy_q = _mm256_add_epi32( xy_high_q, xy_low_q );

            /* Convert to vectore of 8 int32_t to 8 floats */
            __m256 q = _mm256_cvtepi32_ps( xy_q );

            /* Multiply q with scale and accumulate */
            acc = _mm256_fmadd_ps( scale, q, acc );
        }

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
