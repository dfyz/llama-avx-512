#include "avx2_const_me.h"

#include "avx2.h"

#include <immintrin.h>

#include "common-inl.h"

static void quantize_row_q4_0(const float * __restrict__ x, void * __restrict__ vy, int k) {
    const int nb = k / QK;

    auto* __restrict__ y = reinterpret_cast<block_q4_0*>(vy);

    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 7.0f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 7.0f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );
        i2 = _mm256_packs_epi32( i2, i3 );
        // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );

        // Apply offset to translate the range from [ -7 .. +7 ] into [ +1 .. +15 ]
        const __m256i off = _mm256_set1_epi8( 8 );
        i0 = _mm256_add_epi8( i0, off );

        // Compress the vector into 4 bit/value
        __m128i res = packNibbles( i0 );

        // The AVX2 pack instructions above process 16-byte pieces independently
        // For this reason, the order of the values is now wrong, the following shuffle instruction is fixing that
        // vpshufb shuffles 16-bytes vectors, 3 times faster than vpermd which shuffles across the complete 32-bytes vectors
        const __m128i perm = _mm_setr_epi8( 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 );
        res = _mm_shuffle_epi8( res, perm );

        // Store the vector
        _mm_storeu_si128( ( __m128i* )y[i].qs, res );
    }
}

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
        const __m256 scale = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        __m256i bx = bytesFromNibbles( x[i].qs );
        __m256i by = bytesFromNibbles( y[i].qs );

        // Now we have a vector with bytes in [ 0 .. 15 ] interval, and we need sum( (a-8)*(b-8) )
        // The value we're after is equal to sum( a*(b-8) - 8*(b-8) )
        const __m256i off = _mm256_set1_epi8( 8 );
        by = _mm256_sub_epi8( by, off );
        // These weird multiplication instructions compute a0*b0 + a1*b1 for uint8_t a, int8_t b
        __m256i p1 = _mm256_maddubs_epi16( bx, by );
        __m256i p2 = _mm256_maddubs_epi16( off, by );
        __m256i p16 = _mm256_sub_epi16( p1, p2 );

        // We have products of signed bytes, reduced pairwise to int16_t
        // Reduce pairs further to int32_t
        // The following preprocessor branches implement two equivalent methods of doing so
        // Which way is faster, probably depends on CPU.
#if 0
        __m256i i32 = _mm256_slli_epi32( p16, 16 );
        // This works because maximum value of 1 product is -8^2 = +64
        // int16_t lanes don't overflow even with sums of 4 of these numbers
        i32 = _mm256_add_epi16( i32, p16 );
        // Arithmetic shift = sign extend
        i32 = _mm256_srai_epi32( i32, 16 );
#else
        // Competes for the same ports as _mm256_maddubs_epi16, needs the constant vector with ones,
        // and takes 3-5 cycles of latency
        // However, that's 1 instruction instead of 3.
        __m256i i32 = _mm256_madd_epi16( p16, _mm256_set1_epi16( 1 ) );
#endif

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps( i32 );
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

#define MAIN_FUNC_NAME MatMulAvx2ConstMe

#include "main_loop-inl.h"
