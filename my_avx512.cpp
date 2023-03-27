#include "avx2.h"

#include <immintrin.h>

constexpr size_t QK = 32;

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytesFromNibbles( const uint8_t* rsi )
{
    // Load 16 bytes from memory
    __m128i tmp = _mm_loadu_si128( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m256i bytes = _mm256_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    __m256i high = _mm256_andnot_si256( lowMask, bytes );
    __m256i low = _mm256_and_si256( lowMask, bytes );
    high = _mm256_slli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );
    return bytes;
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
}

static void quantize_row_q4_0(const float * __restrict__ x, void * __restrict__ y, int k) {
    const int nb = k / QK;
    const size_t bs = sizeof(float) + QK/2;

    uint8_t * __restrict__ pd = ((uint8_t *)y + 0*bs);
    uint8_t * __restrict__ pb = ((uint8_t *)y + 0*bs + sizeof(float));

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
        *(float *)pd = d;
        pd += bs;
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
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        // Apply offset to translate the range from [ -7 .. +7 ] into [ +1 .. +15 ]
        const __m256i off = _mm256_set1_epi8( 8 );
        i0 = _mm256_add_epi8( i0, off );

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles( i0 );
        _mm_storeu_si128( ( __m128i* )pb, res );
        pb += bs;
    }
}

static inline __m512 dot_q4_0_oneblock_avx512(
    __m512 acc,
    const uint8_t * pd0,
    const uint8_t * pd1,
    const uint8_t * pb0,
    const uint8_t * pb1,
    size_t bs,
    int i
) {
    const float * d0_0 = (const float *) (pd0 + i*bs);
    const float * d1_0 = (const float *) (pd1 + i*bs);

    const uint8_t * __restrict__ p0 = pb0 + (i+0)*bs;
    const uint8_t * __restrict__ p1 = pb1 + (i+0)*bs;

    // Compute combined scale for the block
    float scaleScalar = d0_0[0] * d1_0[0];
    __m512 scale = _mm512_set1_ps( scaleScalar );

    __m256i bx = bytesFromNibbles( p0 );
    __m256i by = bytesFromNibbles( p1 );

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
    return _mm512_fmadd_ps( scale, p, acc );
}

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
        __m512i blk0 = _mm512_maskz_loadu_epi8(blk_2_mask, pd0 + 2*i*bs);
        __m512i blk1 = _mm512_maskz_loadu_epi8(blk_2_mask, pd1 + 2*i*bs);

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

void MyMatMulAvx512(
    ggml_tensor* src0,
    ggml_tensor* src1,
    ggml_tensor* dst,
    char* wdata
) {
    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    char* wd = wdata;
    for (int i13 = 0; i13 < ne13; ++i13) {
        for (int i12 = 0; i12 < ne12; ++i12) {
            for (int i11 = 0; i11 < ne11; ++i11) {
                quantize_row_q4_0((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) wd, ne10);
                wd += (ne10*20)/32;
            }
        }
    }

    const int nr = ne01*ne02*ne03;

    for (int ir = 0; ir < nr; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int i13 = i03;
        const int i12 = i02;

        const int i0 = i01;
        const int i2 = i02;
        const int i3 = i03;

        void * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        char * src1_col =          ((char *)      wdata + (      (0 + i12*ne11 + i13*ne12*ne11)*ne00*20)/32);

        float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

        for (int ic = 0; ic < ne11; ++ic) {
            ggml_vec_dot_q4_0(ne00, &dst_col[ic*ne0], src0_row, ((void *) (src1_col + (ic*ne00*20)/32)));
        }
    }
}