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
