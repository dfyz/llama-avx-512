#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <cstring>
#include <ctime>
#include <vector>

static inline __m128i PackNibblesAvx2( __m256i bytes )
{
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

static inline __m128i PackNibblesAvx512( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
}

static inline std::vector<char> GenRandomInput() {
    std::vector<char> res(32768);
    res[0] = time(NULL) % 16;
    for (size_t i = 1; i < res.size(); ++i) {
        res[i] = (res[i - 1] + 1) % 16;
    }
    return res;;
}

static void BenchPackNibblesAvx2(benchmark::State& state) {
    auto input = GenRandomInput();
    const size_t n = input.size() / sizeof(__m256i);
    std::vector<char> output(input.size() / 2);

    for (auto _ : state) {
        const auto* in_ptr = reinterpret_cast<const __m256i*>(input.data());
        auto* out_ptr = reinterpret_cast<__m128i*>(output.data());
        for (size_t i = 0; i < n; ++i) {
            _mm_storeu_si128(
                out_ptr,
                PackNibblesAvx2(
                    _mm256_loadu_si256(in_ptr)
                )
            );
            ++in_ptr;
            ++out_ptr;
        }
    }
}

static void BenchPackNibblesAvx512(benchmark::State& state) {
    auto input = GenRandomInput();
    const size_t n = input.size() / sizeof(__m256i);
    std::vector<char> output(input.size() / 2);

    for (auto _ : state) {
        const auto* in_ptr = reinterpret_cast<const __m256i*>(input.data());
        auto* out_ptr = reinterpret_cast<__m128i*>(output.data());
        for (size_t i = 0; i < n; ++i) {
            _mm_storeu_si128(
                out_ptr,
                PackNibblesAvx512(
                    _mm256_loadu_si256(in_ptr)
                )
            );
            ++in_ptr;
            ++out_ptr;
        }
    }
}

BENCHMARK(BenchPackNibblesAvx2);
BENCHMARK(BenchPackNibblesAvx512);

BENCHMARK_MAIN();
