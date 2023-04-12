#include <immintrin.h>
#include <sys/mman.h>

int main() {
	char* x = mmap(NULL, 4096 * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	mprotect(x + 4096, 4096, PROT_NONE);

	const size_t len = 40;
	__m512i y = _mm512_maskz_loadu_epi8(0xFFFFFFFFFFULL, (__m512i*)(x + 4096 - len));
	//__m512i z = _mm512_loadu_epi8((__m512i*)(x + 4096 - len));
}
