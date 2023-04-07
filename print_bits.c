#include <stdint.h>
#include <stdio.h>

void print_byte(uint8_t x) {
	for (size_t i = 8; i > 0; --i) {
		putchar((x & (1 << (i - 1))) ? '1' : '0');
		if (i == 5 || i == 4) {
			putchar('\'');
		}
	}
}

int main() {
	for (uint8_t x = 0; x < 16; ++x) {
		uint8_t y = (uint8_t)((int)x - 8);
		printf("%02d: ", x);
		print_byte(x);
		printf(" -> ");
		print_byte(y);
		puts("");
	}

	return 0;
}
