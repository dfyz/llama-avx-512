#include "ggml.h"

#include <algorithm>
#include <fstream>

ggml_tensor LoadFromFile(
    const std::string& file_name,
    size_t ne0,
    size_t ne1,
    bool is_quantized
) {
    ggml_tensor res = {
        .ne = {ne0, ne1, 1, 1},
        .data = nullptr,
        .padding = {0},
    };

    res.nb[0] = is_quantized ? 20 : 4;
    if (is_quantized) {
        res.nb[1] = (ne0 / 32) * res.nb[0];
    } else {
        res.nb[1] = ne0 * res.nb[0];
    }
    const size_t byte_size = ne1 * res.nb[1];
    res.nb[2] = res.nb[3] = byte_size;

    // Intentionally leak memory.
    res.data = new char[byte_size]{};
    std::ifstream inp{file_name, std::ios::binary};
    inp.read(reinterpret_cast<char*>(res.data), byte_size);

    return res;
}
