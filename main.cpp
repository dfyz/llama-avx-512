#include "avx512_modern.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>

constexpr size_t HidSize = 4096;
constexpr size_t InterSize = 11008;
constexpr size_t BatchSize = 8;
constexpr size_t WorkspaceSize = 65536;

struct Tensors {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
    ggml_tensor ref_dst;

    std::vector<char> workspace;
};

Tensors LoadTensors() {
    auto res = Tensors {
        .src0 = LoadFromFile("src0.bin", HidSize, InterSize, true),
        .src1 = LoadFromFile("src1.bin", HidSize, BatchSize, false),
        .ref_dst = LoadFromFile("dst.bin", InterSize, BatchSize, false),
    };
    res.dst = res.ref_dst;
    // Intentionally leak memory
    res.dst.data = new char[res.ref_dst.ByteSize()]{};
    res.workspace.resize(WorkspaceSize);
    return res;
}

void SanityCheck(const Tensors& tensors) {
    const float* actual = reinterpret_cast<const float*>(tensors.dst.data);
    const float* expected = reinterpret_cast<const float*>(tensors.ref_dst.data);
    for (size_t i = 0; i < tensors.dst.NumElems(); ++i) {
        const auto a = actual[i];
        const auto e = expected[i];
        if (std::isnan(a) != std::isnan(e) || std::abs(a - e) > 1e-6) {
            std::stringstream ss;
            ss << "Sanity check failed at index #" << i << ": " << a << " != " << e;
            throw std::runtime_error(ss.str());
        }
    }
}

int main() {
    auto tensors = LoadTensors();
    MatMulAvx512Modern(&tensors.src0, &tensors.src1, &tensors.dst, tensors.workspace.data());
    SanityCheck(tensors);
}
