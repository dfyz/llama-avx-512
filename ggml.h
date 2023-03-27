#pragma once

#include <string>

struct ggml_tensor {
    size_t ne[4];
    size_t nb[4];

    void * data;
    char padding[8];

    size_t ByteSize() const {
      return nb[3];
    }

    size_t NumElems() const {
      return ne[0] * ne[1] * ne[2] * ne[3];
    }
};

ggml_tensor LoadFromFile(
    const std::string& file_name,
    size_t ne0,
    size_t ne1,
    bool is_quantized
);
