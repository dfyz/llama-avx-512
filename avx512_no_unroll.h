#pragma once

#include "ggml.h"

void MatMulAvx512NoUnroll(
    ggml_tensor* src0,
    ggml_tensor* src1,
    ggml_tensor* dst,
    char* wdata
);