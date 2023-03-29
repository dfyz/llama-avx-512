#pragma once

#include "ggml.h"

void MatMulAvx2ConstMe(
    ggml_tensor* src0,
    ggml_tensor* src1,
    ggml_tensor* dst,
    char* wdata
);