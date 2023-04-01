#!/bin/sh

clang++ -O3 -g -std=c++20 -DNDEBUG -mavx2 -mavx512f -mavx512bw -mfma -mavx512vnni -mavx512vl -mavx512vbmi -o main \
    main.cpp \
    avx512_modern.cpp \
    ggml.cpp
