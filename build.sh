#!/bin/sh

g++ -O3 -g -std=c++20 -DNDEBUG -mfma -mavx2 \
    -mavx512f -mavx512bw -mavx512vnni -mavx512vbmi \
    -o main \
    main.cpp \
    avx2.cpp \
    avx2_const_me.cpp \
    avx512.cpp \
    avx512_no_unroll.cpp \
    avx512_modern.cpp \
    ggml.cpp \
    $(pkg-config --libs --cflags benchmark)
