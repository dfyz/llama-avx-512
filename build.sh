#!/bin/sh

g++ -O3 -g -std=c++20 -DNDEBUG -mavx2 -mavx512f -mavx512bw -mfma -mavx512vnni -mavx512vl -mavx512vbmi -o main \
    main.cpp \
    avx2.cpp \
    avx2_const_me.cpp \
    avx512.cpp \
    avx512_no_unroll.cpp \
    my_avx512.cpp \
    ggml.cpp \
    $(pkg-config --libs --cflags benchmark)
