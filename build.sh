#!/bin/sh

g++ -O3 -g -std=c++20 -DNDEBUG -mavx2 -mavx512f -mavx512bw -mfma -o main \
    main.cpp \
    avx2.cpp \
    avx512.cpp \
    ggml.cpp \
    $(pkg-config --libs --cflags benchmark)
