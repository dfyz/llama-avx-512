#!/bin/sh

g++ -O3 -g -std=c++20 -DNDEBUG -mavx2 -mfma -o main \
    main.cpp \
    avx2.cpp \
    ggml.cpp \
    $(pkg-config --libs --cflags benchmark)
    # avx512.cpp avx512_mine.cpp