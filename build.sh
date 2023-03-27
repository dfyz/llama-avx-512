#!/bin/sh

g++ -O3 -o main \
    main.cpp \
    avx2.cpp \
    ggml.cpp \
    $(pkg-config --libs --cflags benchmark)
    # avx512.cpp avx512_mine.cpp