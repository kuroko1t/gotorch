#!/usr/bin/env bash

if [ ! -e build ]; then
    mkdir build
fi

clang++-6.0 --std=c++11 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I/root/gotorch/cpp -I/root/libtorch/include/ -I/root/libtorch/include/torch/csrc/api/include/ -L/root/libtorch/lib -lcaffe2 -lc10 -ltorch -lpthread cpp/gotorch.cpp -o build/libgotorch.so
