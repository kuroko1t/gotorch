#!/usr/bin/eonv bash

if [ ! -e libtorch ]; then
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    unzip libtorch-shared-with-deps-latest.zip
    rm -f libtorch-shared-with-deps-latest.zip
fi

if [ ! -e build ]; then
    mkdir build
fi

GOTORCH_DIR=$(pwd)

clang++-6.0 -Wall -g --std=c++11 -shared  -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$GOTORCH_DIR/cpp -I$GOTORCH_DIR/libtorch/include/ -I$GOTORCH_DIR/libtorch/include/torch/csrc/api/include/ \
            -L$GOTORCH_DIR/libtorch/lib -lcaffe2 -lc10 -ltorch -lpthread cpp/gotorch.cpp -o build/libgotorch.so
