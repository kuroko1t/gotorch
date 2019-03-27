#!/usr/bin/env bash

usage_exit() {
    echo "Usage: ./$0 [-g] (for GPU build), ./$0 (for CPU build) " 1>&2
    exit 1
}

while getopts gch OPT
do
    case $OPT in
        g)  FLAG_GPU=1;echo "--GPU BUILD--";
            ;;
        c)  FLAG_CLEAN=1
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done


if [ "$FLAG_CLEAN" ]; then
    echo "rm -f build"
    rm -rf build
    exit
fi

if [ "$FLAG_GPU" ]; then
    if [ ! -e libtorch ]; then
        wget https://download.pytorch.org/libtorch/nightly/cu90/libtorch-shared-with-deps-latest.zip
        unzip libtorch-shared-with-deps-latest.zip
        rm -f libtorch-shared-with-deps-latest.zip
    fi
else
    if [ ! -e libtorch ]; then
        wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
        unzip libtorch-shared-with-deps-latest.zip
        rm -f libtorch-shared-with-deps-latest.zip
    fi
fi

if [ ! -e build ]; then
    mkdir build
fi

GOTORCH_DIR=$(pwd)

if [ "$FLAG_GPU" ]; then
    echo "koko"
    clang++ -Wall -g --std=c++11 -shared  -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$GOTORCH_DIR/cpp -I$GOTORCH_DIR/libtorch/include/ -I$GOTORCH_DIR/libtorch/include/torch/csrc/api/include/ \
        -L$GOTORCH_DIR/libtorch/lib -lcaffe2 -lc10 -ltorch -lpthread -lcuda -lnvrtc cpp/gotorch.cpp -o build/libgotorch.so
else
    clang++ -Wall -g --std=c++11 -shared  -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$GOTORCH_DIR/cpp -I$GOTORCH_DIR/libtorch/include/ -I$GOTORCH_DIR/libtorch/include/torch/csrc/api/include/ \
        -L$GOTORCH_DIR/libtorch/lib -lcaffe2 -lc10 -ltorch -lpthread cpp/gotorch.cpp -o build/libgotorch.so
fi
