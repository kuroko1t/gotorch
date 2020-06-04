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

if [ "$FLAG_GPU" ]; then
    if [ ! -e libtorch ]; then
        wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.5.0.zip
        unzip libtorch-shared-with-deps-1.5.0.zip
        rm -f libtorch-shared-with-deps-1.5.0.zip
    fi
else
    if [ ! -e libtorch ]; then
        wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.5.0%2Bcpu.zip
        unzip libtorch-shared-with-deps-1.5.0%2Bcpu.zip
        rm -f libtorch-shared-with-deps-1.5.0%2Bcpu.zip
    fi
fi
