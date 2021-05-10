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
        wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip #https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
	      
        unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cu111.zip
	rm -f libtorch-cxx11-abi-shared-with-deps-1.8.1+cu111.zip
    fi
else
    if [ ! -e libtorch ]; then
	wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip #https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip
        unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu.zip
	rm -f libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu.zip
    fi
fi
