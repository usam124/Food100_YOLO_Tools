#!/bin/bash

# Clone darknet repo (or include it in your repo already)
git clone https://github.com/AlexeyAB/darknet.git

cd darknet

# Edit Makefile to enable GPU, CUDNN, OPENCV, and Python
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/PYTHON=0/PYTHON=1/' Makefile

# Build darknet with python bindings
make

# Move built files back to root or appropriate folder
cp darknet.py ../
cp libdarknet.so ../  # or libdarknet.dll on Windows, or libdarknet.dylib on Mac

cd ..
