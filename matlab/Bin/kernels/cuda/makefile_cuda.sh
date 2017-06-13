#!/bin/bash

# This small script shows how to compile the cuda mex files. It has been tested :
# on a Debian Jessie/Sid and Ubuntu 14.04 systems with Cuda 5.5 to 7.5 (packaged version) and matlab R2013b and R2014a.
# If cuda was manually installed make sure that path "CUDAROOT" to cuda libs is correct and that ld knows where 
# libcudart.so.* is located (modify or create a LD_LIBRARY_PATH variable). Please adapt all the other values to fit your configuration. 
# Author : B. Charlier (2017)


#------------------------------------#
#        CHECK THESE PATHS           #
#------------------------------------#


MATLABROOT="/usr/local/MATLAB/R2014a"
CUDAROOT="/usr/local/cuda-7.5/lib64"
MEXC="$MATLABROOT/bin/mex"
CC="/usr/bin/gcc"
NVCC="nvcc"

# CHECK THESE PARAMETERS (depends on your GPU):
COMPUTECAPABILITY=35
USE_DOUBLE=0
NVCCOPT="--use_fast_math"
BLOCKSIZE=192

# NVCC
NVCCFLAGS="-ccbin=$CC -arch=sm_$COMPUTECAPABILITY -Xcompiler -fPIC"
MEXPATH="-I$MATLABROOT/extern/include"

# C
COPTIMFLAG="-O3" 
CLIB="-L$CUDAROOT -lcudart"

INSTALL_DIR="../binaries"

#clean
	rm -f *.o;

#---------------------------------------#
#         fshapes distances             #
#---------------------------------------#

#clean
	rm -f *.o;

# Wasserstein distances (compiled with double)

$NVCC -c -D "USE_DOUBLE_PRECISION=1" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" ./shapes_distances/wasserstein/sinkhornGpuConv.cu $NVCCFLAGS $MEXPATH -o sinkhornGpuConv.o;echo "sinkhornGpuConv.cu successfully compiled";
$NVCC -c -D "USE_DOUBLE_PRECISION=1" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" ./shapes_distances/wasserstein/sinkhornGpuConv_unbalanced.cu $NVCCFLAGS $MEXPATH -o sinkhornGpuConv_unbalanced.o;echo "sinkhornGpuConv_unbalanced.cu successfully compiled";
$NVCC -c -D "USE_DOUBLE_PRECISION=1" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" ./shapes_distances/wasserstein/dsinkhornGpuConv.cu $NVCCFLAGS $MEXPATH -o dsinkhornGpuConv.o;echo "dsinkhornGpuConv.cu successfully compiled";

#mex complilation
	for i in `ls *.o`;do $MEXC GCC=$CC COPTIMFLAGS=$COPTIMFLAG $i $CLIB;done

# install	
	mkdir -p "$INSTALL_DIR/fshapes_distances/wasserstein"

	for i in `ls *.mexa64`;do 
		mv $i "$INSTALL_DIR/fshapes_distances/wasserstein/";
		echo "$i successfully installed"
	done

#---------------------------------------#
#             Kernels                   #
#---------------------------------------#

#nvcc compilation of every .cu files
	for i in `ls *.cu`;do $NVCC -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -c $i $NVCCFLAGS $MEXPATH $NVCCOPT;echo "$i successfully compiled";done

#mex complilation
	for i in `ls *.o`;do $MEXC GCC=$CC COPTIMFLAGS=$COPTIMFLAG $i $CLIB;done

#clean
	rm -f *.o;

# install	
	mkdir -p "$INSTALL_DIR"

	for i in `ls *.mexa64`;do 
		mv $i "$INSTALL_DIR";
		echo "$i successfully installed"
	done

