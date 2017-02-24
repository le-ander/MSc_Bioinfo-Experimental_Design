#!/usr/bin/python2.5

from numpy import *
from numpy.random import *
import abcsysbio
import sys
import re
import time, os

sys.path.insert(0, '/cluster/home/saw112/work/Test_code/pycuda-2016.1.2/pycuda')

import cudasim
import cudasim.EulerMaruyama as EulerMaruyama
import cudasim.Gillespie as Gillespie
import cudasim.Lsoda as Lsoda

from pycuda import compiler, driver
from pycuda import autoinit

from abcsysbio import parse_infoEnt
from abcsysbio import model_py
from abcsysbio import model_cu
from abcsysbio import model_c
from abcsysbio import data
from abcsysbio import input_output

import abcsysbio_parser
from abcsysbio_parser import ParseAndWrite

#BLOCK_SIZE = 16
#RADIUS = 3

mod = compiler.SourceModule("""
    __device__ unsigned int idx3d(int i, int k, int l, int M, int P){
    return k*P + i*M*P + l;
    }

    __device__ unsigned int idx2d(int i, int j, int M){
        return i*M + j;
    }

    __global__ void staticReverse(int *input1, int *input2, int *output1){
        unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int tid = threadIdx.z;
        unsigned int i = blockIdx.z*(blockDim.z*2)+threadIdx.z; 

        __shared__ int s[2][2][64];

        s[threadIdx.x][threadIdx.y][tid] = (input1[idx2d(x,i,128)] + input2[idx2d(y,i,128)]) + (input1[idx2d(x,i+blockDim.z,128)] + input2[idx2d(y,i+blockDim.z,128)]);
        __syncthreads();

        for(unsigned int k=blockDim.z/2; k>0; k>>=1){
        	if(tid < k){
            	s[threadIdx.x][threadIdx.y][tid] += s[threadIdx.x][threadIdx.y][tid+k];
            }
           	__syncthreads();
        }

        if (tid==0) output1[idx2d(x,y,2)] = s[threadIdx.x][threadIdx.y][0];

        output1[idx2d(x,0,2)] = output1[idx2d(x,0,2)]+output1[idx2d(x,1,2)];


    }
    """)

staticReverse_GPU = mod.get_function("staticReverse")
input_1 = ones([2,128]).astype(int32)
input_2 = 2*input_1
output_1 = zeros([2,2]).astype(int32)
output_2 = zeros([2]).astype(int32)

print input_1
print input_2
print output_1

staticReverse_GPU(driver.In(input_1),driver.In(input_2),driver.Out(output_1), block=(int(2),int(2),int(64)),grid=(int(1),int(1),int(1)))

print output_1
