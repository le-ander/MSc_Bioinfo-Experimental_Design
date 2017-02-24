#!/usr/bin/python2.5

from numpy import *
from numpy.random import *
import abcsysbio
import sys
import re
import time, os


from pycuda import compiler, driver
from pycuda import autoinit

#BLOCK_SIZE = 16
#RADIUS = 3

mod = compiler.SourceModule("""
    __global__ void reduce0(int *g_idata, int *g_odata){
        extern __shared__ int sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;

        sdata[tid] = g_idata[i]+g_idata[i+blockDim.x];
        __syncthreads();


        for(unsigned int s=blockDim.x/2; s>0; s>>=1){
            if (tid < s){
                sdata[tid] += sdata[tid+s];
            }
            __syncthreads();
        }

        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
    """)

reduce0_GPU = mod.get_function("reduce0")
input_1 = zeros([512]).astype(int32)
for i, value in enumerate(range(1,513)):
    input_1[i] = value
output_1 = zeros([2]).astype(int32)

print input_1
print sum(input_1)
print output_1

reduce0_GPU(driver.In(input_1),driver.Out(output_1), block=(int(128),int(1),int(1)),grid=(int(2),int(1),int(1)),shared=128*32)

print output_1
print sum(output_1)