import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy

a = gpuarray.arange(400, dtype=numpy.float32)
b = gpuarray.arange(400, dtype=numpy.float32)

from pycuda.reduction import ReductionKernel

krnl = ReductionKernel(numpy.float32,
	numpy.float32,
	arguments="float *x, float *y",
	map_expr="x[i]*y[i]",
	reduce_expr="a+b")

my_dot_prod = krnl(a,b).get()