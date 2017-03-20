from numpy import *
from numpy.random import *

from pycuda import compiler, driver
from pycuda import autoinit

import math


kernel_code_template = """

__device__ unsigned int idx3d(int i, int k, int l, int M, int P)
{
	return k*P + i*M*P + l;
}

__device__ unsigned int idx2d(int i, int j, int M)
{
	return i*M + j;
}

__global__ void testf(float pi, float invdet, double *x, double *mu, float *invcov, double *o1){

    double vector[%(P)s]={0.0};

    o1[0] = 0.0;

	for(int i=0; i<%(P)s; i++){
    	for(int j=0; j<%(P)s; j++){
            vector[i] += (x[j] - mu[j]) * invcov[idx2d(j,i,%(P)s)];
            }
        o1[0] += vector[i] * (x[i] - mu[i]);
	}

    o1[0] = (sqrtf(invdet)/sqrtf(powf(2*pi,%(P)s))) * exp(-0.5*o1[0]);

}
"""
seed(1234)
P = 10

kernel_code = kernel_code_template % {
    'P': P
    }

mod = compiler.SourceModule(kernel_code)

test = mod.get_function("testf")

input = rand(1,P).astype(float64)


x = 2*input
mu = 3*input

cov_pre = rand(P,P).astype(float32)
cov = (cov_pre + cov_pre.T)/2
invcov = linalg.inv(cov)

invdet = abs(linalg.det(invcov))

o1 = zeros((1,1)).astype(float64)

print x
print mu
print invcov

test(float32(math.pi), float32(invdet), driver.In(x),driver.In(mu), driver.In(invcov),driver.Out(o1), block=(int(1),int(1),int(1)),grid=(int(1),int(1),int(1)))

numpy = (sqrt(invdet)/sqrt(pow(2*pi,P)))*exp(-0.5*dot(dot(x-mu,invcov),transpose(x-mu)))

print invdet

print "CPU OUT"
print numpy
print ""
print "GPU OUT"
print o1
