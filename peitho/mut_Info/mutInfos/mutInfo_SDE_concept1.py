from numpy import *
from numpy.random import *

from pycuda import compiler, driver
from pycuda import autoinit

import math


kernel_code_template = """

__device__ unsigned int idx3d(int i, int k, int l, int T, int S)
{
	return i*T*S + k*S + l;
}

__device__ unsigned int idx2d(int i, int j, int T)
{
	return i*T + j;
}

__global__ void testf(float pi, float invdet, double *x, double *mu, float *invcov, double *o1){

	double vector[%(S)s]={0.0};

	o1[0] = 0.0;

	for(int i=0; i<%(S)s; i++){
		for(int j=0; j<%(S)s; j++){
			vector[i] += (x[j] - mu[j]) * invcov[idx2d(j,i,%(S)s)];
		}
		o1[0] += vector[i] * (x[i] - mu[i]);
	}

	o1[0] = (sqrtf(invdet)/sqrtf(powf(2*pi,%(S)s))) * exp(-0.5*o1[0]);

}
"""
#seed(1234)
S = 10

kernel_code = kernel_code_template % {
	'S': S
	}

mod = compiler.SourceModule(kernel_code)

test = mod.get_function("testf")

input = rand(1,S).astype(float64)


x = 6*input
mu = 4*input

cov_pre = 4*rand(S,S).astype(float32)
cov = (cov_pre + cov_pre.T)/2
invcov = linalg.inv(cov)

invdet = abs(linalg.det(invcov))

o1 = zeros((1,1)).astype(float64)

print x
print mu
print invcov

test(float32(math.pi), float32(invdet), driver.In(x),driver.In(mu), driver.In(invcov),driver.Out(o1), block=(int(1),int(1),int(1)),grid=(int(1),int(1),int(1)))

numpy = (sqrt(invdet)/sqrt(pow(2*pi,S)))*exp(-0.5*dot(dot(x-mu,invcov),transpose(x-mu)))

print invdet

print "CPU OUT"
print numpy
print ""
print "GPU OUT"
print o1
print "error", (o1-numpy)/((o1+numpy)/2)
