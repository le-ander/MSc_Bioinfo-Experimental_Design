from numpy import *
from numpy.random import *

from pycuda import compiler, driver
from pycuda import autoinit

import math


kernel_code_template = """
__device__ unsigned int idx4d(int i, int j, int k, int l, int B, int T, int S)
{
	return i*T*S*B + j*T*S + k*S + l;
}

__device__ unsigned int idx3d(int i, int k, int l, int T, int S)
{
	return i*T*S + k*S + l;
}

__device__ unsigned int idx2d(int i, int j, int T)
{
	return i*T + j;
}

__global__ void matrixmult(float pi, float *invdet, double *x, double *mu, float *invcov, double *o1){

	unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;

	double vector1[%(S)s]={0.0};
	double vector2[%(T)s]={0.0};

	o1[0] = 0.0;

	for(int t=0; t<%(T)s; t++){

		for(int s_i=0; s_i<%(S)s; s_i++){

			vector1[idx2d(t,s_i,%(S)s)] = 0.0;

			for(int s_j=0; s_j<%(S)s; s_j++){

				vector1[idx2d(t,s_i,%(S)s)] += (x[idx2d(t,s_j,%(S)s)] - mu[idx2d(t,s_j,%(S)s)]) * invcov[idx3d(t,s_j,s_i,%(S)s,%(S)s)];
			}

			vector2[t] += vector1[idx2d(t,s_i,%(S)s)] * (x[idx2d(t,s_i,%(S)s)] - mu[idx2d(t,s_i,%(S)s)]);
		}

		o1[0] += log(sqrtf(invdet[t])) - 0.5 * vector2[t];
	}

	o1[0] = (1/pow(sqrtf(powf(2*pi,%(S)s)),%(T)s)) * exp(o1[0]);

}
"""

#seed(1234)
T = 2
S = 3

kernel_code = kernel_code_template % {'T': T ,'S': S}

mod = compiler.SourceModule(kernel_code)

test = mod.get_function("matrixmult")

input = rand(1,S).astype(float64)


x1 = 3*input
x2 = 6*input

x = concatenate([x1,x2],axis=0)

mu1 = 7*input
mu2 = 4*input

mu = concatenate([mu1,mu2],axis=0)

cov_pre = rand(S,S).astype(float32)

cov_pre1 = 3*cov_pre
cov_pre2 = 4*cov_pre

cov1 = (cov_pre1 + cov_pre1.T)/2
cov2 = (cov_pre2 + cov_pre2.T)/2
invcov1 = linalg.inv(cov1)
invcov2 = linalg.inv(cov2)

invcov = stack([invcov1,invcov2], axis=0)

invdet = array([abs(linalg.det(invcov1)),abs(linalg.det(invcov2))]).astype(float32)

o1 = zeros((1,1)).astype(float64)

print "x.shape", x.shape
print "mu.shape", mu.shape
print "invcov.shape", invcov.shape

test(float32(math.pi), driver.In(invdet), driver.In(x),driver.In(mu), driver.In(invcov),driver.Out(o1), block=(int(1),int(1),int(1)),grid=(int(1),int(1),int(1)))

numpy1 = (sqrt(invdet[0])/sqrt(pow(2*pi,S)))*exp(-0.5*dot(dot(x[0,:]-mu[0,:],invcov[0,:,:]),transpose(x[0,:]-mu[0,:])))
numpy2 = (sqrt(invdet[1])/sqrt(pow(2*pi,S)))*exp(-0.5*dot(dot(x[1,:]-mu[1,:],invcov[1,:,:]),transpose(x[1,:]-mu[1,:])))

numpy = numpy1 * numpy2

print [linalg.det(invcov1), linalg.det(invcov2)]

print "CPU OUT"
print numpy, [numpy1,numpy2]
print ""
print "GPU OUT"
print o1[0][0]
print ""
print "Error:", (o1[0][0]-numpy)/((o1[0][0]+numpy)/2)
print "Registers:", test.num_regs
