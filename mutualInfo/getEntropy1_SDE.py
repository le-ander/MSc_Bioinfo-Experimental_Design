from numpy import *
from numpy.random import *

from pycuda import compiler, driver
from pycuda import autoinit

import math


kernel_code_template = """
__device__ unsigned int idx3d(int i, int k, int l, int M, int P)
{
	return i*M*P + k*P + l;
}

__device__ unsigned int idx2d(int i, int j, int M)
{
	return i*M + j;
}

__global__ void matrixmult(float pi, float *invdet, double *x, double *mu, float *invcov, double *o1){

	unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;

	double vector1[%(P)s]={0.0};
	double vector2[%(M)s]={0.0};

	o1[0] = 0.0;

	for(int m=0; m<%(M)s; m++){
		for(int p_i=0; p_i<%(P)s; p_i++){
			for(int p_j=0; p_j<%(P)s; p_j++){

				vector1[idx2d(m,p_i,%(P)s)] += (x[idx2d(m,p_j,%(P)s)] - mu[idx2d(m,p_j,%(P)s)]) * invcov[idx3d(m,p_j,p_i,%(P)s,%(P)s)];
			}

			vector2[m] += vector1[idx2d(m,p_i,%(P)s)] * (x[idx2d(m,p_i,%(P)s)] - mu[idx2d(m,p_i,%(P)s)]);
		}

		o1[0] += log(sqrtf(invdet[m])) - 0.5 * vector2[m];
	}

	o1[0] = (1/pow(sqrtf(powf(2*pi,%(P)s)),%(M)s)) * exp(o1[0]);  /*MOVE 1ST PART TO CPU???*/

}
"""
seed(1234)
M = 2
P = 10

kernel_code = kernel_code_template % {'M': M ,'P': P}

mod = compiler.SourceModule(kernel_code)

test = mod.get_function("matrixmult")

input = rand(1,P).astype(float64)


x1 = 2*input
x2 = 4*input

x = concatenate([x1,x2],axis=0)

mu1 = 3*input
mu2 = 5*input

mu = concatenate([mu1,mu2],axis=0)

cov_pre1 = rand(P,P).astype(float32)
cov_pre2 = rand(P,P).astype(float32)
cov1 = (cov_pre1 + cov_pre1.T)/2
cov2 = (cov_pre2 + cov_pre2.T)/2
invcov1 = linalg.inv(cov1)
invcov2 = linalg.inv(cov2)

invcov = stack([invcov1,invcov2], axis=0)

invdet = array([abs(linalg.det(invcov1)),abs(linalg.det(invcov2))]).astype(float32)

o1 = zeros((1,1)).astype(float64)

print x
print mu
print invcov

test(float32(math.pi), driver.In(invdet), driver.In(x),driver.In(mu), driver.In(invcov),driver.Out(o1), block=(int(1),int(1),int(1)),grid=(int(1),int(1),int(1)))

numpy1 = (sqrt(invdet[0])/sqrt(pow(2*pi,P)))*exp(-0.5*dot(dot(x[0,:]-mu[0,:],invcov[0,:,:]),transpose(x[0,:]-mu[0,:])))
numpy2 = (sqrt(invdet[1])/sqrt(pow(2*pi,P)))*exp(-0.5*dot(dot(x[1,:]-mu[1,:],invcov[1,:,:]),transpose(x[1,:]-mu[1,:])))

numpy = numpy1 * numpy2

print [linalg.det(invcov1), linalg.det(invcov2)]

print "CPU OUT"
print numpy, [numpy1,numpy2]
print ""
print "GPU OUT"
print o1[0][0]
