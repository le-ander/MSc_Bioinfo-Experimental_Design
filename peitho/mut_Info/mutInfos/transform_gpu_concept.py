from numpy import *
from numpy.random import *

from pycuda import compiler, driver
from pycuda import autoinit


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

__global__ void fit_x_func(float *b_mat, double *x_mat, double *fit_res_x){

	unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;

	if((ti>=%(N)s)||(tj>=%(B)s)) return;

	for(int i=0; i<%(T)s; i++){

		for(int j=0; j<%(A)s; j++){

			fit_res_x[idx4d(ti,tj,i,j,%(B)s,%(T)s,%(A)s)] = 0.0;

			for (int k=0; k<%(S)s; k++){

				fit_res_x[idx4d(ti,tj,i,j,%(B)s,%(T)s,%(A)s)] += b_mat[idx2d(j, k, %(S)s)] * x_mat[idx4d(ti,tj,i,k,%(B)s,%(T)s,%(S)s)];
			}
		}
	}
}

__global__ void fit_mu_func(float *b_mat, double *mu_mat, double *fit_res_mu){

	unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;

	if((ti>=%(N)s)) return;

	for(int i=0; i<%(T)s; i++){

		for(int j=0; j<%(A)s; j++){

			fit_res_mu[idx3d(ti,i,j,%(T)s,%(A)s)] = 0.0;

			for (int k=0; k<%(S)s; k++){

				fit_res_mu[idx3d(ti,i,j,%(T)s,%(A)s)] += b_mat[idx2d(j, k, %(S)s)] * mu_mat[idx3d(ti,i,k,%(T)s,%(S)s)];
			}
		}
	}
}


__global__ void fit_cov_func(float *b_mat, double *cov_mat, double *fit_res_cov){

	unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;

	if((ti>=%(N)s)||(tj>=%(T)s)) return;

	double vector1[%(AS)s]={0.0};


	for(int i=0; i<%(S)s; i++){

		for(int j=0; j<%(A)s; j++){

			for (int k=0; k<%(S)s; k++){

				vector1[idx2d(j,i,%(S)s)] += b_mat[idx2d(j,k,%(S)s)] * cov_mat[idx4d(ti,tj,k,i,%(T)s,%(S)s,%(S)s)];

			}
		}
	}


	for(int i=0; i<%(A)s; i++){

		for(int j=0; j<%(A)s; j++){

		fit_res_cov[idx4d(ti,tj,i,j,%(T)s,%(A)s,%(A)s)] = 0.0;

			for (int k=0; k<%(S)s; k++){

				fit_res_cov[idx4d(ti,tj,i,j,%(T)s,%(A)s,%(A)s)] += vector1[idx2d(i,k,%(S)s)] * b_mat[idx2d(j,k,%(S)s)];

			}
		}
	}
}
"""

seed(123)
N = 1017
B = 507
T = 3
S = 5
A = 2

kernel_code = kernel_code_template % {
	'N': N,
	'B': B,
	'T': T,
	'A': A,
	'S': S,
	'AS': A*S
	}

mod = compiler.SourceModule(kernel_code)

x_test = mod.get_function("fit_x_func")
mu_test = mod.get_function("fit_mu_func")
cov_test = mod.get_function("fit_cov_func")


b_mat = rand(A,S).astype(float32)
x_mat = rand(N,B,T,S).astype(float64)
mu_mat = rand(N,T,S).astype(float64)
cov_mat = rand(N,S*T,S).astype(float64)
fit_res_x = zeros((N,B,T,A)).astype(float64)
fit_res_mu = zeros((N,T,A)).astype(float64)
fit_res_cov = zeros((N,T*A,A)).astype(float64)

#x_test(driver.In(b_mat),driver.In(x_mat),driver.Out(fit_res_x),block=(int(1),int(1),int(1)),grid=(int(N),int(B),int(1)))
#mu_test(driver.In(b_mat),driver.In(mu_mat),driver.Out(fit_res_mu),block=(int(1),int(1),int(1)),grid=(int(N),int(1),int(1)))
cov_test(driver.In(b_mat),driver.In(cov_mat),driver.Out(fit_res_cov),block=(int(1),int(1),int(1)),grid=(int(N),int(T),int(1)))


u = 0
q = 0
p = q*T
w = p+S
o = p+A

print "SHAPE", cov_mat[u,p:w,:].shape

numpy_x = transpose(dot(b_mat,transpose(x_mat[u,q,:,:])))
numpy_mu = transpose(dot(b_mat,transpose(mu_mat[u,:,:])))
numpy_cov = dot(dot(b_mat,cov_mat[u,p:w,:]),transpose(b_mat))

print b_mat
print cov_mat[0]
print "CPU OUT"
#print numpy_x
#print numpy_mu
print numpy_cov
print ""
print "GPU OUT"
#print fit_res_x[u,q,:,:]
#print fit_res_mu[u,:,:]
print fit_res_cov[u,p:o,:]
#print "error", (fit_res_x-numpy_x)/((fit_res_x+numpy_x)/2)
#print "error", (fit_res_mu-numpy_mu)/((fit_res_mu+numpy_mu)/2)

#print subtract(numpy_x,fit_res_x[u,q,:,:])
