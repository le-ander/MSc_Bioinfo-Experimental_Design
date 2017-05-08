from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit

import sys
import operator

#from mutInfos import launch
from peitho.mut_Info.mutInfos import launch


random.seed(123)
N1 = 7
B = 4
N3 = 4
T = 50
S = 7

data_t = random.rand(N1,B,T,S).astype(float64)
theta_t = random.rand(N1+N3,T,S).astype(float64)
cov_t = random.rand(N1+N3,S*T,S).astype(float64)

##Check input types!!



def mutInfo1SDE(data,theta,cov):

	kernel_code_template = """

	//Function to index 4-dimensional flattened arrays
	__device__ unsigned int idx4d(int i, int j, int k, int l, int B, int T, int S)
	{
		return i*B*T*S + j*T*S + k*S + l;
	}

	//Function to index 3-dimensional flattened arrays
	__device__ unsigned int idx3d(int i, int k, int l, int T, int S)
	{
		return i*T*S + k*S + l;
	}

	//Function to index 2-dimensional flattened arrays
	__device__ unsigned int idx2d(int i, int j, int S)
	{
		return i*S + j;
	}

	//Function to calculate intemediary probabilities for mutual information calculation
	__global__ void kernel_func1SDE(int n1, int b, int n3, double pre, float *invdet, double *x, double *mu, float *invcov, double *res1){

		unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int tk = threadIdx.z + blockDim.z * blockIdx.z;

		if((ti>=n1)||(tj>=b)||(tk>=n3)) return;

		double vector1[%(ST)s] = {0.0};
		double vector2[%(T)s] = {0.0};

		res1[idx3d(ti,tj,tk,b,n3)] = 0.0;

		for(int t=0; t<%(T)s; t++){

			for(int s_i=0; s_i<%(S)s; s_i++){

				vector1[idx2d(t,s_i,%(S)s)] = 0.0;

				for(int s_j=0; s_j<%(S)s; s_j++){

					vector1[idx2d(t,s_i,%(S)s)] += (x[idx4d(ti,tj,t,s_j,b,%(T)s,%(S)s)] - mu[idx3d(tk,t,s_j,%(T)s,%(S)s)]) * invcov[idx4d(tk,t,s_j,s_i,%(T)s,%(S)s,%(S)s)];
				}
				vector2[t] += vector1[idx2d(t,s_i,%(S)s)] * (x[idx4d(ti,tj,t,s_i,b,%(T)s,%(S)s)] - mu[idx3d(tk,t,s_i,%(T)s,%(S)s)]);
			}
			res1[idx3d(ti,tj,tk,b,n3)] += log(sqrtf(invdet[idx2d(tk,t,%(T)s)])) - 0.5 * vector2[t] + pre;
		}
		res1[idx3d(ti,tj,tk,b,n3)] = exp(res1[idx3d(ti,tj,tk,b,n3)]);
	}
	"""

	print "-----Preprocessing Data (matrix inversion etc.)-----"

	# Determine number of particles (N1,N3), betas (B), timepoints (T), species (S)
	N1, B, T, S = data.shape
	N3 = theta.shape[0] - N1

	# Fill placeholders in kernel (pycuda metaprogramming)
	kernel_code = kernel_code_template % {
		'T': T,
		'S': S,
		'ST': S*T
		}

	# Compile GPU kernel
	mod = compiler.SourceModule(kernel_code)

	# Reshape theta array to exclude forst N1 particles
	theta = array(theta)[N1:,:,:]
	cov = array(cov)[N1:,:,:]

	# Create GPU function handle
	gpu_kernel_func1SDE = mod.get_function("kernel_func1SDE")

	# Precalculation for GPU kernel for faster computation
	pre = log(1/(sqrt(pow(2*math.pi,S))))

	# Initialise arrays for inverted covariance matrices and inverted determinants
	invcov = zeros((N3,S*T,S), dtype=float32)
	invdet = zeros((N3,T), dtype=float32)

	# Invert covariance matrices and calculate the determinant of the inverted matrices
	##Remove abs() for invdet
	for i in range(N3):
		for j in range(T):
			invcov[i,j*S:(j+1)*S,:] = linalg.inv(cov[i,j*S:(j+1)*S,:])
			invdet[i,j] = abs(linalg.det(invcov[i,j*S:(j+1)*S,:]))

	######################Calculating second log term###########################

	print "-----Determining optimal kernel launch configuration (for part 1/2)-----"

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, gpu_kernel_func1SDE)
	primes = launch.pFactors(block)
	l1 = int(len(primes)/3)
	l2 = int(len(primes)- 2*l1)
	block_i = reduce(operator.mul, primes[:l2])
	block_j = reduce(operator.mul, primes[l2:l2+l1])
	block_k = reduce(operator.mul, primes[l2+l1:l2+2*l1])
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)+"x"+str(block_k)

	sys.exit()

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid_prelim_i , grid_prelim_j = launch.optimise_gridsize_sde(1, block_i, block_j, T, S)
	grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i, N1))
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, N2))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)
	print "Registers:", gpu_kernel_func1.num_regs , "\n"

	print "-----Calculation part 1 of 2 now running-----"





	res1 = zeros((N1,B,N3), dtype=float64)

	gpu_kernel_func1SDE(int32(N1), int32(B), int32(N3), float64(pre), driver.In(invdet), driver.In(data),driver.In(theta), driver.In(invcov),driver.Out(res1), block=(int(8),int(4),int(4)),grid=(int(1),int(1),int(1)))

	cpu_res = zeros((N1,B,N3), dtype=float64)

	for i in range(N1):
		for j in range(B):
			for k in range(N3):
				for l in range(T):
					cpu_res[i,j,k] += pre + log(sqrt(invdet[k,l])) - 0.5 * dot(dot(expand_dims(data[i,j,l,:]-theta[k,l,:],0),invcov[k,l*S:(l+1)*S,:]),expand_dims(data[i,j,l,:]-theta[k,l,:],1))

				cpu_res[i,j,k] = exp(cpu_res[i,j,k])
	print ""
	print "CPU OUT"
	print cpu_res
	print ""
	print "GPU OUT"
	print res1
	print ""
	print "Error:"
	print divide(res1-cpu_res,res1+cpu_res/2)
	print ""
	print "Registers:", gpu_kernel_func1SDE.num_regs


mutInfo1SDE(data_t,theta_t,cov_t)
