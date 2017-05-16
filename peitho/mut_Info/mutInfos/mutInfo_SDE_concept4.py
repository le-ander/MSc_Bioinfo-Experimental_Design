from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit

import sys
import operator
import copy

# RESET THIS FOR PACKAGE IMPLEMENTATION!!
import launch
#from peitho.mut_Info.mutInfos import launch

## REMOVE SEED
random.seed(123)
N1 = 43
B = 159
N3 = 37
T = 3
S = 3

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

	//Function to calculate sum multivariate gaussians for second log term
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

	//Function to calculate sum multivariate gaussians for first log term
	__global__ void kernel_func2SDE(int n1, int b, double pre, float *invdet, double *x, double *mu, float *invcov, double *res1){

		unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;

		if((ti>=n1)||(tj>=b)) return;

		double vector1[%(ST)s] = {0.0};
		double vector2[%(T)s] = {0.0};

		res1[idx2d(ti,tj,b)] = 0.0;

		for(int t=0; t<%(T)s; t++){

			for(int s_i=0; s_i<%(S)s; s_i++){

				vector1[idx2d(t,s_i,%(S)s)] = 0.0;

				for(int s_j=0; s_j<%(S)s; s_j++){

					vector1[idx2d(t,s_i,%(S)s)] += (x[idx4d(ti,tj,t,s_j,b,%(T)s,%(S)s)] - mu[idx3d(ti,t,s_j,%(T)s,%(S)s)]) * invcov[idx4d(ti,t,s_j,s_i,%(T)s,%(S)s,%(S)s)];
				}
				vector2[t] += vector1[idx2d(t,s_i,%(S)s)] * (x[idx4d(ti,tj,t,s_i,b,%(T)s,%(S)s)] - mu[idx3d(ti,t,s_i,%(T)s,%(S)s)]);
			}
			res1[idx2d(ti,tj,b)] += log(sqrtf(invdet[idx2d(ti,t,%(T)s)])) - 0.5 * vector2[t] + pre;
		}
	}
	"""

	print "\n", "-----Preprocessing Data (matrix inversion etc.)-----", "\n"

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

	# Create GPU function handle
	gpu_kernel_func1SDE = mod.get_function("kernel_func1SDE")
	gpu_kernel_func2SDE = mod.get_function("kernel_func2SDE")

	# Precalculation for GPU kernel for faster computation
	pre = log(1/(sqrt(pow(2*math.pi,S))))

	# Initialise arrays for inverted covariance matrices and inverted determinants
	invcov = zeros((N1+N3,S*T,S), dtype=float32)
	invdet = zeros((N1+N3,T), dtype=float32)

	# Invert covariance matrices and calculate the determinant of the inverted matrices
	##Remove abs() for invdet
	for i in range(N1+N3):
		for j in range(T):
			invcov[i,j*S:(j+1)*S,:] = linalg.inv(cov[i,j*S:(j+1)*S,:])
			invdet[i,j] = abs(linalg.det(invcov[i,j*S:(j+1)*S,:]))

	#################### Calculating second log term ###########################

	print "-----Determining optimal kernel launch configuration (for part 1/2)-----"

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, gpu_kernel_func1SDE)
	primes = launch.pFactors(block)
	l1 = int(len(primes)/3)
	l2 = int(len(primes)- 2*l1)
	block_i = float(reduce(operator.mul, primes[:l2]))
	block_j = float(reduce(operator.mul, primes[l2:l2+l1]))
	block_k = float(reduce(operator.mul, primes[l2+l1:l2+2*l1]))
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)+"x"+str(block_k)


	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid_prelim_i , grid_prelim_j, grid_prelim_k = launch.optimise_gridsize_sde(1, float(block_i), float(block_j), float(block_k), T, S)
	grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i, N1))
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, B))
	grid_k = float(min(autoinit.device.max_grid_dim_y, grid_prelim_k, N3))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)+"x"+str(grid_k)
	print "Registers:", gpu_kernel_func1SDE.num_regs , "\n"

	print "-----Calculation part 1 of 2 now running-----", "\n"

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(B/grid_j))
	numRuns_k = int(ceil(N3/grid_k))

	# Initialise array to hold results of log N3 avarage
	res_log_2 = zeros([N1,B], dtype=float64)

	# Create template array for res1
	try:
		template_res1 = zeros([int(grid_i),int(grid_j),int(grid_k)], dtype=float64)
	except:
		print "ERROR: Not enought memory (RAM) available to create array for GPU results. Reduce GPU grid size."
		sys.exit()

	# Maximum number of particles per run in i, j and k direction
	Ni = int(grid_i)

	# Main nested for-loop for calculation of second log term
	for i in range(numRuns_i):

		# If last run with less that max remaining particles, set Ni to remaining number of particles
		if((int(grid_i)*(i+1)) > N1):
			Ni = int(N1 - grid_i*i)

		# Set i dimension of block and grid for this run
		if(Ni<block_i):
			gi = 1
			bi = Ni
		else:
			gi = ceil(Ni/block_i)
			bi = block_i

		# Maximum number of particles per run in j direction
		Nj = int(grid_j)

		for j in range(numRuns_j):

			# If last run with less that max remaining particles, set Ni to remaining number of particles
			if((int(grid_j)*(j+1)) > B):
				Nj = int(B - grid_j*j)

			# Prepare data that depends on i and j for this run
			data_subset = ascontiguousarray(data[(i*int(grid_i)):(i*int(grid_i)+Ni),(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]) # d1 subunit for the next k runs

			# Set i dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				gj = ceil(Nj/block_j)
				bj = block_j

			# Reset last to "False"
			last = False

			# Maximum number of particles per run in k direction
			Nk = int(grid_k)

			for k in range(numRuns_k):

				# If last run with less that max remaining particles, set Nj to remaining number of particles
				if((int(grid_k)*(k+1)) > N3):
					Nk = int(N3 - grid_k*k)
					last = True

				# Prepare input that depends on k for this run
				theta_subset = theta[N1+(k*int(grid_k)):N1+(k*int(grid_k)+Nk),:,:]
				invcov_subset = invcov[N1+(k*int(grid_k)):N1+(k*int(grid_k)+Nk),:,:]
				invdet_subset = invdet[N1+(k*int(grid_k)):N1+(k*int(grid_k)+Nk),:]

				# Set k dimension of block and grid for this run
				if(Nk<block_k):
					gk = 1
					bk = Nk
				else:
					gk = ceil(Nk/block_k)
					bk = block_k

				# Prepare results array for run
				if last == True:
					res1 = copy.deepcopy(template_res1[:Ni,:Nj,:Nk])
				elif k == 0:
					res1 = copy.deepcopy(template_res1[:Ni,:Nj,:Nk])

				# Call GPU kernel function
				gpu_kernel_func1SDE(int32(Ni), int32(Nj), int32(Nk), float64(pre), driver.In(invdet_subset), driver.In(data_subset),driver.In(theta_subset), driver.In(invcov_subset),driver.Out(res1), block=(int(bi),int(bj),int(bk)),grid=(int(gi),int(gj),int(gk)))

				# Sum over N3 and sore in results array
				res_log_2[i*int(grid_i):i*int(grid_i)+Ni,j*int(grid_j):j*int(grid_j)+Nj] += sum(res1, axis=2)

	# Divide by N3 and take log
	res_log_2 = log(res_log_2) - log(N3)


	#################### Calculating first log term ###########################

	print "-----Determining optimal kernel launch configuration (for part 2/2)-----"

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, gpu_kernel_func2SDE)
	block_i = launch.factor_partial(block)
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)


	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid_prelim_i , grid_prelim_j = launch.optimise_gridsize_sde(2, block_i, block_j, 0, T, S)[0:2]
	grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i, N1))
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, B))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)
	print "Registers:", gpu_kernel_func2SDE.num_regs , "\n"

	print "-----Calculation part 2 of 2 now running-----"

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(B/grid_j))

	# Initialise array to hold results of second log term
	res_log_1 = zeros([N1,B], dtype=float64)

	# Create template array for res1
	try:
		template_res1 = zeros([int(grid_i),int(grid_j)], dtype=float64)
	except:
		print "ERROR: Not enought memory (RAM) available to create array for GPU results. Reduce GPU grid size."
		sys.exit()

	# Maximum number of particles per run in i, j and k direction
	Ni = int(grid_i)

	# Main nested for-loop for calculation of second log term
	for i in range(numRuns_i):

		# If last run with less that max remaining particles, set Ni to remaining number of particles
		if((int(grid_i)*(i+1)) > N1):
			Ni = int(N1 - grid_i*i)

		# Prepare input that depends on i for this run
		theta_subset = theta[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:]
		invcov_subset = invcov[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:]
		invdet_subset = invdet[(i*int(grid_i)):(i*int(grid_i)+Ni),:]

		# Set i dimension of block and grid for this run
		if(Ni<block_i):
			gi = 1
			bi = Ni
		else:
			gi = ceil(Ni/block_i)
			bi = block_i

		# Maximum number of particles per run in j direction
		Nj = int(grid_j)

		# Reset last to "False"
		last = False

		for j in range(numRuns_j):

			# If last run with less that max remaining particles, set Ni to remaining number of particles
			if((int(grid_j)*(j+1)) > B):
				Nj = int(B - grid_j*j)
				last = True

			# Prepare data that depends on i and j for this run
			data_subset = ascontiguousarray(data[(i*int(grid_i)):(i*int(grid_i)+Ni),(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]) # d1 subunit for the next k runs

			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				gj = ceil(Nj/block_j)
				bj = block_j

			# Prepare results array for run
			if last == True:
				res1 = copy.deepcopy(template_res1[:Ni,:Nj])
			elif j == 0:
				res1 = copy.deepcopy(template_res1[:Ni,:Nj])

			# Call GPU kernel function
			gpu_kernel_func2SDE(int32(Ni), int32(Nj), float64(pre), driver.In(invdet_subset), driver.In(data_subset),driver.In(theta_subset), driver.In(invcov_subset),driver.Out(res1), block=(int(bi),int(bj),int(1)),grid=(int(gi),int(gj),int(1)))

			# Store in results array
			res_log_1[i*int(grid_i):i*int(grid_i)+Ni,j*int(grid_j):j*int(grid_j)+Nj] = res1

	# Calculate final result
	mutinfo = sum(res_log_1 - res_log_2) / (N1 * B)
	print mutinfo
	print "\n", "------CPU CALCS RUNNING NOW---------"
###############################CPU TEST#########################################
	cpu_log2 = zeros((N1,B,N3), dtype=float64)
	cpu_log1 = zeros((N1,B), dtype=float64)

	for i in range(N1):
		for j in range(B):
			for k in range(N3):
				for l in range(T):
					cpu_log2[i,j,k] += pre + log(sqrt(invdet[N1+k,l])) - 0.5 * dot(dot(expand_dims(data[i,j,l,:]-theta[N1+k,l,:],0),invcov[N1+k,l*S:(l+1)*S,:]),expand_dims(data[i,j,l,:]-theta[N1+k,l,:],1))

	cpu_log2 = exp(cpu_log2)
	cpu_log2 = sum(cpu_log2, axis=2)
	cpu_log2 = log(cpu_log2) - log(N3)

	for i in range(N1):
		for j in range(B):
			for l in range(T):
				cpu_log1[i,j] += pre + log(sqrt(invdet[i,l])) - 0.5 * dot(dot(expand_dims(data[i,j,l,:]-theta[i,l,:],0),invcov[i,l*S:(l+1)*S,:]),expand_dims(data[i,j,l,:]-theta[i,l,:],1))

	cpu = sum(cpu_log1 - cpu_log2) / (N1 * B)
###############################CPU TEST#########################################

	return mutinfo, cpu

gpu, cpu = mutInfo1SDE(data_t,theta_t,cov_t)

print ""
print "CPU OUT"
print cpu
print ""
print "GPU OUT"
print gpu
print ""
print "Error:"
print cpu - gpu
