from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit

import sys

from mutInfos import launch
import copy


#random.seed(123)
N1 = 201
N2 = 240
B = 107
T = 251
S = 3
A = 2

trans_mat = random.rand(A,S).astype(float32)
data = random.rand(N1,B,T,S).astype(float64)
theta = random.rand(N1+N2,T,S).astype(float64)
cov = random.rand(N1+N2,S*T,S).astype(float64)

##CHECK INPUT TYPES!!


def transform_gpu(data,theta,cov,trans_mat):

	kernel_code_template = """

	//Function to index 4-dimensional flattened arrays
	__device__ unsigned int idx4d(int i, int j, int k, int l, int B, int T, int S)
	{
		return i*T*S*B + j*T*S + k*S + l;
	}

	//Function to index 3-dimensional flattened arrays
	__device__ unsigned int idx3d(int i, int k, int l, int T, int S)
	{
		return i*T*S + k*S + l;
	}

	//Function to index 2-dimensional flattened arrays
	__device__ unsigned int idx2d(int i, int j, int T)
	{
		return i*T + j;
	}

	//Function for affine transformation of data array
	__global__ void data_func(int n, int b, float *trans_mat, double *data, double *resgpu){

		unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;

		if((ti>=n)||(tj>=b)) return;

		for(int i=0; i<%(T)s; i++){

			for(int j=0; j<%(A)s; j++){

				resgpu[idx4d(ti,tj,i,j,b,%(T)s,%(A)s)] = 0.0;

				for (int k=0; k<%(S)s; k++){

					resgpu[idx4d(ti,tj,i,j,b,%(T)s,%(A)s)] += trans_mat[idx2d(j, k, %(S)s)] * data[idx4d(ti,tj,i,k,b,%(T)s,%(S)s)];
				}
			}
		}
	}

	//Function for affine transformation of theta array
	__global__ void theta_func(int n, float *trans_mat, double *theta, double *resgpu){

		unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;

		if(ti>=n) return;

		for(int i=0; i<%(T)s; i++){

			for(int j=0; j<%(A)s; j++){

				resgpu[idx3d(ti,i,j,%(T)s,%(A)s)] = 0.0;

				for (int k=0; k<%(S)s; k++){

					resgpu[idx3d(ti,i,j,%(T)s,%(A)s)] += trans_mat[idx2d(j, k, %(S)s)] * theta[idx3d(ti,i,k,%(T)s,%(S)s)];
				}
			}
		}
	}


	//Function for affine transformation of cov array
	__global__ void cov_func(int n, int t, float *trans_mat, double *cov, double *res_cov){

		unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;

		if((ti>=n)||(tj>=t)) return;

		double vector1[%(AS)s]={0.0};


		for(int i=0; i<%(S)s; i++){

			for(int j=0; j<%(A)s; j++){

				for (int k=0; k<%(S)s; k++){

					vector1[idx2d(j,i,%(S)s)] += trans_mat[idx2d(j,k,%(S)s)] * cov[idx4d(ti,tj,k,i,t,%(S)s,%(S)s)];

				}
			}
		}


		for(int i=0; i<%(A)s; i++){

			for(int j=0; j<%(A)s; j++){

			res_cov[idx4d(ti,tj,i,j,%(T)s,%(A)s,%(A)s)] = 0.0;

				for (int k=0; k<%(S)s; k++){

					res_cov[idx4d(ti,tj,i,j,%(T)s,%(A)s,%(A)s)] += vector1[idx2d(i,k,%(S)s)] * trans_mat[idx2d(j,k,%(S)s)];

				}
			}
		}
	}
	"""

	# Determine number of particles (N), betas (B), timepoints (T), species (S) and transformed species (A)
	N,B,T,S = data.shape
	A = trans_mat.shape[0]

	# Fill placeholders in kernel (pycuda metaprogramming)
	kernel_code = kernel_code_template % {
		'T': T,
		'A': A,
		'S': S,
		'AS': A*S
		}

	# Compile GPU kernel
	mod = compiler.SourceModule(kernel_code)

########################Data Transformation############################################

	# Create kernel function handle
	data_gpu = mod.get_function("data_func")

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, data_gpu)
	block_y = launch.factor_partial(block) # Maximum threads per block
	block_x = block / block_y
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_x)+"x"+str(block_y)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multiple of block size)
	grid_prelim_x , grid_prelim_y, grid = launch.optimise_gridsize_trans(1, block_x, block_y, T, S, A)
	grid_x = float(min(autoinit.device.max_grid_dim_x, grid_prelim_x))
	grid_y = float(min(autoinit.device.max_grid_dim_y, grid_prelim_y))
	# If gridsize in one dimention too large, reshape grid to allow more threads in the second dimension
	if launch.round_up(B,block_y) < min(autoinit.device.max_grid_dim_y,grid_y):
		grid_y = float(B)
		grid_x = float(min(autoinit.device.max_grid_dim_x, launch.round_down(grid/grid_y,block_x),N))
	elif launch.round_up(N,block_x) < min(autoinit.device.max_grid_dim_x,grid_x):
		grid_x = float(N)
		grid_y = float(min(autoinit.device.max_grid_dim_y, launch.round_down(grid/grid_x,block_y),B))
	print "Grid shape:", str(grid_x)+"x"+str(grid_y), grid_x*grid_y
	print "Registers:", data_gpu.num_regs, "\n"


	# Determine required number of runs for i and j
	numRuns_N = int(ceil(N/grid_x))
	numRuns_B = int(ceil(B/grid_y))

	print numRuns_N, numRuns_B

	# Maximum number of particles per run in x dimension
	n = int(grid_x)

	# Main nested for-loop for the transormation of data
	for i in range(numRuns_N):
		# If last run with less that max remaining particles, set n to remaining number of particles
		if((int(grid_x)*(i+1)) > N):
			n = int(N - grid_x*i)

		# Set i dimension of block and grid for this run
		if(n<block_x):
			gx = 1
			bx = n
		else:
			gx = ceil(n/block_x)
			bx = block_x

		b = int(grid_y)

		for j in range(numRuns_B):
			# If last run with less that max remaining particles, set b to remaining number of betas
			if((int(grid_y)*(j+1)) > B):
				b = int(B - grid_y*j)

			# Prepare data for this run
			in_mat = copy.deepcopy(data[(i*int(grid_x)):(i*int(grid_x)+n),(j*int(grid_y)):(j*int(grid_y)+b),:,:])

			# Set j dimension of block and grid for this run
			if(b<block_y):
				gy = 1
				by = b
			else:
				gy = ceil(b/block_y)
				by = block_y

			# Create array to take results
			resgpu = zeros((int(n),int(b),T,A)).astype(float64)

			# Call GPU kernel functions
			data_gpu(int32(n), int32(b), driver.In(trans_mat), driver.In(in_mat), driver.Out(resgpu), block=(int(bx),int(by),1), grid=(int(gx),int(gy),1))

			# Concatenate arrays for all B
			if j == 0:
				resj = copy.deepcopy(resgpu)
			else:
				resj = concatenate((resj,resgpu), axis=1)

		# Concatenate arrays for all N
		if i == 0:
			result_data = copy.deepcopy(resj)
		else:
			result_data = concatenate((result_data, resj), axis=0)



	# Update N for theta and cov Transformation
	N = theta.shape[0]

########################Theta Transformation ############################################

	# Create kernel function handle
	theta_gpu = mod.get_function("theta_func")

	# Launch configuration: Block size and shape (as close to square as possible)
	block_x = launch.optimal_blocksize(autoinit.device, theta_gpu)
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block)+"x1.0"

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multiple of block size)
	grid_prelim_x = launch.optimise_gridsize_trans(2, block_x, 1.0, T, S, A)[0]
	grid_x = float(min(autoinit.device.max_grid_dim_x, grid_prelim_x, N))
	print "Grid shape:", str(grid_x)+"x1.0"
	print "Registers:", data_gpu.num_regs, "\n"


	# Determine required number of runs for i and j
	numRuns_N = int(ceil(N/grid_x))

	print numRuns_N

	# Maximum number of particles per run in x dimension
	n = int(grid_x)

	# Main nested for-loop for the transormation of theta
	for i in range(numRuns_N):
		# If last run with less that max remaining particles, set n to remaining number of particles
		if((int(grid_x)*(i+1)) > N):
			n = int(N - grid_x*i)

		# Set i dimension of block and grid for this run
		if(n<block_x):
			gx = 1
			bx = n
		else:
			gx = ceil(n/block_x)
			bx = block_x

		# Prepare data for this run
		in_mat = copy.deepcopy(theta[(i*int(grid_x)):(i*int(grid_x)+n),:,:])

		# Create array to take results
		resgpu = zeros((int(n),T,A)).astype(float64)

		# Call GPU kernel functions
		theta_gpu(int32(n), driver.In(trans_mat), driver.In(in_mat), driver.Out(resgpu), block=(int(bx),1,1), grid=(int(gx),1,1))

		# Concatenate arrays for all N
		if i == 0:
			result_theta = copy.deepcopy(resgpu)
		else:
			result_theta = concatenate((result_theta, resgpu), axis=0)


########################Covariance Transformation############################################

	# Create kernel function handle
	cov_gpu = mod.get_function("cov_func")

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, theta_gpu)
	block_y = launch.factor_partial(block) # Maximum threads per block
	block_x = block / block_y
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_x)+"x"+str(block_y)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multiple of block size)
	grid_prelim_x , grid_prelim_y, grid = launch.optimise_gridsize_trans(3, block_x, block_y, T, S, A)
	grid_x = float(min(autoinit.device.max_grid_dim_x, grid_prelim_x))
	grid_y = float(min(autoinit.device.max_grid_dim_y, grid_prelim_y))
	# If gridsize in one dimention too large, reshape grid to allow more threads in the second dimension
	if launch.round_up(T,block_y) < min(autoinit.device.max_grid_dim_y,grid_y):
		grid_y = float(T)
		grid_x = float(min(autoinit.device.max_grid_dim_x, launch.round_down(grid/grid_y,block_x),N))
	elif launch.round_up(N,block_x) < min(autoinit.device.max_grid_dim_x,grid_x):
		grid_x = float(N)
		grid_y = float(min(autoinit.device.max_grid_dim_y, launch.round_down(grid/grid_x,block_y),T))
	print "Grid shape:", str(grid_x)+"x"+str(grid_y), grid_x*grid_y
	print "Registers:", data_gpu.num_regs, "\n"


	# Determine required number of runs for i and j
	numRuns_N = int(ceil(N/grid_x))
	numRuns_T = int(ceil(T/grid_y))

	print numRuns_N, numRuns_T

	# Maximum number of particles per run in x dimension
	n = int(grid_x)

	# Main nested for-loop for the transormation of data
	for i in range(numRuns_N):
		# If last run with less that max remaining particles, set n to remaining number of particles
		if((int(grid_x)*(i+1)) > N):
			n = int(N - grid_x*i)

		# Set i dimension of block and grid for this run
		if(n<block_x):
			gx = 1
			bx = n
		else:
			gx = ceil(n/block_x)
			bx = block_x

		t = int(grid_y)

		for j in range(numRuns_T):
			# If last run with less that max remaining particles, set t to remaining number of timepoints
			if((int(grid_y)*(j+1)) > T):
				t = int(t - grid_y*j)

			# Prepare data for this run
			in_mat = copy.deepcopy(cov[(i*int(grid_x)):(i*int(grid_x)+n),(j*int(grid_y)*S):(j*int(grid_y)*S+t*S),:])

			# Set j dimension of block and grid for this run
			if(t<block_y):
				gy = 1
				by = t
			else:
				gy = ceil(t/block_y)
				by = block_y

			# Create array to take results
			resgpu = zeros((int(n),int(A*t),A)).astype(float64)

			# Call GPU kernel functions
			cov_gpu(int32(n), int32(t), driver.In(trans_mat), driver.In(in_mat), driver.Out(resgpu), block=(int(bx),int(by),1), grid=(int(gx),int(gy),1))

			# Concatenate arrays for all t
			if j == 0:
				resj = copy.deepcopy(resgpu)

			else:
				resj = concatenate((resj,resgpu), axis=1)

		# Concatenate arrays for all N
		if i == 0:
			result_cov = copy.deepcopy(resj)
		else:
			result_cov = concatenate((result_cov, resj), axis=0)



	return result_data, result_theta , result_cov

################################################################################################################################

gpu=transform_gpu(data,theta,cov,trans_mat)

print gpu[0].shape
print gpu[1].shape
print gpu[2].shape


########TEST NUMPY DATA#######

for i in range(data.shape[0]):
	for j in range(data.shape[1]):
		resg=dot(data[i,j,:,:],swapaxes(trans_mat,0,1))

		if j==0:
			resj=copy.deepcopy(resg)
		elif j==1:
			resj=stack((resj,resg), axis=0)
		else:
			resj=concatenate((resj,expand_dims(resg, 0)),axis=0)
	if i==0:
		np_data=copy.deepcopy(resj)
	elif i==1:
		np_data=stack((np_data,resj),axis=0)
	else:
		np_data=concatenate((np_data, expand_dims(resj, 0)), axis=0)

print sum(subtract(gpu[0], np_data))

#print gpu[0][5,7,:,:]
#print np_data[5,7,:,:]


########TEST NUMPY THETA#######

for i in range(theta.shape[0]):
	resg=dot(theta[i,:,:],swapaxes(trans_mat,0,1))

	if i==0:
		np_theta=copy.deepcopy(resg)
	elif i==1:
		np_theta=stack((np_theta,resg),axis=0)
	else:
		np_theta=concatenate((np_theta, expand_dims(resg, 0)), axis=0)

print sum(subtract(gpu[1], np_theta))

#print gpu[1][5,:,:]
#print np_theta[5,:,:]


########TEST NUMPY COV#######

for i in range(cov.shape[0]):
	for j in range(data.shape[2]):
		resg=dot(dot(trans_mat,cov[i,j*S:j*S+S,:]),swapaxes(trans_mat,0,1))

		if j==0:
			resj=copy.deepcopy(resg)
		else:
			resj=concatenate((resj,resg),axis=0)

	if i==0:
		np_cov=copy.deepcopy(resj)
	elif i==1:
		np_cov=stack((np_cov,resj),axis=0)
	else:
		np_cov=concatenate((np_cov, expand_dims(resj, 0)), axis=0)

print sum(subtract(gpu[2], np_cov))

#print gpu[2][0,0:2,:]
#print np_cov[0,0:2,:]
