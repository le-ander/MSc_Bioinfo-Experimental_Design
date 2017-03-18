from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit
import copy

import launch, sys

def odd_num(x):
    temp = []
    pos=0
    while x > 1:
        if x%2 ==1:
            temp.append(x)
        x = x >> 1
    return asarray(temp).astype(int32)

# A function to calculate the mutual information between a subset of parameters of a system and an experiment
##(gets called by run_getEntropy2)
##Arguments:
##data - array of tracjectories with noise added
##theta - array of trajectories without noise
##N1,N2,N3 - Number of particles
##sigma - stadard deviation
##scale - scaling constant to prevent nans and infs
#@profile
def getEntropy2(data,theta,N1,N2,N3,sigma,scale):
	# Kernel declaration using pycuda SourceModule

	mod = compiler.SourceModule("""
	__device__ unsigned int idx3d(int i, int k, int l, int M, int P)
	{
		return k*P + i*M*P + l;
	}

	__device__ unsigned int idx2d(int i, int j, int M)
	{
		return i*M + j;
	}


	__global__ void distance1(int len_odd, int* odd_nums, int Ni, int Nj, int M, int P, float sigma, double scale, double *d1, double *d2, double *res1)
	{
		extern __shared__ double s[];

		unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int tid = threadIdx.y;

		s[idx2d(threadIdx.x,tid,blockDim.y)] = 0.0;

		if((i>=Ni)||(j>=Nj)) return;

		for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				s[idx2d(threadIdx.x,tid,blockDim.y)] -= ( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)]);
			}
		}

		s[idx2d(threadIdx.x,tid,blockDim.y)] =  exp(scale + sigma*s[idx2d(threadIdx.x,tid,blockDim.y)]);
		__syncthreads();

        for(unsigned int k=blockDim.y/2; k>0; k>>=1){
            if(tid < k){
                s[idx2d(threadIdx.x,tid,blockDim.y)] += s[idx2d(threadIdx.x,tid+k,blockDim.y)];
            }
            __syncthreads();
        }

        if(len_odd != -1){
	        for(unsigned int l=0; l<len_odd; l+=1){
	            if (tid == 0) s[idx2d(threadIdx.x,0,blockDim.y)] += s[idx2d(threadIdx.x, odd_nums[l]-1 ,blockDim.y)];
	            __syncthreads();
	        }
	    }

		if (tid==0) res1[idx2d(i,blockIdx.y,gridDim.y)] = s[idx2d(threadIdx.x,0,blockDim.y)];

	}

	__global__ void distance2(int len_odd, int* odd_nums, int Nj, int M, int P, float sigma, double scale, double *d1, double *d3, double *res2)
	{
		extern __shared__ double s[];
		int j = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int tid = threadIdx.x;

		s[tid] = 0.0;

		if(j>=Nj) return;

		for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				s[tid] -= (d3[idx3d(j,k,l,M,P)]-d1[idx2d(k,l,P)])*(d3[idx3d(j,k,l,M,P)]-d1[idx2d(k,l,P)]);
			}
		}

		s[tid] =  exp(scale + sigma*s[tid]);
		__syncthreads();

		for(unsigned int k=blockDim.x/2; k>0; k>>=1){
            if(tid < k){
                s[tid] += s[tid+k];
            }
            __syncthreads();
        }

        if(len_odd != -1){
	        for(unsigned int l=0; l<len_odd; l+=1){
	            if (tid == 0) s[0] += s[odd_nums[l]-1];
	            __syncthreads();
	        }
	    }

	    if (tid==0) res2[blockIdx.x] = s[0];
		
	}

	""")


########################Calulation 1############################################

	# Creating handle for global kernel function
	dist_gpu1 = mod.get_function("distance1")

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, dist_gpu1)
	block_i = launch.factor_partial(block) # Maximum threads per block
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid = launch.optimise_gridsize(8.59)
	grid_prelim_i = launch.round_down(sqrt(grid),block_i)
	grid_prelim_j = launch.round_down(grid/grid_prelim_i,block_j)
	# If gridsize in one dimention too large, reshape grid to allow more threads in the second dimension
	if N1 < grid_prelim_i:
		grid_i = float(min(autoinit.device.max_grid_dim_x,N1))
		grid_j = float(min(autoinit.device.max_grid_dim_y, launch.round_down(grid/grid_i,block_j)))
	elif N2 < grid_prelim_j:
		grid_j = float(min(autoinit.device.max_grid_dim_y,N2))
		grid_i = float(min(autoinit.device.max_grid_dim_x, launch.round_down(grid/grid_j,block_i)))
	else:
		grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i))
		grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j))
	print "Maximum gridsize:", grid, "threads"
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N2/grid_j))

	# Prepare input data
	d1 = data[0:N1,:,:].astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	# Determine number of timepoints (M) and number of species (P)
	M = d1.shape[1]
	P = d1.shape[2]

	#Initialize array for results
	result = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i direction
	Ni = int(grid_i)

	# Maximum number of particles per run in j direction
	Nj = int(grid_j)
	
	# Determine M*P*log(scale) for GPU calculations
	mplogscale= M*P*log(scale)

	# Determine 1/(2*sigma*sigma) for
	sigmasq_inv = 1/(2*sigma*sigma)

	# Create template array for res1
	temp_res1 = zeros([Ni,Nj]).astype(float64)
	

	##################################################
	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):
		#print "Runs left:", numRuns_i - i

		# If last run with less that max remaining particles, set Ni to remaining number of particles
		if((int(grid_i)*(i+1)) > N1):
			Ni = int(N1 - grid_i*i)

		# Prepare data that depends on i for this run
		data1 = d1[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:] # d1 subunit for the next j runs

		# Set i dimension of block and grid for this run
		if(Ni<block_i):
			gi = 1  # Grid size in dim i
			bi = Ni # Block size in dim i
		else:
			gi = ceil(Ni/block_i)
			bi = block_i

		# Maximum number of particles per run in j direction
		Nj = int(grid_j)

		# Resets last to "False"
		last = False

		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N2):
				Nj = int(N2 - grid_j*j)
				last = True


			# Prepare data that depends on j for this run
			data2 = d2[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]


			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				gj = ceil(Nj/block_j)
				bj = block_j


			# Prepare results array for this run
			if last == True:
				res1 = copy.deepcopy(temp_res1[:Ni,:int(gj)])
			elif j==0:
				res1 = copy.deepcopy(temp_res1[:Ni,:int(gj)])


			iterations = odd_num(int(bj))
			if iterations.size == 0:
				#print "here"
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				#print "here2"
				temp_1 = iterations.size

			# Call GPU kernel function
			dist_gpu1(int32(temp_1), driver.In(iterations), int32(Ni),int32(Nj), int32(M), int32(P), float32(sigmasq_inv), float64(mplogscale), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)), shared = 3000)

			# Summing rows in GPU output for this run
			result[i*int(grid_i):i*int(grid_i)+Ni,j]=sum(res1, axis=1)


	# Sum all content of new results matrix and add/subtract constants for each row if there are no NANs or infs
	sum_result1=ma.log(sum(result,axis=1))


########################Calulation 2############################################

	# Creating handle for global kernel function
	dist_gpu2 = mod.get_function("distance2")

	# Launch configuration: Size of 1D block
	block = launch.optimal_blocksize(autoinit.device, dist_gpu2)
	print "Block shape:", str(block)+"x1.0"

	# Launch configuration: 1D Grid size (limited by GPU global memory and max grid size of GPU)
	grid = float(min(autoinit.device.max_grid_dim_x, launch.optimise_gridsize(7.76)))
	print "Grid shape:", str(grid_i)+"x1.0"

	# Prepare input data
	d3 = array(theta)[(N1+N2):(N1+N2+sum(N3)),:,:].astype(float64)

	#Initialize array for results
	result = zeros([N1,max([int(ceil(res_d2/grid)) for res_d2 in N3])])

	# Maximum number of particles per run in j direction
	Nj = int(grid)

	# Create template array for res1
	temp_res2 = zeros([Nj]).astype(float64)



	for i in range(N1):

		# Prepare data that depends on i for this run
		data1 = d1[i,:,:]

		# Maximum number of particles per run in j direction
		Nj = int(grid)

		# Determine required number of runs for i and j
		numRuns_j2 = int(ceil(N3[i]/grid))

		# Resets last to "False"
		last = False

		for j in range(numRuns_j2):
			#print "runs left:", numRuns_j2 - j

			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid)*(j+1)) > N3[i]):
				Nj = int(N3[i] - grid*j)
				last = True


			# Prepare data that depends on j for this run
			data3 = d3[(i*N3[i]+j*int(grid)):(i*N3[i]+j*int(grid)+Nj),:,:]


			# Set j dimension of block and grid for this run
			if(Nj<block):
				gj = 1
				bj = Nj
			else:
				gj = ceil(Nj/block)
				bj = block

			# Prepare results array for this run
			if last==True:
				res2 = copy.deepcopy(temp_res2[:int(gj)])
			elif j==0:
				res2 = copy.deepcopy(temp_res2[:int(gj)])


			iterations = odd_num(int(bj))
			if iterations.size == 0:
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				temp_1 = iterations.size

			# Call GPU kernel function
			dist_gpu2(int32(temp_1), driver.In(iterations), int32(Nj), int32(M), int32(P), float32(sigmasq_inv), float64(mplogscale), driver.In(data1), driver.In(data3),  driver.Out(res2), block=(int(bj),1,1), grid=(int(gj),1), shared=3000)
			
			# Sum all elements in results array for this run
			result[i,j] = sum(res2)


	# Sum all content of new results matrix and add/subtract constants for each row if there are no NANs or infs
	sum_result2=ma.log(sum(result, axis=1))
	


########################Final Computations######################################
	
	# Precalculating required variables for next steps
	mplogscale= M*P*log(scale)
	mplogpisigma= M*P*log(2.0*pi*sigma*sigma)
	logN2 = log(float(N2))
	
	# Final division to give mutual information
	
	## Determine infinites
	count_inf1=ma.count_masked(sum_result1)
	count_inf2=ma.count_masked(sum_result2)

	## Creating a joint masked
	master_mask = ma.mask_or(ma.getmaskarray(sum_result1), ma.getmaskarray(sum_result2), shrink=False)
	count_all_inf = sum(master_mask)

	## Inverting boolean array for indexing purposes in the next step
	mask = ~master_mask
	
	## Calculating proportions of Infs
	print "Proportion of infs in 1st summation", int(((count_inf1)/float(N1))*100), "%"
	print "Proportion of infs in 2nd summation", int(((count_inf2)/float(N1))*100), "%"	
	print "Proportion of infs in total", int(((count_all_inf)/float(N1))*100), "%"

	## Final summations
	sum1 = sum(sum_result1[mask])-logN2*(N1-count_all_inf)-mplogscale*(N1-count_all_inf)-mplogpisigma*(N1-count_all_inf)
	sum2 = sum(sum_result2[mask]) - sum(log(N3)[mask]) - mplogscale*(N1-count_all_inf) - mplogpisigma*(N1-count_all_inf)
	
	Info = (sum2 - sum1)/float(N1 - count_all_inf) ###Should be 2*N1 here?

	return(Info)

# A function calling getEntropy2 for all provided experiments and outputs the mutual information
##Argument: model_obj - an object containing all experiments and all their associated information
def run_getEntropy2(model_obj):
	#Initiates list for mutual information
	MutInfo2 = []
	#Cycles through experiments
	for experiment in range(model_obj.nmodels):
		#Extracts N1, N2, N3
		if model_obj.initialprior == False:
			pos = model_obj.pairParamsICS[model_obj.cuda[experiment]].index([x[1] for x in model_obj.x0prior[experiment]])
			N1 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][0]
			N2 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][1]
			N3 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][2]
		else:
			pos = model_obj.cudaout_structure[model_obj.cuda[experiment]][0]
			N1 = pos[0]
			N2 = pos[1]
			N3 = pos[2]

		#Calculates mutual information
		print "-----Calculating Mutual Information for Experiment", experiment+1,"-----"
		MutInfo2.append(getEntropy2(model_obj.trajectories[experiment],model_obj.cudaout[experiment],N1,N2,N3,model_obj.sigma,model_obj.scale[experiment]))
		print "Mutual Information for Experiment", str(experiment+1)+":", MutInfo2[experiment]

	#Returns mutual information
	return MutInfo2