from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit

import peitho.mut_Info.mutInfos.launch as launch
import sys
import copy


# A function to calculate the mutual information between all parameters of a system and an experiment
##(gets called by run_mutInfo1)
##Arguments:
##data - array of tracjectories with noise added
##theta - array of trajectories without noise
##N1,N2 - Number of particles
##sigma - stadard deviation
##scale - scaling constant to prevent nans and infs
def mutInfo1(data,theta,N1,N2,sigma,scale):
	# Kernel declaration using pycuda SourceModule

	mod = compiler.SourceModule("""

	//Function to index 3-dimensional flattened arrays
	__device__ unsigned int idx3d(int i, int k, int l, int T, int S)
	{
		return k*S + i*T*S + l;
	}

	//Function to index 2-dimensional flattened arrays
	__device__ unsigned int idx2d(int i, int j, int T)
	{
		return i*T + j;
	}

	//Function to calculate intemediary probabilities for mutual information calculation
	__global__ void kernel_func1(int len_odd, int* odd_nums, int Ni, int Nj, int T, int S, float sigma, double scale, double *d1, double *d2, double *res1)
	{
		//Define shared memory dynamically
		extern __shared__ double s[];

		//Assign thread indicies
		unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;

		//Assign thread index for shared memory
		unsigned int tid = threadIdx.y;

		//Initialise shared memory values
		s[idx2d(threadIdx.x,tid,blockDim.y)] = 0.0;

		//Return threads that are not needed
		if((i>=Ni)||(j>=Nj)) return;

		//Calculate probabilities between trajectory x_i and mean mu_j
		for(int k=0; k<T; k++){
			for(int l=0; l<S; l++){
				s[idx2d(threadIdx.x,tid,blockDim.y)] -= ( d2[idx3d(j,k,l,T,S)]-d1[idx3d(i,k,l,T,S)])*( d2[idx3d(j,k,l,T,S)]-d1[idx3d(i,k,l,T,S)]);
			}
		}

		s[idx2d(threadIdx.x,tid,blockDim.y)] =  exp(scale + sigma*s[idx2d(threadIdx.x,tid,blockDim.y)]);
		__syncthreads();

		//First part of reduction - collapses each threadblock down to one vector
		for(unsigned int k=blockDim.y/2; k>0; k>>=1){
			if(tid < k){
				s[idx2d(threadIdx.x,tid,blockDim.y)] += s[idx2d(threadIdx.x,tid+k,blockDim.y)];
			}
			__syncthreads();
		}

		//Final part of reduction involves adding back any columns that were missed out from the first part
		if(len_odd != -1){
			for(unsigned int l=0; l<len_odd; l+=1){
				if (tid == 0) s[idx2d(threadIdx.x,0,blockDim.y)] += s[idx2d(threadIdx.x, odd_nums[l]-1 ,blockDim.y)];
				__syncthreads();
			}
		}

		//Load the data back into global memory
		if (tid==0) res1[idx2d(i,blockIdx.y,gridDim.y)] = s[idx2d(threadIdx.x,0,blockDim.y)];

	}
	""")

	# Creating handle for global kernel function
	gpu_kernel_func1 = mod.get_function("kernel_func1")

	# Prepare input data
	d1 = data.astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	# Determine number of timepoints (T) and number of species (S)
	T = d1.shape[1]
	S = d1.shape[2]

	print "-----Determining optimal kernel launch configuration-----"

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, gpu_kernel_func1, 8)
	block_i = launch.factor_partial(block) # Maximum threads per block
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid_prelim_i , grid_prelim_j = launch.optimise_gridsize(1, block_i, block_j, T, S)
	grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i, N1))
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, N2))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)
	print "Registers:", gpu_kernel_func1.num_regs, "\n"

	print "-----Calculation part 1 of 1 now running-----"

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N2/grid_j))

	#Initialize array for results
	result = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i and j direction
	Ni = int(grid_i)
	Nj = int(grid_j)

	# Create template array for res1
	try:
		temp_res1 = zeros([Ni,int(ceil(Nj/block_j))]).astype(float64)
	except:
		print "ERROR: Not enought memory (RAM) available to create array for GPU results. Reduce GPU grid size."
		sys.exit()

	# Determine T*S*log(scale) for GPU calculation
	tslogscale= T*S*log(scale)

	# Determine 1/2*sigma*sigma for GPU calculation
	sigmasq_inv = 1/(2*sigma*sigma)

	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):

		# If last run with less that max remaining particles, set Ni to remaining number of particles
		if((int(grid_i)*(i+1)) > N1):
			Ni = int(N1 - grid_i*i)

		# Prepare data that depends on i for this run
		data1 = d1[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:] # d1 subunit for the next j runs

		# Set i dimension of block and grid for this run
		if(Ni<block_i):
			gi = 1
			bi = Ni
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

			# Prepare results array for run
			if last==True:
				res1 = copy.deepcopy(temp_res1[:Ni,:int(gj)])
			elif j==0:
				res1 = copy.deepcopy(temp_res1[:Ni,:int(gj)])

			iterations = launch.odd_num(int(bj))

			if iterations.size == 0:
				#print "here"
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				#print "here2"
				temp_1 = iterations.size

			# Call GPU kernel functions
			gpu_kernel_func1(int32(temp_1), driver.In(iterations),int32(Ni),int32(Nj), int32(T), int32(S), float32(sigmasq_inv), float64(tslogscale), driver.In(data1), driver.In(data2), driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj),1), shared = int(bi*bj*8) )

			# Summing rows in GPU output for this run
			result[i*int(grid_i):i*int(grid_i)+Ni,j]=sum(res1, axis=1)

	# Sum all content of new results matrix and add/subtract constants for each row if there are no NANs or infs
	sum_result=ma.log(sum(result,axis=1))
	count_inf=ma.count_masked(sum_result)
	sum1 = -ma.sum(sum_result)+log(float(N2))*(N1-count_inf)+tslogscale*(N1-count_inf)+T*S*log(2.0*pi*sigma*sigma)*(N1-count_inf)

	# Raise error if calculation below cannot be carried out due to div by 0
	if count_inf == N1:
		print "ERROR: Too many nan/inf values in output, could not calculate mutual information. Consider increasing particle size pr ada[ting prior distributions."
		sys.exit()

	# Final division to give mutual information
	Info = (sum1 / float(N1- count_inf) - T*S/2.0*(log(2.0*pi*sigma*sigma)+1))

	print "-----Calculation part 1 of 1 complete-----\n"

	# Calculate and print proportion of infinites
	infs_na_prop=int((count_inf/float(N1))*100)
	print "Proportion of infs and NAs", infs_na_prop, "% ("+str(count_inf)+" infs)\n"

	return Info,count_inf,infs_na_prop

# A function calling mutInfo1 for all provided experiments and outputs the mutual information
##Argument: model_obj - an object containing all experiments and all their associated information
def run_mutInfo1(model_obj,input_SBML):
	#Initiates list to hold mutual information
	MutInfo1 = []
	#Initiates list to hold number of infinites
	MutInfo1_infs = []
	#Initiates list to hold percentage of infinites
	MutInfo1_infs_prop = []	
	#Cycles through each experiment
	for experiment in range(model_obj.nmodels):

		#Extracts N1 and N2
		if model_obj.initialprior == False:
			pos = model_obj.pairParamsICS[model_obj.cuda[experiment]].index([x[1] for x in model_obj.x0prior[experiment]])
			N1 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][0]
			N2 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][1]
		else:
			pos = model_obj.cudaout_structure[model_obj.cuda[experiment]][0]
			N1 = pos[0]
			N2 = pos[1]

		#Calculates mutual information
		print "-----Calculating Mutual Information for Experiment", experiment+1, "for", input_SBML,"-----\n"
		
		## Assign mutual information and number of infinites to lists
		temp_list=mutInfo1(model_obj.trajectories[experiment],model_obj.cudaout[experiment],N1,N2,model_obj.sigma,model_obj.scale[experiment])
		MutInfo_lists = [MutInfo1, MutInfo1_infs, MutInfo1_infs_prop]
		for x, lst in zip(temp_list, MutInfo_lists):
			lst.append(x)

		## Print out mutual information
		print "Mutual Information for Experiment", str(experiment+1)+":", MutInfo1[experiment], "\n"

	#Returns mutual information
	return MutInfo1, MutInfo1_infs, MutInfo1_infs_prop
