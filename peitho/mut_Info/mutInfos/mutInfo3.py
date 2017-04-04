from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit
import peitho.mut_Info.mutInfos.launch as launch
import sys
import copy


# A function to calculate the mutual information between the outcome of two experiments
##(gets called by run_mutInfo1)
##Arguments: (Ref = reference experiment, Mod = alternative experiment)
##data - array of tracjectories with noise added
##theta - array of trajectories without noise
##N1,N2 - Number of particles
##sigma - standard deviation
##scale - scaling constant to prevent nans and infs
#@profile
def mutInfo3(dataRef,thetaRef,dataMod,thetaMod,N1,N2,N3,N4,sigma_ref,sigma_mod,scale_ref,scale_mod):
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
	__global__ void kernel_func3(int len_odd, int* odd_nums, int Ni, int Nj, int T_Ref, int S_Ref, int T_Mod, int S_Mod, float sigma_ref, float sigma_mod, double mpscale_sum, double *d1, double *d2, double *d3, double *d4, double *res1)
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
		s[idx2d(blockDim.x+threadIdx.x,tid,blockDim.y)] = 0.0;

		//Return threads that are not needed
		if((i>=Ni)||(j>=Nj)) return;

		//Calculate probabilities between trajectory x_i and mean mu_j
		for(int k=0; k<T_Ref; k++){
			for(int l=0; l<S_Ref; l++){
				s[idx2d(threadIdx.x,tid,blockDim.y)] -= ( d2[idx3d(j,k,l,T_Ref,S_Ref)]-d1[idx3d(i,k,l,T_Ref,S_Ref)])*( d2[idx3d(j,k,l,T_Ref,S_Ref)]-d1[idx3d(i,k,l,T_Ref,S_Ref)]);
			}
		}

		//Calculate probabilities between trajectory x*_i and mean mu_j
		for(int k=0; k<T_Mod; k++){
			for(int l=0; l<S_Mod; l++){
				s[idx2d(blockDim.x+threadIdx.x,tid,blockDim.y)] -= ( d4[idx3d(j,k,l,T_Mod,S_Mod)]-d3[idx3d(i,k,l,T_Mod,S_Mod)])*( d4[idx3d(j,k,l,T_Mod,S_Mod)]-d3[idx3d(i,k,l,T_Mod,S_Mod)]);
			}
		}

		s[idx2d(threadIdx.x,tid,blockDim.y)] =  exp(mpscale_sum+ sigma_ref*s[idx2d(threadIdx.x,tid,blockDim.y)]+sigma_mod*s[idx2d(blockDim.x+threadIdx.x,tid,blockDim.y)]);
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

	//Function to calculate intemediary probabilities for mutual information calculation
	__global__ void kernel_func1(int len_odd, int* odd_nums,int Ni, int Nj, int T, int S, float sigma, double mpscale, double *d5, double *d6, double *res2)
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
				s[idx2d(threadIdx.x,tid,blockDim.y)] -= ( d6[idx3d(j,k,l,T,S)]-d5[idx3d(i,k,l,T,S)])*( d6[idx3d(j,k,l,T,S)]-d5[idx3d(i,k,l,T,S)]);
			}
		}

		s[idx2d(threadIdx.x,tid,blockDim.y)] =  exp(mpscale + sigma*s[idx2d(threadIdx.x,tid,blockDim.y)]);
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
		if (tid==0) res2[idx2d(i,blockIdx.y,gridDim.y)] = s[idx2d(threadIdx.x,0,blockDim.y)];

	}

	""")

	# Creating handles for global kernel functions
	gpu_kernel_func3 = mod.get_function("kernel_func3")
	gpu_kernel_func1 = mod.get_function("kernel_func1")

	# Prepare input data
	d1 = dataRef.astype(float64)
	d2 = thetaRef[N1:(N1+N2),:,:].astype(float64)
	d3 = dataMod.astype(float64)
	d4 = array(thetaMod)[N1:(N1+N2),:,:].astype(float64)
	d6 = array(thetaRef)[(N1+N2):(N1+N2+N3),:,:].astype(float64)
	d8 = array(thetaMod)[(N1+N2):(N1+N2+N4),:,:].astype(float64)

	# Determine number of timepoints (T) and number of species (S)
	##Reference experiment
	T_Ref=d1.shape[1]
	S_Ref=d1.shape[2]
	##Alternative experiment
	T_Mod=d3.shape[1]
	S_Mod=d3.shape[2]

	# Determine T*S*scale and 1/(2*sigma^2) for reference and alternative experiment
	##Reference experiment
	tsscale_ref = T_Ref*S_Ref*scale_ref
	sigma_inv_ref = 1.0/(2.0*sigma_ref*sigma_ref)

	##Alternative experiment
	tsscale_mod = T_Mod*S_Mod*scale_mod
	sigma_inv_mod = 1/(2.0*sigma_mod*sigma_mod)

	##Sum of scaling factors
	mpscale_sum = tsscale_ref+tsscale_mod


########################Calulation 1############################################

	print "-----Determining optimal kernel launch configuration (for part 1/3)-----"

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, gpu_kernel_func3, 16)
	block_i = launch.factor_partial(block)
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid_prelim_i , grid_prelim_j = launch.optimise_gridsize(3, block_i, block_j, T_Mod, S_Mod, T_Ref, S_Ref)
	grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i, N1))
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, N2))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)
	print "Registers:", gpu_kernel_func3.num_regs, "\n"

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N2/grid_j))

	#Initialize arrays for results
	result1 = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i and j direction
	Ni = int(grid_i)
	Nj = int(grid_j)

	# Create template array for res1
	try:
		temp_res1 = zeros([Ni,int(ceil(Nj/block_j))]).astype(float64)
	except:
		print "ERROR: Not enought memory (RAM) available to create array for GPU results. Reduce GPU grid size."
		sys.exit()

	print "-----Calculation part 1 of 3 now running-----"

	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):

		# If last run with less that max remaining particles, set Ni to remaining number of particles
		if((int(grid_i)*(i+1)) > N1):
			Ni = int(N1 - grid_i*i)

		# Prepare data that depends on i for this run
		data1 = d1[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:]
		data3 = d3[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:]

		# Set i dimension of block and grid for this run
		if(Ni<block_i):
			gi = 1
			bi = Ni
		else:
			bi = block_i
			gi = ceil(Ni/block_i)

		# Maximum number of particles per run in j direction
		Nj = int(grid_j)

		# Resets last to "False"
		last = False

		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N2):
				Nj = int(N2 - grid_j*j)
				last=True

			# Prepare data that depends on j for this run
			data2 = d2[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]
			data4 = d4[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				bj = block_j
				gj = ceil(Nj/block_j)

			# Prepare results array for this run
			if last==True:
				res1 = copy.deepcopy(temp_res1[:Ni,:int(gj)])
			elif j==0:
				res1 = copy.deepcopy(temp_res1[:Ni,:int(gj)])

			iterations = launch.odd_num(int(bj))
			if iterations.size == 0:
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				temp_1 = iterations.size

			# Call GPU kernel functions
			gpu_kernel_func3(int32(temp_1), driver.In(iterations), int32(Ni), int32(Nj), int32(T_Ref), int32(S_Ref), int32(T_Mod), int32(S_Mod), float32(sigma_inv_ref), float32(sigma_inv_mod), float64(mpscale_sum), driver.In(data1), driver.In(data2), driver.In(data3), driver.In(data4), driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)),shared=int(2*bi*bj*8))

			# Summing rows in GPU output for this run
			result1[i*int(grid_i):i*int(grid_i)+Ni,j] = sum(res1,axis=1)

	print "-----Calculation part 1 of 3 completed-----\n"

########################Calulation 2############################################

	print "-----Determining optimal kernel launch configuration (for part 2/3)-----"

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, gpu_kernel_func1, 8)
	block_i = launch.factor_partial(block)
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multiple of block size)
	grid_prelim_i , grid_prelim_j = launch.optimise_gridsize(1, block_i, block_j, T_Ref, S_Ref)
	grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i, N1))
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, N3))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)
	print "Registers:", gpu_kernel_func1.num_regs, "\n"

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N3/grid_j))

	#Initialize arrays for results
	result2 = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i and j direction
	Ni = int(grid_i)
	Nj = int(grid_j)

	# Create template array for res1
	try:
		temp_res2 = zeros([Ni,int(ceil(Nj/block_j))]).astype(float64)
	except:
		print "ERROR: Not enought memory (RAM) available to create array for GPU results. Reduce GPU grid size."
		sys.exit()

	print "-----Calculation part 2 of 3 now running-----"

	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):

		# If last run with less that max remaining particles, set Ni to remaining number of particles
		if((int(grid_i)*(i+1)) > N1):
			Ni = int(N1 - grid_i*i)

		# Prepare data that depends on i for this run
		data1 = d1[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:]

		# Set i dimension of block and grid for this run
		if(Ni<block_i):
			gi = 1
			bi = Ni
		else:
			bi = block_i
			gi = ceil(Ni/block_i)

		# Maximum number of particles per run in j direction
		Nj = int(grid_j)

		# Resets last to "False"
		last = False

		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N3):
				Nj = int(N3 - grid_j*j)
				last=True

			# Prepare data that depends on j for this run
			data6 = d6[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				bj = block_j
				gj = ceil(Nj/block_j)

			# Prepare results array for this run
			if last==True:
				res2 = copy.deepcopy(temp_res2[:Ni,:int(gj)])
			elif j==0:
				res2 = copy.deepcopy(temp_res2[:Ni,:int(gj)])

			iterations = launch.odd_num(int(bj))
			if iterations.size == 0:
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				temp_1 = iterations.size

			# Call GPU kernel functions
			gpu_kernel_func1(int32(temp_1), driver.In(iterations), int32(Ni), int32(Nj), int32(T_Ref), int32(S_Ref), float32(sigma_inv_ref), float64(tsscale_ref), driver.In(data1), driver.In(data6), driver.Out(res2), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)), shared=int(bi*bj*8))

			# Summing rows in GPU output for this run
			result2[i*int(grid_i):i*int(grid_i)+Ni,j] = sum(res2, axis=1)

	print "-----Calculation part 2 of 3 completed-----\n"

########################Calulation 3############################################

	print "-----Determining optimal kernel launch configuration (for part 3/3)-----"

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, gpu_kernel_func1, 8)
	block_i = launch.factor_partial(block)
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid_prelim_i , grid_prelim_j = launch.optimise_gridsize(1, block_i, block_j, T_Mod, S_Mod)
	grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i, N1))
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, N4))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)
	print "Registers:", gpu_kernel_func1.num_regs, "\n"

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N4/grid_j))

	#Initialize arrays for results
	result3 = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i and j direction
	Ni = int(grid_i)
	Nj = int(grid_j)

	# Create template array for res1
	try:
		temp_res3 = zeros([Ni,int(ceil(Nj/block_j))]).astype(float64)
	except:
		print "ERROR: Not enought memory (RAM) available to create array for GPU results. Reduce GPU grid size."
		sys.exit()

	print "-----Calculation part 3 of 3 now running-----"

	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):
		#print "Runs left:", numRuns_i - i

		# If last run with less that max remaining particles, set Ni to remaining number of particles
		if((int(grid_i)*(i+1)) > N1):
			Ni = int(N1 - grid_i*i)

		# Prepare data that depends on i for this run
		data3 = d3[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:]

		# Set i dimension of block and grid for this run
		if(Ni<block_i):
			gi = 1
			bi = Ni
		else:
			bi = block_i
			gi = ceil(Ni/block_i)

		# Maximum number of particles per run in j direction
		Nj = int(grid_j)

		# Resets last to "False"
		last = False

		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N4):
				Nj = int(N4 - grid_j*j)
				last=True

			# Prepare data that depends on j for this run
			data8 = d8[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				bj = block_j
				gj = ceil(Nj/block_j)

			# Prepare results array for this run
			if last==True:
				res3 = copy.deepcopy(temp_res3[:Ni,:int(gj)])
			elif j==0:
				res3 = copy.deepcopy(temp_res3[:Ni,:int(gj)])

			iterations = launch.odd_num(int(bj))

			if iterations.size == 0:
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				temp_1 = iterations.size

			# Call GPU kernel functions
			gpu_kernel_func1(int32(temp_1), driver.In(iterations), int32(Ni), int32(Nj), int32(T_Mod), int32(S_Mod), float32(sigma_inv_mod), float64(tsscale_mod), driver.In(data3), driver.In(data8), driver.Out(res3), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)),shared=int(bi*bj*8))

			# Summing rows in GPU output for this run
			result3[i*int(grid_i):i*int(grid_i)+Ni,j] = sum(res3, axis=1)

	print "-----Calculation part 3 of 3 completed-----\n"

########################Final Computations######################################

	# Sum all content of new results matrix and add/subtract constants for each row if there are no NANs or infs
	sum_result1=ma.log(sum(result1,axis=1))
	count_inf1=ma.count_masked(sum_result1)

	sum_result2=ma.log(sum(result2,axis=1))
	count_inf2=ma.count_masked(sum_result2)

	sum_result3=ma.log(sum(result3,axis=1))
	count_inf3=ma.count_masked(sum_result3)

	# Creating a joint masked
	master_mask = ma.mask_or(ma.mask_or(sum_result1.mask, sum_result2.mask), sum_result3.mask)

	#Sum of all Infs
	count_all_inf = sum(master_mask)

	# Inverting boolean array for indexing purposes in the next step
	mask = ~master_mask

	# Raise error if calculation below cannot be carried out due to div by 0
	if count_all_inf == N1:
		print "ERROR: Too many nan/inf values in output, could not calculate mutual information. Consider increasing particle size or adapting prior distributions."
		sys.exit()

	# Final summation
	sum_2= sum(sum_result1[mask]) - sum(sum_result2[mask]) - sum(sum_result3[mask]) - log(float(N2))*(N1-count_all_inf) + log(float(N3))*(N1-count_all_inf) + log(float(N4))*(N1-count_all_inf )

	# Final division to give mutual information
	Info2 = sum_2/float(N1-count_all_inf)

	# Calculate proportion of infinites and print Infs  results
	count_all_inf_prop=int((count_all_inf/float(N1))*100)
	print "Proportion of infs in 1st summation", int(((count_inf1)/float(N1))*100), "% ("+str(count_inf1)+" infs)"
	print "Proportion of infs in 2nd summation", int(((count_inf2)/float(N1))*100), "% ("+str(count_inf2)+" infs)"
	print "Proportion of infs in 3rd summation", int(((count_inf3)/float(N1))*100), "% ("+str(count_inf3)+" infs)"
	print "Proportion of infinite trajectories in total", count_all_inf_prop, "% ("+str(count_all_inf)+" infs)\n"

	return Info2, count_all_inf, count_all_inf_prop


# A function calling mutInfo3 for all provided experiments and outputs the mutual information
##Arguments:
##model_obj - an object containing all alternative experiments and all their associated information
##ref_obj - an object containing the reference experiment and all associated information
def run_mutInfo3(model_obj, ref_obj, input_SBML):
	#Initiates list for mutual information
	MutInfo3 = []
	#Initiates list to hold number of infinites
	MutInfo3_infs = []
	#Initiates list to hold percentage of infinites
	MutInfo3_infs_prop = []

	#Cycles through experiments
	for experiment in range(model_obj.nmodels):

		#Extracts N1,N2,N3,N4
		if model_obj.initialprior == False:
			pos = model_obj.pairParamsICS[model_obj.cuda[experiment]].index([x[1] for x in model_obj.x0prior[experiment]])
			pos2 = ref_obj.pairParamsICS[ref_obj.cuda[0]].index([x[1] for x in ref_obj.x0prior[0]])
			N1_mod = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][0]
			N2_mod = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][1]
			N1_ref = ref_obj.cudaout_structure[ref_obj.cuda[experiment]][pos2][0]
			N2_ref = ref_obj.cudaout_structure[ref_obj.cuda[experiment]][pos2][1]

			N3 = ref_obj.cudaout_structure[ref_obj.cuda[experiment]][pos2][2]
			N4 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][3]
		else:
			pos = model_obj.cudaout_structure[model_obj.cuda[experiment]][0]
			pos2 = ref_obj.cudaout_structure[ref_obj.cuda[0]][0]
			N1_mod = pos[0]
			N2_mod = pos[1]
			N1_ref = pos2[0]
			N2_ref = pos2[1]

			N3 = pos2[2]
			N4 = pos[3]

		#Need to take minimum as N1 and N2 may differ between reference and experiments
		N1 = min(N1_mod,N1_ref)
		N2 = min(N2_mod,N2_ref)

		#Calculates mutual information
		print "-----Calculating Mutual Information for Experiment", experiment+1, "for", input_SBML,"-----\n"
		
		## Assign mutual information and number of infinites to lists
		temp_list=mutInfo3(ref_obj.trajectories[0],ref_obj.cudaout[0],model_obj.trajectories[experiment], model_obj.cudaout[experiment],N1,N2,N3,N4,ref_obj.sigma,model_obj.sigma,ref_obj.scale[0],model_obj.scale[experiment])
		MutInfo_lists = [MutInfo3, MutInfo3_infs, MutInfo3_infs_prop]
		for x, lst in zip(temp_list, MutInfo_lists):
			lst.append(x)

		## Print out mutual information
		print "Mutual Information for Experiment", str(experiment+1)+":", MutInfo3[experiment], "\n"

	#Returns mutual information
	return MutInfo3, MutInfo3_infs, MutInfo3_infs_prop
