from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit
import sys
import launch
import copy

def odd_num(x):
	temp = []
	pos=0
	while x > 1:
		if x%2 ==1:
			temp.append(x)
		x = x >> 1
	return asarray(temp).astype(int32)

# A function to calculate the mutual information between the outcome of two experiments
##(gets called by run_getEntropy1)
##Arguments: (Ref = reference experiment, Mod = alternative experiment)
##data - array of tracjectories with noise added
##theta - array of trajectories without noise
##N1,N2 - Number of particles
##sigma - standard deviation
##scale - scaling constant to prevent nans and infs
#@profile
def getEntropy3(dataRef,thetaRef,dataMod,thetaMod,N1,N2,N3,N4,sigma_ref,sigma_mod,scale_ref,scale_mod):
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

	__global__ void distance1(int len_odd, int* odd_nums, int Ni, int Nj, int M_Ref, int P_Ref, int M_Mod, int P_Mod, float sigma_ref, float sigma_mod, double mpscale_sum, double *d1, double *d2, double *d3, double *d4, double *res1)
	{
		extern __shared__ double s[];

		unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int tid = threadIdx.y;

		s[idx2d(threadIdx.x,tid,blockDim.y)] = 0.0;
		s[idx2d(blockDim.x+threadIdx.x,tid,blockDim.y)] = 0.0;

		if((i>=Ni)||(j>=Nj)) return;

		for(int k=0; k<M_Ref; k++){
			for(int l=0; l<P_Ref; l++){
				s[idx2d(threadIdx.x,tid,blockDim.y)] -= ( d2[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])*( d2[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)]);
			}
		}

		for(int k=0; k<M_Mod; k++){
			for(int l=0; l<P_Mod; l++){
				s[idx2d(blockDim.x+threadIdx.x,tid,blockDim.y)] -= ( d4[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])*( d4[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)]);
			}
		}

		s[idx2d(threadIdx.x,tid,blockDim.y)] =  exp(mpscale_sum+ sigma_ref*s[idx2d(threadIdx.x,tid,blockDim.y)]+sigma_mod*s[idx2d(blockDim.x+threadIdx.x,tid,blockDim.y)]);
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



	__global__ void distance2(int len_odd, int* odd_nums,int Ni, int Nj, int M, int P, float sigma, double mpscale, double *d5, double *d6, double *res2)
	{
		extern __shared__ double s[];

		unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int tid = threadIdx.y;

		s[idx2d(threadIdx.x,tid,blockDim.y)] = 0.0;

		if((i>=Ni)||(j>=Nj)) return;


		for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				s[idx2d(threadIdx.x,tid,blockDim.y)] -= ( d6[idx3d(j,k,l,M,P)]-d5[idx3d(i,k,l,M,P)])*( d6[idx3d(j,k,l,M,P)]-d5[idx3d(i,k,l,M,P)]);
			}
		}

		s[idx2d(threadIdx.x,tid,blockDim.y)] =  exp(mpscale + sigma*s[idx2d(threadIdx.x,tid,blockDim.y)]);
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

		if (tid==0) res2[idx2d(i,blockIdx.y,gridDim.y)] = s[idx2d(threadIdx.x,0,blockDim.y)];

	}

	""")

	# Creating handles for global kernel functions
	dist_gpu1 = mod.get_function("distance1")
	dist_gpu2 = mod.get_function("distance2")

	# Prepare input data
	d1 = dataRef.astype(float64)
	d2 = thetaRef[N1:(N1+N2),:,:].astype(float64)
	d3 = dataMod.astype(float64)
	d4 = array(thetaMod)[N1:(N1+N2),:,:].astype(float64)
	d6 = array(thetaRef)[(N1+N2):(N1+N2+N3),:,:].astype(float64)
	d8 = array(thetaMod)[(N1+N2):(N1+N2+N4),:,:].astype(float64)

	# Determine number of timepoints (M) and number of species (P)
	##Reference experiment
	M_Ref=d1.shape[1]
	P_Ref=d1.shape[2]
	##Alternative experiment
	M_Mod=d3.shape[1]
	P_Mod=d3.shape[2]

	# Determine M*P*scale and 1/(2*sigma^2) for reference and alternative experiment
	##Reference experiment
	mpscale_ref = M_Ref*P_Ref*scale_ref
	sigma_inv_ref = 1.0/(2.0*sigma_ref*sigma_ref)

	##Alternative experiment
	mpscale_mod = M_Mod*P_Mod*scale_mod
	sigma_inv_mod = 1/(2.0*sigma_mod*sigma_mod)

	##Sum of scaling factors
	mpscale_sum = mpscale_ref+mpscale_mod


########################Calulation 1############################################

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, dist_gpu1)
	block_i = launch.factor_partial(block)
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid = launch.optimise_gridsize(9.0)
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

	#Initialize arrays for results
	result1 = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i and j direction
	Ni = int(grid_i)
	Nj = int(grid_j)

	# Create template array for res1
	temp_res1 = zeros([Ni,Nj]).astype(float64)

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

			iterations = odd_num(int(bj))
			if iterations.size == 0:
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				temp_1 = iterations.size

			# Call GPU kernel functions
			dist_gpu1(int32(temp_1), driver.In(iterations), int32(Ni), int32(Nj), int32(M_Ref), int32(P_Ref), int32(M_Mod), int32(P_Mod), float32(sigma_inv_ref), float32(sigma_inv_mod), float64(mpscale_sum), driver.In(data1), driver.In(data2), driver.In(data3), driver.In(data4), driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)),shared=int(2*bi*bj*8))

			# Summing rows in GPU output for this run
			result1[i*int(grid_i):i*int(grid_i)+Ni,j] = sum(res1,axis=1) ###Could be done on GPU?


########################Calulation 2############################################

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, dist_gpu2)
	block_i = launch.factor_partial(block)
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid = launch.optimise_gridsize(9.0)
	grid_prelim_i = launch.round_down(sqrt(grid),block_i)
	grid_prelim_j = launch.round_down(grid/grid_prelim_i,block_j)
	# If gridsize in one dimention too large, reshape grid to allow more threads in the second dimension
	if N1 < grid_prelim_i:
		grid_i = float(min(autoinit.device.max_grid_dim_x,N1))
		grid_j = float(min(autoinit.device.max_grid_dim_y, launch.round_down(grid/grid_i,block_j)))
	elif N3 < grid_prelim_j:
		grid_j = float(min(autoinit.device.max_grid_dim_y,N3))
		grid_i = float(min(autoinit.device.max_grid_dim_x, launch.round_down(grid/grid_j,block_i)))
	else:
		grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i))
		grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j))
	print "Maximum gridsize:", grid, "threads"
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N3/grid_j))

	#Initialize arrays for results
	result2 = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i and j direction
	Ni = int(grid_i)
	Nj = int(grid_j)

	# Create template array for res1
	temp_res2 = zeros([Ni,Nj]).astype(float64)

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

			iterations = odd_num(int(bj))
			if iterations.size == 0:
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				temp_1 = iterations.size

			# Call GPU kernel functions
			dist_gpu2(int32(temp_1), driver.In(iterations), int32(Ni), int32(Nj), int32(M_Ref), int32(P_Ref), float32(sigma_inv_ref), float64(mpscale_ref), driver.In(data1), driver.In(data6), driver.Out(res2), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)), shared=int(bi*bj*8))

			# Summing rows in GPU output for this run
			result2[i*int(grid_i):i*int(grid_i)+Ni,j] = sum(res2, axis=1) ###Could be done on GPU?


########################Calulation 3############################################

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, dist_gpu2)
	block_i = launch.factor_partial(block)
	block_j = block / block_i
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)

	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid = launch.optimise_gridsize(9.0)
	grid_prelim_i = launch.round_down(sqrt(grid),block_i)
	grid_prelim_j = launch.round_down(grid/grid_prelim_i,block_j)
	# If gridsize in one dimention too large, reshape grid to allow more threads in the second dimension
	if N1 < grid_prelim_i:
		grid_i = float(min(autoinit.device.max_grid_dim_x,N1))
		grid_j = float(min(autoinit.device.max_grid_dim_y, launch.round_down(grid/grid_i,block_j)))
	elif N4 < grid_prelim_j:
		grid_j = float(min(autoinit.device.max_grid_dim_y,N4))
		grid_i = float(min(autoinit.device.max_grid_dim_x, launch.round_down(grid/grid_j,block_i)))
	else:
		grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i))
		grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j))
	print "Maximum gridsize:", grid, "threads"
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N4/grid_j))

	#Initialize arrays for results
	result3 = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i and j direction
	Ni = int(grid_i)
	Nj = int(grid_j)

	# Create template array for res1
	temp_res3 = zeros([Ni,Nj]).astype(float64)

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

			iterations = odd_num(int(bj))

			if iterations.size == 0:
				temp_1=-1
				iterations = zeros([1]).astype(float64)
			else:
				temp_1 = iterations.size

			# Call GPU kernel functions
			dist_gpu2(int32(temp_1), driver.In(iterations), int32(Ni), int32(Nj), int32(M_Mod), int32(P_Mod), float32(sigma_inv_mod), float64(mpscale_mod), driver.In(data3), driver.In(data8), driver.Out(res3), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)),shared=int(bi*bj*8))

			# Summing rows in GPU output for this run
			result3[i*int(grid_i):i*int(grid_i)+Ni,j] = sum(res3, axis=1) ###Could be done on GPU?


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

	# Printing Infs  results
	print "", "Infs"
	print "1", count_inf1
	print "2", count_inf2
	print "3", count_inf3
	print "total", count_all_inf


	return(Info2)


# A function calling getEntropy3 for all provided experiments and outputs the mutual information
##Arguments:
##model_obj - an object containing all alternative experiments and all their associated information
##ref_obj - an object containing the reference experiment and all associated information
def run_getEntropy3(model_obj, ref_obj):
	#Initiates list for mutual information
	MutInfo3 = []

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
		print "-----Calculating Mutual Information for Experiment", experiment+1,"-----"
		MutInfo3.append(getEntropy3(ref_obj.trajectories[0],ref_obj.cudaout[0],model_obj.trajectories[experiment], model_obj.cudaout[experiment],N1,N2,N3,N4,ref_obj.sigma,model_obj.sigma,ref_obj.scale[0],model_obj.scale[experiment]))
		print "Mutual Information for Experiment", str(experiment+1)+":", MutInfo3[experiment]

	#Returns mutual information
	return MutInfo3
