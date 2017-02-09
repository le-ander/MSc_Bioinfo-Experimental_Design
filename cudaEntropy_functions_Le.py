import math

def run_cudasim(m_object, parameters, species):
	modelTraj = []
	# For each model in turn...
	##Should run over cudafiles
	for mod in range(m_object.nmodels):
		# Define CUDA filename for cudasim
		cudaCode = m_object.name[mod] + '.cu'
		# Create ODEProblem object
		modelInstance = Lsoda.Lsoda(m_object.times, cudaCode, dt=m_object.dt)
		# Solve ODEs using Lsoda algorithm
		##Different parameters and species matrices for i in nmodels?
		result = modelInstance.run(parameters, species)
		modelTraj.append(result[:,0])
	return modelTraj

def remove_na(m_object, modelTraj):
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Create a list of indices of particles that have an NA in their row
		##Why using 7:8 when summing? -> Change this
		index = [p for p, i in enumerate(isnan(sum(asarray(modelTraj[mod])[:,7:8,:],axis=2))) if i==True]
		# Delete row of 1. results and 2. parameters from the output array for which an index exists
		for i in index:
			delete(modelTraj[mod], (i), axis=0)

	return modelTraj

def add_noise_to_traj(m_object, modelTraj, sigma, N1):##Need to ficure out were to get N1 from
	ftheta = []
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Create array with noise of same size as the trajectory array (only the first N1 particles)
		noise = normal(loc=0.0, scale=sigma,size=shape(modelTraj[mod][0:N1,:,:]))
		# Add noise to trajectories and output new 'noisy' trajectories
		traj = array(modelTraj[mod][0:N1,:,:]) + noise
		ftheta.append(traj)
	# Return final trajectories for 0:N1 particles
	return ftheta

def get_max_dist(m_object, ftheta):
	maxDistTraj = []
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Calculate the maximum distance for each model and store it in array
		maxDistTraj.append(amax(ftheta[mod]) - amin(ftheta[mod]))

	return maxDistTraj

def get_mutinf_all_param(m_object, ftheta, modelTraj, maxDistTraj, sigma):
	MutInfo1 = []
	# For each model in turn....
	for mod in range(m_object.nmodels):
		# Run function to get the mutual information for all parameters
		MutInfo1.append(getEntropy1(ftheta[mod],sigma,array(modelTraj[mod]),maxDistTraj[mod]))

	return MutInfo1

def round_down(num, divisor):
	return num - (num%divisor)

def round_up(num, divisor):
	if num == divisor:
		return 1
	else:
		return num - (num%divisor) + divisor

def max_active_blocks_per_sm(device, function, blocksize, dyn_smem_per_block=0):
	# Define variables based on device and fucntion properties
	regs_per_thread = function.num_regs
	smem_per_function = function.shared_size_bytes
	warp_size = device.warp_size
	max_threads_per_block = min(device.max_threads_per_block, function.max_threads_per_block)
	max_threads_per_sm = device.max_threads_per_multiprocessor
	max_regs_per_block = device.max_registers_per_block
	max_smem_per_block = device.max_shared_memory_per_block
	if device.compute_capability()[0] == 2:
		reg_granul = 64
		warp_granul = 2
		smem_granul = 128
		max_regs_per_sm = 32768
		max_blocks_per_sm = 8
		if regs_per_thread in [21,22,29,30,37,38,45,46]:
			reg_granul = 128
	elif device.compute_capability() == (3,7):
		reg_granul = 256
		warp_granul = 4
		smem_granul = 256
		max_regs_per_sm = 131072
		max_blocks_per_sm = 16
	elif device.compute_capability()[0] == 3:
		reg_granul = 256
		warp_granul = 4
		smem_granul = 256
		max_regs_per_sm = 65536
		max_blocks_per_sm = 16
	elif device.compute_capability() == (6,0):
		reg_granul = 256
		warp_granul = 2
		smem_granul = 256
		max_regs_per_sm = 65536
		max_blocks_per_sm = 32
	else:
		reg_granul = 256
		warp_granul = 4
		smem_granul = 256
		max_regs_per_sm = 65536
		max_blocks_per_sm = 32

	# Calculate the maximum number of blocks, limited by register count
	if regs_per_thread > 0:
		regs_per_warp = round_up(regs_per_thread * warp_size, reg_granul)
		max_warps_per_sm = round_down(max_regs_per_block / regs_per_warp, warp_granul)
		warps_per_block = int(ceil(float(blocksize) / warp_size))
		block_lim_regs = int(max_warps_per_sm / warps_per_block) * int(max_regs_per_sm / max_regs_per_block)
	else:
		block_lim_regs = max_blocks_per_sm

 	# Calculate the maximum number of blocks, limited by blocks/SM or threads/SM
	block_lim_tSM = max_threads_per_sm / blocksize
	block_lim_bSM = max_blocks_per_sm

	# Calculate the maximum number of blocks, limited by shared memory
	req_smem = smem_per_function + dyn_smem_per_block
	if req_smem > 0:
		smem_per_block = round_up(req_smem, smem_granul)
		block_lim_smem = max_smem_per_block / smem_per_block
	else:
		block_lim_smem = max_blocks_per_sm

	# Find the maximum number of blocks based on the limits calculated above
	block_lim = min(block_lim_regs, block_lim_tSM, block_lim_bSM, block_lim_smem)

	#print "block_lims", [block_lim_regs, block_lim_tSM, block_lim_bSM, block_lim_smem]
	#print "BLOCK_LIM", block_lim
	#print "BLOCKSIZE", blocksize

	return block_lim

def optimal_blocksize(device, function):
	# Iterate through block sizes to find largest occupancy
	max_blocksize = min(device.max_threads_per_block, function.max_threads_per_block)
	achieved_occupancy = 0
	blocksize = device.warp_size
	while blocksize <= max_blocksize:
		occupancy = blocksize * max_active_blocks_per_sm(device, function, blocksize)
		#print "OCCUPANCY", occupancy, "\n---------------------"
		if occupancy > achieved_occupancy:
			optimal_blocksize = blocksize
			achieved_occupancy = occupancy
		if achieved_occupancy == device.max_threads_per_multiprocessor:
			break
		blocksize += device.warp_size
	#print "OPTIMAL BLOCKSIZE", optimal_blocksize

	return optimal_blocksize

def optimise_grid_structure(gmem_per_thread=102400): #need to define correct memory requirement for kernel
	# DETERMINE TOTAL NUMBER OF THREADS LIMITED BY GLOBAL MEMORY
	# Read total global memory of device
	avail_mem = autoinit.device.total_memory()
	# Calculate maximum number of threads, assuming global memory usage of 100 KB per thread
	max_threads = floor(avail_mem / gmem_per_thread)

def getEntropy1(data,sigma,theta,maxDistTraj):

	#kernel declaration
	mod = compiler.SourceModule("""
	__device__ unsigned int idx3d(int i, int k, int l, int M, int P)
	{
		return k*P + i*M*P + l;

	}

	__device__ unsigned int idx2d(int i, int j, int M)
	{
		return i*M + j;

	}

	__global__ void distance1(int Ni, int Nj, int M, int P, float sigma, double a, double *d1, double *d2, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				   x1 = x1 +log(a) - (d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*(d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma);
			}
	}

	res1[idx2d(i,j,Nj)] = exp(x1);
	}
	""")

	# Prepare data
	N1 = 100##Change this in add noise function call as well!
	N2 = 900

	d1 = data.astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	# Set up scaling factor to avoid working with too small numbers
	preci = pow(10,-34)
	FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))
	print "FmaxDistTraj:",FmaxDistTraj
	if(FmaxDistTraj<preci):
		a = pow(1.79*pow(10,300),1.0/(d1.shape[1]*d1.shape[2]))
	else:
		a = pow(preci,1.0/(d1.shape[1]*d1.shape[2]))*1.0/FmaxDistTraj
	print "preci:", preci, "a:", a

	# Assigning main kernel function to a variable
	dist_gpu1 = mod.get_function("distance1")

	# Split data to correct size to run on GPU
	# What does this number represent?, Should be defined as an int, can then clean up formulas further down#
	Max = 100.0
	# Define square root of maximum threads per block
	R = 15.0

	# Determine required number of runs for i and j
	numRuns = int(ceil(N1/float(Max)))
	numRuns2 = int(ceil(N2/float(Max)))

	result2 = zeros([N1,numRuns2])

	countsi = 0
	Ni = int(Max)

	for i in range(numRuns):
		countsj = 0
		Nj = int(Max)

		if((int(Max)*(i+1)) > N1): # If last run with less that max remaining trajectories
			Ni = int(N1 - Max*i) # Set Ni to remaining number of particels

		for j in range(numRuns2):
			if((int(Max)*(j+1)) > N2): # If last run with less that max remaining trajectories
				Nj = int(N2 - Max*j) # Set Nj to remaining number of particels

			data1 = d1[(i*int(Max)):(i*int(Max)+Ni),:,:] # d1 subunit for this run (same vector 9 times)
			data2 = d2[(j*int(Max)):(j*int(Max)+Nj),:,:] # d2 subunit for this run (9 different vecttors)

			M = data1.shape[1] # number of timepoints in d1 subunit
			P = data1.shape[2] # number of species in d1 subunit

			res1 = zeros([Ni,Nj]).astype(float64) # results vector [shape(data1)*shape(data2)]

			if(Ni<R):
				gi = 1  # grid width  (no of blocks in i direction, i.e. gi * gj gives number of blocks)
				bi = Ni # block width (no of threads in i direction, i.e. bi * bj gives size of each block (max. R*R))
			else:
				gi = ceil(Ni/R)
				bi = R
			if(Nj<R):
				gj = 1  # grid length
				bj = Nj # block length
			else:
				gj = ceil(Nj/R)
				bj = R

			# Invoke GPU calculations (takes data1 and data2 as input, outputs res1)
			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float64(a), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			for k in range(Ni):
				result2[(i*int(Max)+k),j] = sum(res1[k,:])

			countsj = countsj+Nj

		countsi = countsi+Ni

	sum1 = 0.0   # intermediate result sum
	counter = 0  # counts number of nan in matrix
	counter2 = 0 # counts number of inf sums in matrix

	for i in range(N1):
		if(isnan(sum(result2[i,:]))): counter=counter+1
		elif(isinf(log(sum(result2[i,:])))): counter2=counter2+1
		else:
			sum1 = sum1 - log(sum(result2[i,:])) + log(float(N2)) + M*P*log(a) +  M*P*log(2.0*pi*sigma*sigma)

	Info = sum1 / float(N1-counter-counter2)
	Info = Info - M*P/2.0*(log(2.0*pi*sigma*sigma)+1)

	return(Info)
