from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit

import launch

# A function to calculate the mutual information between all parameters of a system and an experiment
##(gets called by run_getEntropy1)
##Arguments:
##data - array of tracjectories with noise added
##theta - array of trajectories without noise
##N1,N2 - Number of particles
##sigma - stadard deviation
##scale - scaling constant to prevent nans and infs
#@profile
def getEntropy1(data,theta,N1,N2,sigma,scale):
	# Kernel declaration using pycuda SourceModule

	mod = compiler.SourceModule("""
	
	__device__ __constant__ double scale_const;
	__device__ __constant__ float sigma_const;
	__device__ __constant__ int M_const;
	__device__ __constant__ int P_const;


	__device__ unsigned int idx3d(int i, int k, int l)
	{
		return k*P_const + i*M_const*P_const + l;
	}

	__device__ unsigned int idx2d(int i, int j, int M)
	{
		return i*M + j;
	}


	__global__ void distance1(int Ni, int Nj, double *d1, double *d2, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M_const; k++){
		for(int l=0; l<P_const; l++){
			x1 += scale_const - ( d2[idx3d(j,k,l)]-d1[idx3d(i,k,l)])*( d2[idx3d(j,k,l)]-d1[idx3d(i,k,l)])/(sigma_const);
		}
	}

	res1[idx2d(i,j,Nj)] = exp(x1);
	}
	""")

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
	d1 = data.astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	# Determine number of timepoints (M) and number of species (P)
	M = d1.shape[1]
	P = d1.shape[2]

	#Initialize array for results
	result = zeros([N1,numRuns_j])

	# Maximum number of particles per run in i direction
	Ni = int(grid_i)

	# Determine log(scale) for GPU calculation
	logscale = log(scale)

	# Determine 2*sigma*sigma for GPU calculation
	sigmasq = 2*sigma*sigma

	# Transfer constants into constant memory on GPU
	CONSTm = array(M).astype(int32)
	CONST_M,_ = mod.get_global("M_const")
	CONSTp = array(P).astype(int32)
	CONST_P,_ = mod.get_global("P_const")
	CONSTscale = array(logscale).astype(float64)
	CONST_scale,_ = mod.get_global("scale_const") 
	CONSTsq = array(sigmasq).astype(float32)
	CONST_sq,_ = mod.get_global("sigma_const")

	driver.memcpy_htod(CONST_M, CONSTm)
	driver.memcpy_htod(CONST_P, CONSTp)
	driver.memcpy_htod(CONST_scale, CONSTscale)
	driver.memcpy_htod(CONST_sq, CONSTsq)

	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):
		print "Runs left:", numRuns_i - i

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


		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N2):
				Nj = int(N2 - grid_j*j)

			# Prepare data that depends on j for this run
			data2 = d2[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			# Prepare results array for this run
			res1 = zeros([Ni,Nj]).astype(float64) ###Could move into if statements (only if ni or nj change)
			#res1 = init_res1(Ni, Nj)

			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				gj = ceil(Nj/block_j)
				bj = block_j

			# Call GPU kernel function
			dist_gpu1(int32(Ni),int32(Nj), driver.In(data1), driver.In(data2), driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# Summing rows in GPU output for this run
			for k in range(Ni):
				result[(i*int(grid_i)+k),j] = sum(res1[k,:]) ###Could be done on GPU?

	# Initialising required variables for next steps
	sum1 = 0.0
	count_na = 0
	count_inf = 0
	logN2 = log(float(N2))
	mplogscale= M*P*logscale
	mplogpisigma= M*P*log(2.0*pi*sigma*sigma)

	# Sum all content of new results matrix and add/subtract constants for each row if there are no NANs or infs
	for i in range(N1):
		if(isnan(sum(result[i,:]))): count_na += 1
		elif(isinf(log(sum(result[i,:])))): count_inf += 1
		else:
			sum1 -= log(sum(result[i,:])) - logN2 - mplogscale -  mplogpisigma
	print "Proportion of NAs", int((count_na/float(N1))*100), "%" ###Can we really get NAs??
	print "Proportion of infs", int((count_inf/float(N1))*100), "%"

	# Final division to give mutual information
	Info = (sum1 / float(N1 - count_na - count_inf)) - M*P/2.0*(log(2.0*pi*sigma*sigma)+1)

	return(Info)

# A function calling getEntropy1 for all provided experiments and outputs the mutual information
##Argument: model_obj - an object containing all experiments and all their associated information
def run_getEntropy1(model_obj):
	MutInfo1 = []
	for experiment in range(model_obj.nmodels):

		if model_obj.initialprior == False:
			pos = model_obj.pairParamsICS[model_obj.cuda[experiment]].index([x[1] for x in model_obj.x0prior[experiment]])
			N1 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][0]
			N2 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][1]
		else:
			pos = model_obj.cudaout_structure[model_obj.cuda[experiment]][0]
			N1 = pos[0]
			N2 = pos[1]

		print "-----Calculating Mutual Information for Experiment", experiment+1,"-----"

		MutInfo1.append(getEntropy1(model_obj.trajectories[experiment],model_obj.cudaout[experiment],N1,N2,model_obj.sigma,model_obj.scale[experiment]))
		print "Mutual Information for Experiment", str(experiment+1)+":", MutInfo1[experiment]

	return MutInfo1
