from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit
import copy

import launch

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


	__global__ void distance1(int Ni, int Nj, int M, int P, float sigma, double scale, double *d1, double *d2, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M; k++){
		for(int l=0; l<P; l++){
			x1 -= ( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)]);
		}
	}

	res1[idx2d(i,j,Nj)] = exp(scale+sigma*x1);
	}

	__global__ void distance2(int Nj, int M, int P, float sigma, double scale, double *d1, double *d3, double *res2)
	{
	int j = threadIdx.x + blockDim.x * blockIdx.x;

	if(j>=Nj) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M; k++){
		for(int l=0; l<P; l++){
			x1 -= (d3[idx3d(j,k,l,M,P)]-d1[idx2d(k,l,P)])*(d3[idx3d(j,k,l,M,P)]-d1[idx2d(k,l,P)]);
		}
	}

	res2[j] = exp(scale+sigma*x1);
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
	
	######################Optimisation####################
	# Determine M*P*log(scale) for GPU calculations
	mplogscale= M*P*log(scale)

	# Determine 1/(2*sigma*sigma) for
	sigmasq_inv = 1/(2*sigma*sigma)

	# Create template array for res1
	temp_res1 = zeros([Ni,Nj]).astype(float64)
	print temp_res1.shape



	##################################################
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
			gi = 1  # Grid size in dim i
			bi = Ni # Block size in dim i
		else:
			gi = ceil(Ni/block_i)
			bi = block_i

		# Maximum number of particles per run in j direction
		Nj = int(grid_j)

		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N2):
				Nj = int(N2 - grid_j*j)
				# Prepare results array for this run
				res1 = copy.deepcopy(temp_res1[:Ni,:Nj])
			elif j==0:
				# Prepare results array for this run
				res1 = copy.deepcopy(temp_res1[:Ni,:Nj])

			# Prepare data that depends on j for this run
			data2 = d2[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			
			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				gj = ceil(Nj/block_j)
				bj = block_j

			# Call GPU kernel function
			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigmasq_inv), float64(mplogscale), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# Summing rows in GPU output for this run
			result[i*int(grid_i):i*int(grid_i)+Ni,j]=sum(res1, axis=1)


	# Initialising required variables for next steps
	mplogscale= M*P*log(scale)
	mplogpisigma= M*P*log(2.0*pi*sigma*sigma)
	logN2 = log(float(N2))

	# Sum all content of new results matrix and add/subtract constants for each row if there are no NANs or infs
	
	sum_result=ma.log(sum(result,axis=1))
	count_inf1=ma.count_masked(sum_result)
	sum1 = sum(sum_result)-logN2*(N1-count_inf1)-mplogscale*(N1-count_inf1)-mplogpisigma*(N1-count_inf1)
	print sum1


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

		for j in range(numRuns_j2):
			#print "runs left:", numRuns_j2 - j

			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid)*(j+1)) > N3[i]):
				Nj = int(N3[i] - grid*j)
				# Prepare results array for this run
				res2 = copy.deepcopy(temp_res2[:Nj])
			elif j==0:
				# Prepare results array for this run
				res2 = copy.deepcopy(temp_res2[:Nj])

			# Prepare data that depends on j for this run
			data3 = d3[(i*N3[i]+j*int(grid)):(i*N3[i]+j*int(grid)+Nj),:,:]

			# Set j dimension of block and grid for this run
			if(Nj<block):
				gj = 1
				bj = Nj
			else:
				gj = ceil(Nj/block)
				bj = block

			# Call GPU kernel function
			dist_gpu2(int32(Nj), int32(M), int32(P), float32(sigmasq_inv), float64(mplogscale), driver.In(data1), driver.In(data3),  driver.Out(res2), block=(int(bj),1,1), grid=(int(gj),1))

			# Sum all elements in results array for this run
			result[i,j] = sum(res2)

	# Initialising required variables for next steps


	# Sum all content of new results matrix and add/subtract constants for each row if there are no NANs or infs
	

	
	'''
	for i in range(N1):
		if(isnan(sum(result[i,:]))): count2_na += 1
		elif(isinf(log(sum(result[i,:])))): count2_inf += 1
		else:
			sum2 += log(sum(result[i,:])) - log(float(N3[i])) - mplogscale -  mplogpisigma
			#sumstatic += - log(float(N3[i])) - M*P*log(scale) -  M*P*log(2.0*pi*sigma*sigma)
	'''
	print result
	print result.shape
	sum_result2=ma.log(sum(result, axis=1))
	print sum_result2.shape
	print sum_result2
	count_inf2=ma.count_masked(sum_result)
	print count_inf2
	print sum(sum_result2)
	print log(ma.masked_array(N3,sum_result2.mask))
	print ma.masked_array(N3,~sum_result2.mask)
	print sum(log(ma.masked_array(N3,~sum_result2.mask)))  
	print mplogscale*(N1-count_inf2)
	print mplogpisigma*(N1-count_inf2)

	sum2 = sum(sum_result2) - sum(log(ma.masked_array(N3,sum_result2.mask))) - mplogscale*(N1-count_inf2) - mplogpisigma*(N1-count_inf2)
	print sum2

########################Final Computations######################################

#	print "Proportion of NAs", int(((count1_na+count2_na)/float(2*N1))*100), "%" ###Can we really get NAs??
	print "Proportion of infs", int(((count_inf1+count_inf2)/float(2*N1))*100), "%"

	# Final division to give mutual information
	Info = (sum2 - sum1)/float(N1 - count_inf1 -count_inf2) ###Should be 2*N1 here?

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
