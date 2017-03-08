from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit
import sys
import launch

# A function to calculate the mutual information between the outcome of two experiments
##(gets called by run_getEntropy1)
##Arguments: (Ref = reference experiment, Mod = alternative experiment)
##data - array of tracjectories with noise added
##theta - array of trajectories without noise
##N1,N2 - Number of particles
##sigma - stadard deviation
##scale - scaling constant to prevent nans and infs

#MutInfo3.append(getEntropy3(ref_obj.trajectories[0],model_obj.trajectories[experiment],ref_obj.cudaout[0], model_obj.cudaout[experiment],N1,N2,N3,N4,ref_obj.sigma,model_obj.sigma,max(model_obj.scale[experiment],ref_obj.scale[0])))
#dataRef = ref traj
#thetaRef
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

	__global__ void distance1(int Ni, int Nj, int M_Ref, int P_Ref, int M_Mod, int P_Mod, float sigma_ref, float sigma_mod, double scale_ref, double scale_mod, double *d1, double *d2, double *d3, double *d4, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x1;
	x1 = 0.0;
	double x3;
	x3 = 0.0;

	for(int k=0; k<M_Ref; k++){
		for(int l=0; l<P_Ref; l++){
			x1 = x1 + scale_ref - ( d2[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])*( d2[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])/(2.0*sigma_ref*sigma_ref);
		}
	}

	for(int k=0; k<M_Mod; k++){
		for(int l=0; l<P_Mod; l++){
			x3 = x3 + scale_mod - ( d4[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])*( d4[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])/(2.0*sigma_mod*sigma_mod);
		}
	}

	res1[idx2d(i,j,Nj)] = exp(x1+x3);
	}

	__global__ void distance2(int Ni, int Nj, int M_Ref, int P_Ref, int M_Mod, int P_Mod, float sigma_ref, float sigma_mod, double scale_ref, double scale_mod, double *d1, double *d6, double *res2)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x2;
	x2 = 0.0;

	for(int k=0; k<M_Ref; k++){
		for(int l=0; l<P_Ref; l++){
			x2 = x2 + scale_ref - ( d6[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])*( d6[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])/(2.0*sigma_ref*sigma_ref);
		}
	}

	res2[idx2d(i,j,Nj)] = exp(x2);
	}

	__global__ void distance3(int Ni, int Nj, int M_Ref, int P_Ref, int M_Mod, int P_Mod, float sigma_ref, float sigma_mod, double scale_ref, double scale_mod, double *d3, double *d8, double *res3)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x3;
	x3 = 0.0;

	for(int k=0; k<M_Mod; k++){
		for(int l=0; l<P_Mod; l++){
			x3 = x3 + scale_mod - ( d8[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])*( d8[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])/(2.0*sigma_mod*sigma_mod);
		}
	}

	res3[idx2d(i,j,Nj)] = exp(x3);
	}

	""")

	# Creating handles for global kernel functions
	dist_gpu1 = mod.get_function("distance1")
	dist_gpu2 = mod.get_function("distance2")
	dist_gpu3 = mod.get_function("distance3")

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

	# Maximum number of particles per run in i direction
	Ni = int(grid_i)

	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):
		print "Runs left:", numRuns_i - i

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

		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N2):
				Nj = int(N2 - grid_j*j)

			# Prepare data that depends on j for this run
			data2 = d2[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]
			data4 = d4[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			# Prepare result arrays for this run
			res1 = zeros([Ni,Nj]).astype(float64)

			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				bj = block_j
				gj = ceil(Nj/block_j)

			# Call GPU kernel functions
			dist_gpu1(int32(Ni), int32(Nj), int32(M_Ref), int32(P_Ref), int32(M_Mod), int32(P_Mod), float32(sigma_ref), float32(sigma_mod), float64(scale_ref), float64(scale_mod), driver.In(data1), driver.In(data2), driver.In(data3), driver.In(data4), driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# Summing rows in GPU output for this run
			for k in range(Ni):
				result1[(i*int(grid_i)+k),j] = sum(res1[k,:]) ###Could be done on GPU?


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

	# Maximum number of particles per run in i direction
	Ni = int(grid_i)

	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):
		print "Runs left:", numRuns_i - i

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

		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N3):
				Nj = int(N3 - grid_j*j)

			# Prepare data that depends on j for this run
			data6 = d6[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			# Prepare result arrays for this run
			res2 = zeros([Ni,Nj]).astype(float64)

			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				bj = block_j
				gj = ceil(Nj/block_j)

			# Call GPU kernel functions
			dist_gpu2(int32(Ni), int32(Nj), int32(M_Ref), int32(P_Ref), int32(M_Mod), int32(P_Mod), float32(sigma_ref), float32(sigma_mod), float64(scale_ref), float64(scale_mod), driver.In(data1), driver.In(data6), driver.Out(res2), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# Summing rows in GPU output for this run
			for k in range(Ni):
				result2[(i*int(grid_i)+k),j] = sum(res2[k,:]) ###Could be done on GPU?


########################Calulation 3############################################

	# Launch configuration: Block size and shape (as close to square as possible)
	block = launch.optimal_blocksize(autoinit.device, dist_gpu3)
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

	# Maximum number of particles per run in i direction
	Ni = int(grid_i)

	# Main nested for-loop for mutual information calculations
	for i in range(numRuns_i):
		print "Runs left:", numRuns_i - i

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

		for j in range(numRuns_j):
			# If last run with less that max remaining particles, set Nj to remaining number of particles
			if((int(grid_j)*(j+1)) > N4):
				Nj = int(N4 - grid_j*j)

			# Prepare data that depends on j for this run
			data8 = d8[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			# Prepare result arrays for this run
			res3 = zeros([Ni,Nj]).astype(float64)

			# Set j dimension of block and grid for this run
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				bj = block_j
				gj = ceil(Nj/block_j)

			# Call GPU kernel functions
			dist_gpu3(int32(Ni), int32(Nj), int32(M_Ref), int32(P_Ref), int32(M_Mod), int32(P_Mod), float32(sigma_ref), float32(sigma_mod), float64(scale_ref), float64(scale_mod), driver.In(data3), driver.In(data8), driver.Out(res3), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# Summing rows in GPU output for this run
			for k in range(Ni):
				result3[(i*int(grid_i)+k),j] = sum(res3[k,:]) ###Could be done on GPU?


########################Final Computations######################################

	# Initialising required variables for next steps
	sum1 = 0.0
	sumres1 = 0.0
	sumres2 = 0.0
	sumres3 = 0.0
	a1 = 0.0
	a2 = 0.0
	a3 = 0.0
	count1_na = 0
	count1_inf = 0
	count2_na = 0
	count2_inf = 0
	count3_na = 0
	count3_inf = 0

	# Sum all content of new results matrix and add/subtract constants for each row if there are no NANs or infs
	for i in range(N1):
		if(isinf(log(sum(result1[i,:])))):
			count1_inf=count1_inf+1
		elif(isnan(log(sum(result1[i,:])))):
			count1_na=count1_na+1
		elif(isinf(log(sum(result2[i,:])))):
			count2_inf=count2_inf+1
		elif(isnan(log(sum(result2[i,:])))):
			count_na=count2_na+1
		elif(isinf(log(sum(result3[i,:])))):
			count3_inf=count3_inf+1
		elif(isnan(log(sum(result3[i,:])))):
			count3_na=count3_na+1
		else:
			sum1 += log(sum(result1[i,:])) - log(sum(result2[i,:])) - log(sum(result3[i,:])) - log(float(N2)) + log(float(N3)) + log(float(N4))
			# Optional calculations of intermediate results
			#a1 += log(sum(result1[i,:])) - log(N2) - M_Ref*P_Ref*log(2.0*pi*sigma_ref*sigma_ref) - M_Mod*P_Mod*log(2.0*pi*sigma_mod*sigma_mod) - M_Ref*P_Ref*scale - M_Mod*P_Mod*scale
			#a2 -= log(sum(result2[i,:])) + log(N3) + M_Ref*P_Ref*log(2.0*pi*sigma_ref*sigma_ref) + M_Ref*P_Ref*scale
			#a3 -= log(sum(result3[i,:])) + log(N4) + M_Mod*P_Mod*log(2.0*pi*sigma_mod*sigma_mod) + M_Mod*P_Mod*scale

	# Sum number of all NAs and Infs
	count_all = count1_inf + count1_na + count2_inf + count2_na + count3_inf + count3_na

	# Printing intermediate results
	#print "a1: ", a1/float(N1-count1_na-count1_inf) , "a2: ", a2/float(N1-count2_na-count2_inf), "a3: ", a3/float(N1-count3_na-count3_inf)
	#print "all: ", a1/float(N1-count1_na-count1_inf) + a2/float(N1-count2_na-count2_inf) + a3/float(N1-count3_na-count3_inf)
	#print "sum1: ", sum1
	print count1_inf, count1_na
	print count2_inf, count2_na
	print count3_inf, count3_na

	# Final division to give mutual information
	Info = sum1/float(N1-count_all)

	###Testing block....REMOVE!
	print Info
	print "DONE"
	sys.exit()

	return(Info)

# A function calling getEntropy3 for all provided experiments and outputs the mutual information
##Arguments:
##model_obj - an object containing all alternative experiments and all their associated information
##ref_obj - an object containing the reference experiment and all associated information
def run_getEntropy3(model_obj, ref_obj):

	MutInfo3 = []
	for experiment in range(model_obj.nmodels):

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

		N1 = min(N1_mod,N1_ref)
		N2 = min(N2_mod,N2_ref)

		print "-----Calculating Mutual Information for Experiment", experiment+1,"-----"
		MutInfo3.append(getEntropy3(ref_obj.trajectories[0],ref_obj.cudaout[0],model_obj.trajectories[experiment], model_obj.cudaout[experiment],N1,N2,N3,N4,ref_obj.sigma,model_obj.sigma,ref_obj.scale[0],model_obj.scale[experiment]))
		print "Mutual Information for Experiment", str(experiment+1)+":", MutInfo1[experiment]

	return MutInfo3
