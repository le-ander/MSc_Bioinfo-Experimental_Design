from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit

import warnings
import sys
import operator
import copy

from peitho.mut_Info.mutInfos import launch
from peitho.mut_Info.mutInfos import transform_gpu


def mutInfo3SDE(dataMod,thetaMod,covMod,dataRef,thetaRef,covRef):

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
	__global__ void kernel_func1SDE(int n1, int b, int n4, double pre, float *invdet, double *x, double *mu, float *invcov, double *res1){

		unsigned int ti = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int tj = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int tk = threadIdx.z + blockDim.z * blockIdx.z;

		if((ti>=n1)||(tj>=b)||(tk>=n4)) return;

		double vector1[%(ST)s] = {0.0};
		double vector2[%(T)s] = {0.0};

		res1[idx3d(ti,tj,tk,b,n4)] = 0.0;

		for(int t=0; t<%(T)s; t++){

			for(int s_i=0; s_i<%(S)s; s_i++){

				vector1[idx2d(t,s_i,%(S)s)] = 0.0;

				for(int s_j=0; s_j<%(S)s; s_j++){

					vector1[idx2d(t,s_i,%(S)s)] += (x[idx4d(ti,tj,t,s_j,b,%(T)s,%(S)s)] - mu[idx3d(tk,t,s_j,%(T)s,%(S)s)]) * invcov[idx4d(tk,t,s_j,s_i,%(T)s,%(S)s,%(S)s)];
				}
				vector2[t] += vector1[idx2d(t,s_i,%(S)s)] * (x[idx4d(ti,tj,t,s_i,b,%(T)s,%(S)s)] - mu[idx3d(tk,t,s_i,%(T)s,%(S)s)]);
			}
			vector2[t]=log(vector2[t]+1);
			res1[idx3d(ti,tj,tk,b,n4)] += log(sqrtf(invdet[idx2d(tk,t,%(T)s)])) - 0.5 * vector2[t] + pre;
		}
		res1[idx3d(ti,tj,tk,b,n4)] = exp(res1[idx3d(ti,tj,tk,b,n4)]);
	}
	"""

	# Determine number of particles (N1,N4), betas (B), timepoints (T), species (S)
	N1, B_ref, T_ref, S_ref = dataRef.shape
	B_mod, T_mod, S_mod = dataMod.shape[1:]
	N4 = thetaMod.shape[0] - N1


	#################### Calculating model probability ###########################

	print "\n", "-----Preprocessing Data (matrix inversion etc.) (for part 1/2)-----", "\n"

	# Precalculation for GPU kernel for faster computation
	pre = log(1/(sqrt(pow(2*math.pi,S_mod))))

	# Fill placeholders in kernel (pycuda metaprogramming)
	kernel_code = kernel_code_template % {
		'T': T_mod,
		'S': S_mod,
		'ST': S_mod*T_mod
		}

	# Compile GPU kernel
	mod = compiler.SourceModule(kernel_code)

	# Create GPU function handle
	gpu_kernel_func1SDE = mod.get_function("kernel_func1SDE")

	# Initialise arrays for inverted covariance matrices and inverted determinants
	invcovMod = zeros((N4,S_mod*T_mod,S_mod), dtype=float32)
	invdetMod = zeros((N4,T_mod), dtype=float32)

	# Invert covariance matrices and calculate the determinant of the inverted matrices
	for i in range(N4):
		for j in range(T_mod):
			invcovMod[i,j*S_mod:(j+1)*S_mod,:] = linalg.inv(covMod[N1+i,j*S_mod:(j+1)*S_mod,:])
			invdetMod[i,j] = linalg.det(invcovMod[i,j*S_mod:(j+1)*S_mod,:])


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
	grid_prelim_i , grid_prelim_j, grid_prelim_k = launch.optimise_gridsize_sde(1, float(block_i), float(block_j), float(block_k), T_mod, S_mod)
	grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i, N1))
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, B_mod))
	grid_k = float(min(autoinit.device.max_grid_dim_y, grid_prelim_k, N4))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)+"x"+str(grid_k)
	print "Registers:", gpu_kernel_func1SDE.num_regs , "\n"

	print "-----Calculation part 1 of 2 now running-----", "\n"

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(B_mod/grid_j))
	numRuns_k = int(ceil(N4/grid_k))

	# Initialise array to hold results of model probability
	res_log_mod = zeros([N1,B_mod,N4], dtype=float64)

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
			if((int(grid_j)*(j+1)) > B_mod):
				Nj = int(B_mod - grid_j*j)

			# Prepare data that depends on i and j for this run
			data_subset = ascontiguousarray(dataMod[(i*int(grid_i)):(i*int(grid_i)+Ni),(j*int(grid_j)):(j*int(grid_j)+Nj),:,:])

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
				if((int(grid_k)*(k+1)) > N4):
					Nk = int(N4 - grid_k*k)
					last = True

				# Prepare input that depends on k for this run
				theta_subset = thetaMod[N1+(k*int(grid_k)):N1+(k*int(grid_k)+Nk),:,:]
				invcov_subset = invcovMod[(k*int(grid_k)):(k*int(grid_k)+Nk),:,:]
				invdet_subset = invdetMod[(k*int(grid_k)):(k*int(grid_k)+Nk),:]

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

				# Store results for model probabilities
				res_log_mod[i*int(grid_i):i*int(grid_i)+Ni,j*int(grid_j):j*int(grid_j)+Nj,k*int(grid_k):k*int(grid_k)+Nk] = res1




	#################### Calculating reference model probability ###########################

	print "\n", "-----Preprocessing Data (matrix inversion etc.) (for part 2/2)-----", "\n"

	# Precalculation for GPU kernel for faster computation
	pre = log(1/(sqrt(pow(2*math.pi,S_ref))))

	# Fill placeholders in kernel (pycuda metaprogramming)
	kernel_code = kernel_code_template % {
		'T': T_ref,
		'S': S_ref,
		'ST': S_ref*T_ref
		}

	# Compile GPU kernel
	mod = compiler.SourceModule(kernel_code)

	# Create GPU function handle
	gpu_kernel_func1SDE = mod.get_function("kernel_func1SDE")

	# Initialise arrays for inverted covariance matrices and inverted determinants
	invcovRef = zeros((N4,S_ref*T_ref,S_ref), dtype=float32)
	invdetRef = zeros((N4,T_ref), dtype=float32)

	# Invert covariance matrices and calculate the determinant of the inverted matrices
	for i in range(N4):
		for j in range(T_ref):
			invcovRef[i,j*S_ref:(j+1)*S_ref,:] = linalg.inv(covRef[N1+i,j*S_ref:(j+1)*S_ref,:])
			invdetRef[i,j] = linalg.det(invcovRef[i,j*S_ref:(j+1)*S_ref,:])


	print "-----Determining optimal kernel launch configuration (for part 2/2)-----"

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
	grid_prelim_i , grid_prelim_j, grid_prelim_k = launch.optimise_gridsize_sde(1, float(block_i), float(block_j), float(block_k), T_ref, S_ref)
	grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j, B_ref))
	print "Grid shape:", str(grid_i)+"x"+str(grid_j)+"x"+str(grid_k), "\n"

	print "-----Calculation part 2 of 2 now running-----", "\n"

	# Determine required number of runs for i and j
	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(B_ref/grid_j))
	numRuns_k = int(ceil(N4/grid_k))

	# Initialise array to hold results of model probability
	res_log_ref = zeros([N1,B_ref,N4], dtype=float64)

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
			if((int(grid_j)*(j+1)) > B_ref):
				Nj = int(B_ref - grid_j*j)

			# Prepare data that depends on i and j for this run
			data_subset = ascontiguousarray(dataRef[(i*int(grid_i)):(i*int(grid_i)+Ni),(j*int(grid_j)):(j*int(grid_j)+Nj),:,:])

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
				if((int(grid_k)*(k+1)) > N4):
					Nk = int(N4 - grid_k*k)
					last = True

				# Prepare input that depends on k for this run
				theta_subset = thetaRef[N1+(k*int(grid_k)):N1+(k*int(grid_k)+Nk),:,:]
				invcov_subset = invcovRef[(k*int(grid_k)):(k*int(grid_k)+Nk),:,:]
				invdet_subset = invdetRef[(k*int(grid_k)):(k*int(grid_k)+Nk),:]

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

				# Store results for model probabilities
				res_log_ref[i*int(grid_i):i*int(grid_i)+Ni,j*int(grid_j):j*int(grid_j)+Nj,k*int(grid_k):k*int(grid_k)+Nk] = res1


	#################### Calculating reference model probability ###########################

	print "-----Final calculations now running-----", "\n"

	masked1 = ma.masked_invalid(expand_dims(res_log_mod, axis=2)*expand_dims(res_log_ref, axis=1))
	masked2 = ma.masked_invalid(res_log_mod)
	masked3 = ma.masked_invalid(res_log_ref)

	term1 = average(average(log(ma.average(masked1, axis=3)), axis=2), axis=1)
	term2 = average(log(ma.average(masked2, axis=2)), axis=1)
	term3 = average(log(ma.average(masked3, axis=2)), axis=1)

	mutinfo = average(term1 - term2 - term3, axis=0)

	inf_count1 = float(ma.count_masked(masked1))
	inf_count2 = float(ma.count_masked(masked2))
	inf_count3 = float(ma.count_masked(masked3))
	sum_inf_count = inf_count1 + inf_count2 + inf_count3
	sum_inf_prop = ((inf_count1*100)/(N1*B_mod*B_ref*N4) + (inf_count2*100)/(N1*B_mod*N4) + (inf_count3*100)/(N1*B_ref*N4))/3
	print "Percentage of infinites: Term 1: %.1f %%, Term 2: %.1f %%, Term 3: %.1f %%"%((inf_count1*100)/(N1*B_mod*B_ref*N4), (inf_count2*100)/(N1*B_mod*N4), (inf_count3*100)/(N1*B_ref*N4))

	return mutinfo, sum_inf_count, sum_inf_prop

def run_mutInfo3_SDE(model_obj, ref_model_obj, input_SBML  ):
	#Initiates list to hold mutual information
	MutInfo3_SDE = []
	#Initiates list to hold number of infinites
	MutInfo3_SDE_infs = []
	#Initiates list to hold percentage of infinites
	MutInfo3_SDE_infs_prop = []

	t_data_ref, t_theta_ref, t_cov_ref = transform_gpu.transform_gpu(ref_model_obj.cudaout[0],ref_model_obj.mus[0],ref_model_obj.covariances[0], ref_model_obj.B[0])

	#Cycles through each experiment
	for experiment in range(model_obj.nmodels):
		#transform
		print "-----Performing matrix transformation for Experiment", experiment+1, "for", input_SBML,"-----\n"
		#print model_obj.cudaout[experiment]
		t_data, t_theta, t_cov = transform_gpu.transform_gpu(model_obj.cudaout[experiment],model_obj.mus[experiment],model_obj.covariances[experiment], model_obj.B[experiment])
		#print "Data:", t_data

		#Calculates mutual information
		print "-----Calculating Mutual Information for Experiment", experiment+1, "for", input_SBML,"-----\n"


		## Assign mutual information and number of infinites to lists
		temp_list=mutInfo3SDE(t_data, t_theta, t_cov, t_data_ref, t_theta_ref, t_cov_ref)
		MutInfo_lists = [MutInfo3_SDE, MutInfo3_SDE_infs, MutInfo3_SDE_infs_prop]
		for x, lst in zip(temp_list, MutInfo_lists):
			lst.append(x)

		## Print out mutual information
		print "Mutual Information for Experiment", str(experiment+1)+":", MutInfo3_SDE[experiment], "\n"
	return MutInfo3_SDE, MutInfo3_SDE_infs, MutInfo3_SDE_infs_prop
