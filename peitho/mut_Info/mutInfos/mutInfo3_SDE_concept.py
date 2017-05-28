from numpy import *

from pycuda import compiler, driver
from pycuda import autoinit

import warnings
import sys
import operator
import copy

from peitho.mut_Info.mutInfos import launch

## REMOVE SEED
random.seed(123)
N1 = 43
B_mod = 17
B_ref = 19
N4 = 37
T = 33
S = 13

data_ref_t = random.rand(N1,B_ref,T,S).astype(float64)
theta_ref_t = random.rand(N1+N4,T,S).astype(float64)
cov_ref_t = random.rand(N1+N4,S*T,S).astype(float64)
data_mod_t = random.rand(N1,B_mod,T,S).astype(float64)
theta_mod_t = random.rand(N1+N4,T,S).astype(float64)
cov_mod_t = random.rand(N1+N4,S*T,S).astype(float64)

##Check input types!!


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
			res1[idx3d(ti,tj,tk,b,n4)] += log(sqrtf(invdet[idx2d(tk,t,%(T)s)])) - 0.5 * vector2[t] + pre;
		}
		res1[idx3d(ti,tj,tk,b,n4)] = exp(res1[idx3d(ti,tj,tk,b,n4)]);
	}
	"""

	print "\n", "-----Preprocessing Data (matrix inversion etc.)-----", "\n"

	# Determine number of particles (N1,N4), betas (B), timepoints (T), species (S)
	N1, B_ref, T, S = dataRef.shape
	B_mod = dataMod.shape[1]
	N4 = thetaMod.shape[0] - N1

	# Fill placeholders in kernel (pycuda metaprogramming)
	kernel_code = kernel_code_template % {
		'T': T,
		'S': S,
		'ST': S*T
		}

	# Compile GPU kernel
	mod = compiler.SourceModule(kernel_code)

	# Create GPU function handle
	gpu_kernel_func1SDE = mod.get_function("kernel_func1SDE")

	# Precalculation for GPU kernel for faster computation
	pre = log(1/(sqrt(pow(2*math.pi,S))))

	# Initialise arrays for inverted covariance matrices and inverted determinants
	invcovMod = zeros((N4,S*T,S), dtype=float32)
	invdetMod = zeros((N4,T), dtype=float32)
	invcovRef = zeros((N4,S*T,S), dtype=float32)
	invdetRef = zeros((N4,T), dtype=float32)

	# Invert covariance matrices and calculate the determinant of the inverted matrices
	##Remove abs() for invdet
	for i in range(N4):
		for j in range(T):
			invcovMod[i,j*S:(j+1)*S,:] = linalg.inv(covMod[N1+i,j*S:(j+1)*S,:])
			invdetMod[i,j] = abs(linalg.det(invcovMod[i,j*S:(j+1)*S,:]))

	##Remove abs() for invdet
	for i in range(N4):
		for j in range(T):
			invcovRef[i,j*S:(j+1)*S,:] = linalg.inv(covRef[N1+i,j*S:(j+1)*S,:])
			invdetRef[i,j] = abs(linalg.det(invcovRef[i,j*S:(j+1)*S,:]))

	#################### Calculating model probability ###########################

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
	grid_prelim_i , grid_prelim_j, grid_prelim_k = launch.optimise_gridsize_sde(1, float(block_i), float(block_j), float(block_k), T, S)
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

	print "-----Determining optimal kernel launch configuration (for part 2/2)-----"

	# Launch configuration: Block size and shape (as close to square as possible)
	print "Optimal blocksize:", block, "threads"
	print "Block shape:", str(block_i)+"x"+str(block_j)+"x"+str(block_k)


	# Launch configuration: Grid size (limited by GPU global memory) and grid shape (multipe of block size)
	grid_prelim_i , grid_prelim_j, grid_prelim_k = launch.optimise_gridsize_sde(1, float(block_i), float(block_j), float(block_k), T, S)
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
	print "Percentage of infinites: Term 1: %.1f %%, Term 2: %.1f %%, Term 3: %.1f %%"%((inf_count1*100)/(N1*B_mod*B_ref*N4), (inf_count2*100)/(N1*B_mod*N4), (inf_count3*100)/(N1*B_ref*N4))

	print "\n", mutinfo


	print "\n", "------CPU CALCS RUNNING NOW---------"
###############################CPU TEST#########################################
	cpu_1 = zeros((N1,B_mod,N4), dtype=float64)
	cpu_2 = zeros((N1,B_ref,N4), dtype=float64)

	for i in range(N1):
		for j in range(B_mod):
			for k in range(N4):
				for l in range(T):
					cpu_1[i,j,k] += pre + log(sqrt(invdetMod[k,l])) - 0.5 * dot(dot(expand_dims(dataMod[i,j,l,:]-thetaMod[N1+k,l,:],0),invcovMod[k,l*S:(l+1)*S,:]),expand_dims(dataMod[i,j,l,:]-thetaMod[N1+k,l,:],1))
	cpu_1 = exp(cpu_1)

	for i in range(N1):
		for j in range(B_ref):
			for k in range(N4):
				for l in range(T):
					cpu_2[i,j,k] += pre + log(sqrt(invdetRef[k,l])) - 0.5 * dot(dot(expand_dims(dataRef[i,j,l,:]-thetaRef[N1+k,l,:],0),invcovRef[k,l*S:(l+1)*S,:]),expand_dims(dataRef[i,j,l,:]-thetaRef[N1+k,l,:],1))
	cpu_2 = exp(cpu_2)

	cpu_masked1 = ma.masked_invalid(expand_dims(cpu_1, axis=2)*expand_dims(cpu_2, axis=1))
	cpu_masked2 = ma.masked_invalid(cpu_1)
	cpu_masked3 = ma.masked_invalid(cpu_2)

	cpu_term1 = average(average(log(ma.average(cpu_masked1, axis=3)), axis=2), axis=1)
	cpu_term2 = average(log(ma.average(cpu_masked2, axis=2)), axis=1)
	cpu_term3 = average(log(ma.average(cpu_masked3, axis=2)), axis=1)

	cpu = average(cpu_term1 - cpu_term2 - cpu_term3, axis=0)

	cpu_inf_count1 = float(ma.count_masked(masked1))
	cpu_inf_count2 = float(ma.count_masked(masked2))
	cpu_inf_count3 = float(ma.count_masked(masked3))
	print "Percentage of CPU infinites: Term 1: %.1f %%, Term 2: %.1f %%, Term 3: %.1f %%"%((cpu_inf_count1*100)/(N1*B_mod*B_ref*N4), (cpu_inf_count2*100)/(N1*B_mod*N4), (cpu_inf_count3*100)/(N1*B_ref*N4))
###############################CPU TEST#########################################

	return mutinfo, cpu

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	gpu, cpu = mutInfo3SDE(data_mod_t, theta_mod_t, cov_mod_t, data_ref_t, theta_ref_t, cov_ref_t)

print ""
print "CPU OUT"
print cpu
print ""
print "GPU OUT"
print gpu
print ""
print "Error:"
print cpu - gpu
