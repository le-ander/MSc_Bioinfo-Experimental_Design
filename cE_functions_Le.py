#!/usr/bin/python2.5

from numpy import *
from numpy.random import *
import math
import re

import cudasim.Lsoda as Lsoda

from pycuda import compiler, driver
from pycuda import autoinit

from abcsysbio import parse_infoEnt
from abcsysbio_parser import ParseAndWrite

try:
	import cPickle as pickle
except:
	import pickle

import time
import sys
sys.path.insert(0, ".")

def run_cudasim(m_object, parameters, species):
	modelTraj = []
	##Should run over cudafiles
	# Define CUDA filename for cudasim
	cudaCode = m_object.name[0] + '.cu'
	# Create ODEProblem object
	modelInstance = Lsoda.Lsoda(m_object.times, cudaCode, dt=m_object.dt)
	# Solve ODEs using Lsoda algorithm
	##Different parameters and species matrices for i in nmodels?
	result = modelInstance.run(parameters, species)
	modelTraj.append(result[:,0])

	return modelTraj

def remove_na(m_object, modelTraj):
	# Create a list of indices of particles that have an NA in their row
	##Why using 7:8 when summing? -> Change this
	index = [p for p, i in enumerate(isnan(sum(asarray(modelTraj[0])[:,7:8,:],axis=2))) if i==True]
	# Delete row of 1. results and 2. parameters from the output array for which an index exists
	for i in index:
		delete(modelTraj[mod], (i), axis=0)

	return modelTraj

def add_noise_to_traj(m_object, modelTraj, sigma, N1):##Need to ficure out were to get N1 from
	ftheta = []
	# Create array with noise of same size as the trajectory array (only the first N1 particles)
	noise = normal(loc=0.0, scale=sigma,size=shape(modelTraj[0][0:N1,:,:]))
	# Add noise to trajectories and output new 'noisy' trajectories
	traj = array(modelTraj[0][0:N1,:,:]) + noise
	ftheta.append(traj)

	# Return final trajectories for 0:N1 particles
	return ftheta

def scaling(modelTraj, ftheta, sigma):
	maxDistTraj = max([math.fabs(amax(modelTraj) - amin(ftheta)),math.fabs(amax(ftheta) - amin(modelTraj))])

	preci = pow(10,-34)
	FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))

	if(FmaxDistTraj<preci):
		scale = pow(1.79*pow(10,300),1.0/(ftheta[0].shape[1]*ftheta[0].shape[2]))
	else:
		scale = pow(preci,1.0/(ftheta[0].shape[1]*ftheta[0].shape[2]))*1.0/FmaxDistTraj

	return scale

def pickle_object(object):
	pickle.dump(object, open("save_point.pkl", "wb"))

def unpickle_object(filename="savepoint.pkl"):
	object = pickle.load(open(filename, "rb"))

	return object

def get_mutinf_all_param(m_object, ftheta, N1, N2, sigma, modelTraj, scale):
	MutInfo1 = []
	# Run function to get the mutual information for all parameters
	MutInfo1.append(getEntropy1(ftheta[0],N1,N2,sigma,array(modelTraj[0]),scale))

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

def optimise_grid_structure(gmem_per_thread=8.59): ##need to define correct memory requirement for kernel (check for other cards)
	# DETERMINE TOTAL NUMBER OF THREADS LIMITED BY GLOBAL MEMORY
	# Read total global memory of device
	avail_mem = autoinit.device.total_memory()
	# Calculate maximum number of threads, assuming global memory usage of 100 KB per thread
	max_threads = int(sqrt(avail_mem / gmem_per_thread))
	##could change it to be a multiple of block size?
	##should it really return sqrt here?
	return max_threads

def getEntropy1(data,N1,N2,sigma,theta,scale):

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

	__global__ void distance1(int Ni, int Nj, int M, int P, float sigma, double scale, double *d1, double *d2, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M; k++){
		for(int l=0; l<P; l++){
			x1 = x1 +log(scale) - (d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*(d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma);
		}
	}

	res1[idx2d(i,j,Nj)] = exp(x1);
	}
	""")

	# Assigning main kernel function to a variable
	dist_gpu1 = mod.get_function("distance1")

	##should be defined as an int, can then clean up formulas further down
	Max = 100.0 # Define square root of maximum threads per grid
	R = 15.0 # Maximum threads per block

	# Determine required number of runs for i and j
	##need float here?
	numRuns = int(ceil(N1/float(Max)))
	numRuns2 = int(ceil(N2/float(Max)))

	result2 = zeros([N1,numRuns2])

	# Prepare data
	d1 = data.astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	M = d1.shape[1] # number of timepoints
	P = d1.shape[2] # number of species

	Ni = int(Max)


	for i in range(numRuns):
		print "Runs left:", numRuns-i
		if((int(Max)*(i+1)) > N1): # If last run with less that max remaining trajectories
			Ni = int(N1 - Max*i) # Set Ni to remaining number of particels

		if(Ni<R):
			gi = 1  # Grid size in dim i
			bi = Ni # Block size in dim i
		else:
			gi = ceil(Ni/R)
			bi = R

		data1 = d1[(i*int(Max)):(i*int(Max)+Ni),:,:] # d1 subunit for the next j runs

		Nj = int(Max)


		for j in range(numRuns2):
			if((int(Max)*(j+1)) > N2): # If last run with less that max remaining trajectories
				Nj = int(N2 - Max*j) # Set Nj to remaining number of particels

			data2 = d2[(j*int(Max)):(j*int(Max)+Nj),:,:] # d2 subunit for this run

			##could move into if statements (only if ni or nj change)
			res1 = zeros([Ni,Nj]).astype(float64) # results vector [shape(data1)*shape(data2)]

			if(Nj<R):
				gj = 1  # Grid size in dim j
				bj = Nj # Block size in dim j
			else:
				gj = ceil(Nj/R)
				bj = R

			# Invoke GPU calculations (takes data1 and data2 as input, outputs res1)
			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float64(scale), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# First summation (could be done on GPU?)
			for k in range(Ni):
					result2[(i*int(Max)+k),j] = sum(res1[k,:])

	sum1 = 0.0
	count_na = 0
	count_inf = 0

	for i in range(N1):
		if(isnan(sum(result2[i,:]))): count_na += 1
		elif(isinf(log(sum(result2[i,:])))): count_inf += 1
		else:
			sum1 += - log(sum(result2[i,:])) + log(float(N2)) + M*P*log(scale) +  M*P*log(2.0*pi*sigma*sigma)

	Info = (sum1 / float(N1 - count_na - count_inf)) - M*P/2.0*(log(2.0*pi*sigma*sigma)+1)

	return(Info)

def getEntropy3(data,N1,N2,N3,sigma,theta,scale):
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

	__device__ unsigned int idx2d2(int k, int l, int P)
	{
		return k*P + l;

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
				x1 = x1 + log(a) - ( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma);
			}
	}

	res1[idx2d(i,j,Nj)] = exp(x1);
	}

	__global__ void distance2(int Ni, int M, int P, float sigma, double a, double *d1, double *d3, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i>=Ni) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				x1 = x1 + log(a) - ( d3[idx3d(i,k,l,M,P)]-d1[idx2d2(k,l,P)])*( d3[idx3d(i,k,l,M,P)]-d1[idx2d2(k,l,P)])/(2.0*sigma*sigma);
			}
	}

	res1[i] = exp(x1);
	}
	""")

	# Assigning main kernel function to a variable
	dist_gpu1 = mod.get_function("distance1")

	##should be defined as an int, can then clean up formulas further down
	Max = 100.0 # Define square root of maximum threads per grid
	R = 15.0 # Maximum threads per block

	# Determine required number of runs for i and j
	##need float here?
	numRuns = int(ceil(N1/float(Max)))
	numRuns2 = int(ceil(N2/float(Max)))

	result2 = zeros([N1,numRuns2])

	# Prepare data
	d1 = data.astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	M = d1.shape[1] # number of timepoints
	P = d1.shape[2] # number of species

	Ni = int(Max)


	for i in range(numRuns):
		print "Runs left:", numRuns-i
		if((int(Max)*(i+1)) > N1): # If last run with less that max remaining trajectories
			Ni = int(N1 - Max*i) # Set Ni to remaining number of particels

		if(Ni<R):
			gi = 1  # Grid size in dim i
			bi = Ni # Block size in dim i
		else:
			gi = ceil(Ni/R)
			bi = R

		data1 = d1[(i*int(Max)):(i*int(Max)+Ni),:,:] # d1 subunit for the next j runs

		Nj = int(Max)


		for j in range(numRuns2):
			if((int(Max)*(j+1)) > N2): # If last run with less that max remaining trajectories
				Nj = int(N2 - Max*j) # Set Nj to remaining number of particels

			data2 = d2[(j*int(Max)):(j*int(Max)+Nj),:,:] # d2 subunit for this run

			##could move into if statements (only if ni or nj change)
			res1 = zeros([Ni,Nj]).astype(float64) # results vector [shape(data1)*shape(data2)]

			if(Nj<R):
				gj = 1  # Grid size in dim j
				bj = Nj # Block size in dim j
			else:
				gj = ceil(Nj/R)
				bj = R

			# Invoke GPU calculations (takes data1 and data2 as input, outputs res1)
			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float64(scale), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# First summation (could be done on GPU?)
			for k in range(Ni):
				result2[(i*int(Max)+k),j] = sum(res1[k,:])

	sum1 = 0.0
	count_na = 0
	count_inf = 0

	for i in range(N1):
		if(isnan(sum(result2[i,:]))): count_na += 1
		elif(isinf(log(sum(result2[i,:])))): count_inf += 1
		else:
			sum1 += - log(sum(result2[i,:])) + log(float(N2)) + M*P*log(scale) +  M*P*log(2.0*pi*sigma*sigma)

######## part A finished with results saved in sum1

	dist_gpu2 = mod.get_function("distance2")

	##need this defined again here??
	Max = 256.0
	R = 15.0

	numRuns3 = int(ceil(N3/Max))

	result2 = zeros([N1,numRuns3])

	d3 = array(theta)[(N1+N2):(N1+N2+N1*N3),:,:].astype(float64)

	for N1i in range(N1):
		for i in range(numRuns3):
			print "runs left:", numRuns - i

			si = int(Max)

			s = int(Max)
			if((s*(i+1)) > N1):
				si = int(N1 - Max*i)


			data3 = d3[(N1i*N3+i*int(Max)):(N1i*N3+i*int(Max)+si),:,:]
			data1 = d1[N1i,:,:]

			Ni=data3.shape[0]

			resB1 = zeros([Ni]).astype(float64)

			if(Ni<R):
				gi = 1
				bi = Ni
			else:
				bi = R
				gi = ceil(Ni/R)

			dist_gpu2(int32(Ni), int32(M), int32(P), float32(sigma), float64(a), driver.In(data1), driver.In(data3),  driver.Out(resB1), block=(int(bi),1,1), grid=(int(gi),1))

			result2[N1i,i] = sum(resB1[:])


	sumB1 = 0.0
	counter = 0
	counter2 = 0

	for i in range(N1):
		if(isnan(sum(result2[i,:]))): counter=counter+1
		if(isinf(log(sum(result2[i,:])))): counter2=counter2+1
		else:
			sumB1 = sumB1 + log(sum(result2[i,:])) - log(float(N3)) - M*P*log(a) -  M*P*log(2.0*pi*sigma*sigma)

######## part B finished with results saved in sumB1

	Info = (sumB1 - sum1)/float(N1)

	return(Info)

def getEntropy2(dataRef,dataY,N,sigma,theta1,theta2):

	#kernel declaration
	mod = compiler.SourceModule("""
	__device__ unsigned int idx3d(int i, int k, int l, int M, int P)
	{
		return k*P + i*M*P + l;

	}

	__device__ unsigned int idx2d(int i, int j, int M)
	{
		return i + j*M;d3

	}

	__global__ void distance2(int N, int M, int P, float sigma, float pi, float *d1, float *d2, float *d3, float *d4, float *res2, float *res3)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=N)||(j>=N)) return;

	float x2;
	float x3;
	x2 = 1.0;
	x3 = 1.0;

	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				   x2 = x2 * 1.0/sqrt(2.0*pi*sigma*sigma)*exp(-( d2[idx3d(i,k,l,M,P)]-d1[idx3d(j,k,l,M,P)])*( d2[idx3d(i,k,l,M,P)]-d1[idx3d(j,k,l,M,P)])/(2.0*sigma*sigma));
				   x3 = x3 * 1.0/sqrt(2.0*pi*sigma*sigma)*exp(-( d4[idx3d(i,k,l,M,P)]-d3[idx3d(j,k,l,M,P)])*( d4[idx3d(i,k,l,M,P)]-d3[idx3d(j,k,l,M,P)])/(2.0*sigma*sigma));
			}
	}

	res2[idx2d(i,j,N)] = x2;
	res3[idx2d(i,j,N)] = x3;


	}

	__global__ void distance1(int N, int M, int P, float sigma, float pi, float *d1, float *d2, float *d3, float *d4, float *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=N)||(j>=N)) return;

	float x1;
	x1 = 1.0;

	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				   x1 = x1 * 1.0/sqrt(2.0*pi*sigma*sigma)*exp(-( d2[idx3d(i,k,l,M,P)]-d1[idx3d(j,k,l,M,P)])*( d2[idx3d(i,k,l,M,P)]-d1[idx3d(j,k,l,M,P)])/(2.0*sigma*sigma))* 1.0/sqrt(2.0*pi*sigma*sigma)*exp(-( d4[idx3d(i,k,l,M,P)]-d3[idx3d(j,k,l,M,P)])*( d4[idx3d(i,k,l,M,P)]-d3[idx3d(j,k,l,M,P)])/(2.0*sigma*sigma));
			}
	}

	res1[idx2d(i,j,N)] = x1;


	}
	""")



	# prepare data

	N1 = 400
	N2 = N1
	N3 = N1
	N4 = N1

	d1 = dataRef[0:N1,:,:].astype(float32)
	d2 = array(theta1)[N1:(N1+N2),:,:].astype(float32)
	d3 = dataY[0:N1,:,:].astype(float32)
	d4 = array(theta2)[N1:(N1+N2),:,:].astype(float32)

	d5 = dataRef[0:N1,:,:].astype(float32)
	d6 = array(theta1)[(N1+N2):(N1+N2+N3),:,:].astype(float32)
	d7 = dataY[0:N1,:,:].astype(float32)
	d8 = array(theta2)[(N1+N2+N3):(N1+N2+N3+N4),:,:].astype(float32)

	result1 = zeros([N1,N1]).astype(float32)
	result2 = zeros([N1,N1]).astype(float32)
	result3 = zeros([N1,N1]).astype(float32)

	# split data to correct size to run on GPU
	Max = 256.0
	dist_gpu1 = mod.get_function("distance1")
	dist_gpu2 = mod.get_function("distance2")

	print dist_gpu1.num_regs

	numRuns = int(ceil(N1/Max))
	print "numRuns: ", numRuns

	for i in range(numRuns):
		for j in range(numRuns):

			s = N1/numRuns
			data1 = d1[(i*s):(i*s+s),:,:]
			data2 = d2[(j*s):(j*s+s),:,:]
			data3 = d3[(i*s):(i*s+s),:,:]
			data4 = d4[(j*s):(j*s+s),:,:]

			data5 = d5[(i*s):(i*s+s),:,:]
			data6 = d6[(j*s):(j*s+s),:,:]
			data7 = d7[(i*s):(i*s+s),:,:]
			data8 = d8[(j*s):(j*s+s),:,:]


			N=data1.shape[0]
			M=data1.shape[1]
			P=data1.shape[2]
			res1 = zeros([N,N]).astype(float32)
			res2 = zeros([N,N]).astype(float32)
			res3 = zeros([N,N]).astype(float32)

			# invoke kernel
			if(N<15):
				dist_gpu1(int32(N), int32(M), int32(P), float32(sigma), float32(pi), driver.In(data1), driver.In(data2), driver.In(data3), driver.In(data4), driver.Out(res1), block=(N,N,1), grid=(1,1))
				dist_gpu2(int32(N), int32(M), int32(P), float32(sigma), float32(pi), driver.In(data5), driver.In(data6), driver.In(data7), driver.In(data8), driver.Out(res2), driver.Out(res3), block=(N,N,1), grid=(1,1))
			else:
				g = ceil(N/15.0)
				dist_gpu1(int32(N), int32(M), int32(P), float32(sigma), float32(pi), driver.In(data1), driver.In(data2), driver.In(data3), driver.In(data4), driver.Out(res1), block=(15,15,1), grid=(int(g),int(g)))
				dist_gpu2(int32(N), int32(M), int32(P), float32(sigma), float32(pi), driver.In(data5), driver.In(data6), driver.In(data7), driver.In(data8), driver.Out(res2), driver.Out(res3), block=(15,15,1), grid=(int(g),int(g)))


			result1[(i*s):(i*s+s),(j*s):(j*s+s)] = res1
			result2[(i*s):(i*s+s),(j*s):(j*s+s)] = res2
			result3[(i*s):(i*s+s),(j*s):(j*s+s)] = res3


	sum1 = 0.0
	a1 = 0.0
	a2 = 0.0
	a3 = 0.0

	counter = 0
	for i in range(N1):
		if(isinf(log(sum(result1[i,:])/N2)) or isinf(log(sum(result2[i,:])/N3)) or isinf(log(sum(result3[i,:])/N4))): counter=counter+1
		else: sum1 = sum1 + log(sum(result1[i,:])/N2) - log(sum(result2[i,:])/N3) - log(sum(result3[i,:])/N4)

		a1 = a1 + log(sum(result1[i,:])/N2)
		a2 = a2 + log(sum(result2[i,:])/N3)
		a3 = a3 + log(sum(result3[i,:])/N4)



	print "a1: ", a1/float(i+1) , "a2: ", a2/float(i+1), "a3: ", a3/float(i+1)
	print "all: ",  a1/float(i+1) - a2/float(i+1) - a3/float(i+1)

	Info = sum1/float(N1)
	print "counter: ", counter
	return(Info)
