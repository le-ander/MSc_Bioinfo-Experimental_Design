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

	print "Smallest optimal blocksize on this GPU:", optimal_blocksize
	print "Achieved theoretical GPU occupancy", (float(achieved_occupancy)/device.max_threads_per_multiprocessor)*100, "%"

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
	gridmax = 100.0 # Define square root of maximum threads per grid
	blockmax = 15.0 # Maximum threads per block

	# Determine required number of runs for i and j
	##need float here?
	numRuns_i = int(ceil(N1/float(gridmax)))
	numRuns_j = int(ceil(N2/float(gridmax)))

	res_t2 = zeros([N1,numRuns_j])

	# Prepare data
	d1 = data.astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	M = d1.shape[1] # number of timepoints
	P = d1.shape[2] # number of species

	Ni = int(gridmax)


	for i in range(numRuns_i):
		print "Runs left:", numRuns_i-i
		if((int(gridmax)*(i+1)) > N1): # If last run with less that max remaining trajectories
			Ni = int(N1 - gridmax*i) # Set Ni to remaining number of particels

		if(Ni<blockmax):
			gi = 1  # Grid size in dim i
			bi = Ni # Block size in dim i
		else:
			gi = ceil(Ni/blockmax)
			bi = blockmax

		data1 = d1[(i*int(gridmax)):(i*int(gridmax)+Ni),:,:] # d1 subunit for the next j runs

		Nj = int(gridmax)


		for j in range(numRuns_j):
			if((int(gridmax)*(j+1)) > N2): # If last run with less that max remaining trajectories
				Nj = int(N2 - gridmax*j) # Set Nj to remaining number of particels

			data2 = d2[(j*int(gridmax)):(j*int(gridmax)+Nj),:,:] # d2 subunit for this run

			##could move into if statements (only if ni or nj change)
			res1 = zeros([Ni,Nj]).astype(float64) # results vector [shape(data1)*shape(data2)]

			if(Nj<blockmax):
				gj = 1  # Grid size in dim j
				bj = Nj # Block size in dim j
			else:
				gj = ceil(Nj/blockmax)
				bj = blockmax

			# Invoke GPU calculations (takes data1 and data2 as input, outputs res1)
			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float64(scale), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# First summation (could be done on GPU?)
			for k in range(Ni):
					res_t2[(i*int(gridmax)+k),j] = sum(res1[k,:])

	sum1 = 0.0
	count_na = 0
	count_inf = 0

	for i in range(N1):
		if(isnan(sum(res_t2[i,:]))): count_na += 1
		elif(isinf(log(sum(res_t2[i,:])))): count_inf += 1
		else:
			sum1 += - log(sum(res_t2[i,:])) + log(float(N2)) + M*P*log(scale) +  M*P*log(2.0*pi*sigma*sigma)

	Info = (sum1 / float(N1 - count_na - count_inf)) - M*P/2.0*(log(2.0*pi*sigma*sigma)+1)

	return(Info)

# Used to be getEntropy3!
def getEntropy2(data,N1,N2,N3,sigma,theta,scale):
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

	// Can remove this function?
	__device__ unsigned int idx2d2(int k, int l, int P)
	{
		return k*P + l;

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
				x1 = x1 + log(scale) - ( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma);
			}
	}

	res1[idx2d(i,j,Nj)] = exp(x1);
	}

	__global__ void distance2(int Ni, int M, int P, float sigma, double scale, double *d1, double *d3, double *res2)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i>=Ni) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				x1 = x1 + log(scale) - ( d3[idx3d(i,k,l,M,P)]-d1[idx2d2(k,l,P)])*( d3[idx3d(i,k,l,M,P)]-d1[idx2d2(k,l,P)])/(2.0*sigma*sigma);
			}
	}

	res2[i] = exp(x1);
	}
	""")

	# Assigning main kernel function to a variable
	dist_gpu1 = mod.get_function("distance1")

	##should be defined as an int, can then clean up formulas further down
	gridmax = 100.0 # Define square root of maximum threads per grid
	blockmax = 15.0 # Maximum threads per block

	# Determine required number of runs for i and j
	##need float here?
	numRuns_i = int(ceil(N1/float(gridmax)))
	numRuns_j = int(ceil(N2/float(gridmax)))

	res_t2 = zeros([N1,numRuns_j])

	# Prepare data
	d1 = data.astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	M = d1.shape[1] # number of timepoints
	P = d1.shape[2] # number of species

	Ni = int(gridmax)


	for i in range(numRuns_i):
		print "Runs left:", numRuns_i - i
		if((int(gridmax)*(i+1)) > N1): # If last run with less that max remaining trajectories
			Ni = int(N1 - gridmax*i) # Set Ni to remaining number of particels

		if(Ni<blockmax):
			gi = 1  # Grid size in dim i
			bi = Ni # Block size in dim i
		else:
			gi = ceil(Ni/blockmax)
			bi = blockmax

		data1 = d1[(i*int(gridmax)):(i*int(gridmax)+Ni),:,:] # d1 subunit for the next j runs

		Nj = int(gridmax)


		for j in range(numRuns_j):
			if((int(gridmax)*(j+1)) > N2): # If last run with less that max remaining trajectories
				Nj = int(N2 - gridmax*j) # Set Nj to remaining number of particels

			data2 = d2[(j*int(gridmax)):(j*int(gridmax)+Nj),:,:] # d2 subunit for this run

			##could move into if statements (only if ni or nj change)
			res1 = zeros([Ni,Nj]).astype(float64) # results vector [shape(data1)*shape(data2)]

			if(Nj<blockmax):
				gj = 1  # Grid size in dim j
				bj = Nj # Block size in dim j
			else:
				gj = ceil(Nj/blockmax)
				bj = blockmax

			# Invoke GPU calculations (takes data1 and data2 as input, outputs res1)
			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float64(scale), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			# First summation (could be done on GPU?)
			for k in range(Ni):
				res_t2[(i*int(gridmax)+k),j] = sum(res1[k,:])

	sum1 = 0.0
	count_na = 0
	count_inf = 0

	for i in range(N1):
		if(isnan(sum(res_t2[i,:]))): count_na += 1
		elif(isinf(log(sum(res_t2[i,:])))): count_inf += 1
		else:
			sum1 += - log(sum(res_t2[i,:])) + log(float(N2)) + M*P*log(scale) +  M*P*log(2.0*pi*sigma*sigma)

######## part A finished with results saved in sum1

	dist_gpu2 = mod.get_function("distance2")

	##need this defined again here??
	gridmax = 256.0
	blockmax = 15.0

	numRuns_j2 = int(ceil(N3/gridmax))

	res_t2 = zeros([N1,numRuns_j2])

	d3 = array(theta)[(N1+N2):(N1+N2+N1*N3),:,:].astype(float64)


	for i in range(N1):

		data1 = d1[i,:,:]

		Nj = int(gridmax)

		for j in range(numRuns_j2):
			print "runs left:", numRuns_j2 - j

			if((int(gridmax)*(j+1)) > N3):
				Nj = int(N3 - gridmax*j)

			data3 = d3[(i*N3+j*int(gridmax)):(i*N3+j*int(gridmax)+Nj),:,:]

			res2 = zeros([Nj]).astype(float64)

			if(Nj<blockmax):
				gj = 1
				bj = Nj
			else:
				gj = ceil(Nj/blockmax)
				bj = blockmax

			dist_gpu2(int32(Nj), int32(M), int32(P), float32(sigma), float64(scale), driver.In(data1), driver.In(data3),  driver.Out(res2), block=(1,int(bj),1), grid=(1,int(gj)))

			res_t2[i,j] = sum(res2[:])


	sum2 = 0.0
	count2_na = 0
	count2_inf = 0

	for i in range(N1):
		if(isnan(sum(res_t2[i,:]))): count2_na += 1
		elif(isinf(log(sum(res_t2[i,:])))): count2_inf += 1
		else:
			sum2 += log(sum(res_t2[i,:])) - log(float(N3)) - M*P*log(scale) -  M*P*log(2.0*pi*sigma*sigma)

	######## part B finished with results saved in sum2

	Info = (sum2 - sum1)/float(N1 - count_na - count_inf - count2_na - count2_inf)

	return(Info)
