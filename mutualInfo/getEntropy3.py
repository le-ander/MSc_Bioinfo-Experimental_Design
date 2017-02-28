from numpy import *
from numpy.random import *
import math
import re

from pycuda import compiler, driver
from pycuda import autoinit

import time
import sys


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

	return float(optimal_blocksize)

def optimise_grid_structure(gmem_per_thread):
	# DETERMINE TOTAL NUMBER OF THREADS LIMITED BY GLOBAL MEMORY
	# Note: need to manually check that max grid dimensions are not exceeded
	# Read total global memory of device
	avail_mem = driver.mem_get_info()[0]
	# Calculate maximum number of threads
	max_threads = floor(avail_mem / gmem_per_thread)

	##could change it to be a multiple of block size?
	return max_threads

def factor_partial(N):
	for R in range(int(sqrt(N)),1,-1):
		if N%R == 0:
			return float(R)

'''
###Does not work! See comments in the function
def scaling_gE3(modelTraj, ftheta, sigma):
	###maxDistTraj is the same for all combintions of ref and alt experiments, however, the calc of "a" from maxDistTraj is different for each combination.
	###Not sure how to implement this outside of gE3...
	maxDistTraj = []

	for tp in range(shape(modelTraj)[1]):
		maxDistTraj.append(max(amax(ftheta[:,tp,:]),amax(modelTraj[:,tp,:])) - min(amin(ftheta[:,tp,:]),amin(modelTraj[:,tp,:])))

	M_Ref=ftheta.shape[1]
	P_Ref=ftheta.shape[2]

	###How to select the right alternative model in this run??
	M_Alt=XX.shape[1]
	P_Alt=XX.shape[2]

	M_Max = float(max(M_Ref,M_Alt))
	P_Max = float(max(P_Ref,P_Alt))

	preci = pow(10,-34)

	aa1 = log(preci)/(2.0*M_Max*P_Max) + (maxDistTraj*maxDistTraj)/(2.0*sigma*sigma)
	aa2 = math.log(pow(10,300))/(2.0*M_Max*P_Max)

	print "aa1: ",aa1, "aa2: ",aa2
	if(aa1<aa2): a = aa1
	else: a = 0.0

	#FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))
	#print "FmaxDistTraj:",FmaxDistTraj
	#if(FmaxDistTraj<preci):
	#	a = pow(1.79*pow(10,300),1.0/(2.0*d1.shape[1]*d1.shape[2]*d3.shape[1]*d3.shape[2]))
	#else:
	#	a = pow(1.79*pow(10,300),1.0/(2.0*d1.shape[1]*d1.shape[2]*d3.shape[1]*d3.shape[2]))

	print "preci:", preci,"a:",a

	return a
'''


def getEntropy3(dataRef,dataMod,N1,N2,N3,N4,sigma,thetaRef,thetaMod,maxDistTraj):

	#kernel declaration
	mod = compiler.SourceModule("""
	#include <stdio.h>

	__device__ unsigned int idx3d(int i, int k, int l, int M, int P)
	{
		return k*P + i*M*P + l;

	}

	__device__ unsigned int idx2d(int i, int j, int M)
	{
		return i*M + j;

	}

	__global__ void distance1(int Ni, int Nj, int M_Ref, int M_Ref, int M_Alt, int P_Alt, float sigma, double a, double *d1, double *d2, double *d3, double *d4, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x1;
	x1 = 0.0;
	double x3;
	x3 = 0.0;
	for(int k=0; k<M_Ref; k++){
			for(int l=0; l<M_Ref; l++){
				   x1 = x1 + a - ( d2[idx3d(j,k,l,M_Ref,M_Ref)]-d1[idx3d(i,k,l,M_Ref,M_Ref)])*( d2[idx3d(j,k,l,M_Ref,M_Ref)]-d1[idx3d(i,k,l,M_Ref,M_Ref)])/(2.0*sigma*sigma);
			}
	}


	for(int k=0; k<M_Alt; k++){
			for(int l=0; l<P_Alt; l++){
				   x3 = x3 + a - ( d4[idx3d(j,k,l,M_Alt,P_Alt)]-d3[idx3d(i,k,l,M_Alt,P_Alt)])*( d4[idx3d(j,k,l,M_Alt,P_Alt)]-d3[idx3d(i,k,l,M_Alt,P_Alt)])/(2.0*sigma*sigma);
			}
	}


	res1[idx2d(i,j,Nj)] = exp(x1+x3);
	}


	__global__ void distance2(int Ni, int Nj, int M_Ref, int M_Ref, int M_Alt, int P_Alt, float sigma, double a, double *d1, double *d2, double *d3, double *d4, double *res2, double *res3)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x2;
	double x3;
	x2 = 0.0;
	x3 = 0.0;

	for(int k=0; k<M_Ref; k++){
			for(int l=0; l<M_Ref; l++){
				   x2 = x2 + a - ( d2[idx3d(j,k,l,M_Ref,M_Ref)]-d1[idx3d(i,k,l,M_Ref,M_Ref)])*( d2[idx3d(j,k,l,M_Ref,M_Ref)]-d1[idx3d(i,k,l,M_Ref,M_Ref)])/(2.0*sigma*sigma);
			}
	}
	for(int k=0; k<M_Alt; k++){
			for(int l=0; l<P_Alt; l++){
				   x3 = x3 + a - ( d4[idx3d(j,k,l,M_Alt,P_Alt)]-d3[idx3d(i,k,l,M_Alt,P_Alt)])*( d4[idx3d(j,k,l,M_Alt,P_Alt)]-d3[idx3d(i,k,l,M_Alt,P_Alt)])/(2.0*sigma*sigma);
			}
	}

	res2[idx2d(i,j,Nj)] = exp(x2);
	res3[idx2d(i,j,Nj)] = exp(x3);


	}


	""")

	dist_gpu1 = mod.get_function("distance1")
	dist_gpu2 = mod.get_function("distance2")

	block = optimal_blocksize(autoinit.device, dist_gpu1)
	block_i = factor_partial(block)
	block_j = block / block_i
	print "block, BLOCK_I and _j",block, block_i, block_j

	###Need to detect exact memory use here!
	grid = optimise_grid_structure(15.0)
	grid_prelim_i = round_down(sqrt(grid),block_i)
	grid_prelim_j = round_down(grid/grid_prelim_i,block_j)
	print "PRELIM: grid, GRID_i and _j", grid, grid_prelim_i, grid_prelim_j

	if N1 < grid_prelim_i:
		grid_i = float(min(autoinit.device.max_grid_dim_x,N1))
		grid_j = float(min(autoinit.device.max_grid_dim_y, round_down(grid/grid_i,block_j)))
	elif N2 < grid_prelim_j:
		grid_j = float(min(autoinit.device.max_grid_dim_y,N2))
		grid_i = float(min(autoinit.device.max_grid_dim_x, round_down(grid/grid_j,block_i)))
	else:
		grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i))
		grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j))
	print "grid, GRID_i and _j", grid, grid_i, grid_j

	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N2/grid_j))
	print "numRuns_i: ", numRuns_i
	print "numRuns_j: ", numRuns_j

	###Why is thetaMod[(N1+N2):(N1+N2+N3)] not used??
	d1 = dataRef[0:N1,:,:].astype(float64)
	d2 = array(thetaRef)[N1:(N1+N2),:,:].astype(float64)
	d3 = dataMod[0:N1,:,:].astype(float64)
	d4 = array(thetaMod)[N1:(N1+N2),:,:].astype(float64)
	d6 = array(thetaRef)[(N1+N2):(N1+N2+N3),:,:].astype(float64)
	d8 = array(thetaMod)[(N1+N2+N3):(N1+N2+N3+N4),:,:].astype(float64)

	M_Ref=d1.shape[1]
	P_Ref=d1.shape[2]

	M_Alt=d3.shape[1]
	P_Alt=d3.shape[2]

	result1 = zeros([N1,numRuns_j])
	result2 = zeros([N1,numRuns_j])
	result3 = zeros([N1,numRuns_j])


	###Move scaling into seperate function
	####################SCALING#################################################
	preci = pow(10,-34)

	aa1 = log(preci)/(2.0*M_Max*P_Max) + (maxDistTraj*maxDistTraj)/(2.0*sigma*sigma)
	aa2 = math.log(pow(10,300))/(2.0*M_Max*P_Max)

	print "aa1: ",aa1, "aa2: ",aa2
	if(aa1<aa2): a = aa1
	else: a = 0.0

	#FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))
	#print "FmaxDistTraj:",FmaxDistTraj
	#if(FmaxDistTraj<preci):
	#	a = pow(1.79*pow(10,300),1.0/(2.0*d1.shape[1]*d1.shape[2]*d3.shape[1]*d3.shape[2]))
	#else:
	#	a = pow(1.79*pow(10,300),1.0/(2.0*d1.shape[1]*d1.shape[2]*d3.shape[1]*d3.shape[2]))

	print "preci:", preci,"a:",a
	################END SCALING#################################################

	Ni = int(grid_i)

	for i in range(numRuns_i):
		print "runs left:", numRuns_i - i

		Ni = int(grid_i)
		Nj = int(grid_i)

		if((int(grid_i)*(i+1)) > N1):
			Ni = int(N1 - grid_i*i)

		data1 = d1[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:]
		data3 = d3[(i*int(grid_i)):(i*int(grid_i)+Ni),:,:]

		Nj = int(grid_j)

		for j in range(numRuns_j):
			if((int(grid_j)*(j+1)) > N2):
				Nj = int(N2 - grid_j*j)

			data2 = d2[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]
			data4 = d4[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]
			data6 = d6[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]
			data8 = d8[(j*int(grid_j)):(j*int(grid_j)+Nj),:,:]

			res1 = zeros([Ni,Nj]).astype(float64)
			res2 = zeros([Ni,Nj]).astype(float64)
			res3 = zeros([Ni,Nj]).astype(float64)

			if(Ni<block_i):
				gi = 1
				bi = Ni
			else:
				bi = block_i
				gi = ceil(Ni/block_i)
			if(Nj<block_j):
				gj = 1
				bj = Nj
			else:
				bj = block_j
				gj = ceil(Nj/block_j)

			dist_gpu1(int32(Ni), int32(Nj), int32(M_Ref), int32(M_Ref), int32(M_Alt), int32(P_Alt), float32(sigma), float64(a), driver.In(data1), driver.In(data2), driver.In(data3), driver.In(data4), driver.Out(res1),block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			dist_gpu2(int32(Ni), int32(Nj), int32(M_Ref), int32(M_Ref), int32(M_Alt), int32(P_Alt), float32(sigma), float64(a), driver.In(data1), driver.In(data6), driver.In(data3), driver.In(data8), driver.Out(res2), driver.Out(res3),block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			for k in range(Ni):
				result1[(i*int(grid_i)+k),j] = sum(res1[k,:])
				result2[(i*int(grid_i)+k),j] = sum(res2[k,:])
				result3[(i*int(grid_i)+k),j] = sum(res3[k,:])

	sum1 = 0.0
	a1 = 0.0
	a2 = 0.0
	a3 = 0.0

	count1_na = 0
	count1_inf = 0
	count2_na = 0
	count2_inf = 0
	count3_na = 0
	count3_inf = 0

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
			sum1 = sum1 + log(sum(result1[i,:])) - log(sum(result2[i,:])) - log(sum(result3[i,:])) - log(float(N2)) + log(float(N3)) + log(float(N4))

			#a1 = a1 + log(sum(result1[i,:])) - log(N2) - 2.0*M_Ref*M_Ref*log(2.0*pi*sigma*sigma) - 2*M_Ref*M_Ref*a
			#a2 = a2 - log(sum(result2[i,:])) + log(N3) +  M_Ref*M_Ref*log(2.0*pi*sigma*sigma) + M_Ref*M_Ref*a
			#a3 = a3 - log(sum(result3[i,:])) + log(N4) +  M_Ref*M_Ref*log(2.0*pi*sigma*sigma) + M_Ref*M_Ref*a

	count_all = count1_inf + count1_na + count2_inf + count2_na + count3_inf + count3_na

	#print "a1: ", a1/float(N1-count1_na-count1_inf) , "a2: ", a2/float(N1-count2_na-count2_inf), "a3: ", a3/float(N1-count3_na-count3_inf)
	#print "all: ", a1/float(N1-count1_na-count1_inf) + a2/float(N1-count2_na-count2_inf) + a3/float(N1-count3_na-count3_inf)
	#print "sum1: ", sum1

	Info = sum1/float(N1-count_all)

	print "count1_na: ", count1_na
	print "count1_inf: ", count1_inf
	print "count2_na: ", count2_na
	print "count2_inf: ", count2_inf
	print "count3_na: ", count3_na
	print "count3_inf: ", count3_inf

	return(Info)
