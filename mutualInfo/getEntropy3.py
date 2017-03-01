from numpy import *
from numpy.random import *
import math
import re

from pycuda import compiler, driver
from pycuda import autoinit

import launch

import time
import sys


def getEntropy3(dataRef,thetaRef,dataMod,thetaMod,N1,N2,N3,N4,sigma_ref,sigma_mod,scale):

	mod = compiler.SourceModule("""

	__device__ unsigned int idx3d(int i, int k, int l, int M, int P)
	{
		return k*P + i*M*P + l;

	}

	__device__ unsigned int idx2d(int i, int j, int M)
	{
		return i*M + j;

	}

	__global__ void distance1(int Ni, int Nj, int M_Ref, int P_Ref, int M_Mod, int P_Mod, float sigma_ref, float sigma_mod, double scale, double *d1, double *d2, double *d3, double *d4, double *res1)
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
				   x1 = x1 + scale - ( d2[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])*( d2[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])/(2.0*sigma_ref*sigma_ref);
			}
	}


	for(int k=0; k<M_Mod; k++){
			for(int l=0; l<P_Mod; l++){
				   x3 = x3 + scale - ( d4[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])*( d4[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])/(2.0*sigma_mod*sigma_mod);
			}
	}


	res1[idx2d(i,j,Nj)] = exp(x1+x3);
	}


	__global__ void distance2(int Ni, int Nj, int M_Ref, int P_Ref, int M_Mod, int P_Mod, float sigma_ref, float sigma_mod, double scale, double *d1, double *d2, double *d3, double *d4, double *res2, double *res3)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x2;
	double x3;
	x2 = 0.0;
	x3 = 0.0;

	for(int k=0; k<M_Ref; k++){
			for(int l=0; l<P_Ref; l++){
				   x2 = x2 + scale - ( d2[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])*( d2[idx3d(j,k,l,M_Ref,P_Ref)]-d1[idx3d(i,k,l,M_Ref,P_Ref)])/(2.0*sigma_ref*sigma_ref);
			}
	}
	for(int k=0; k<M_Mod; k++){
			for(int l=0; l<P_Mod; l++){
				   x3 = x3 + scale - ( d4[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])*( d4[idx3d(j,k,l,M_Mod,P_Mod)]-d3[idx3d(i,k,l,M_Mod,P_Mod)])/(2.0*sigma_mod*sigma_mod);
			}
	}

	res2[idx2d(i,j,Nj)] = exp(x2);
	res3[idx2d(i,j,Nj)] = exp(x3);


	}


	""")

	dist_gpu1 = mod.get_function("distance1")
	dist_gpu2 = mod.get_function("distance2")

	block = launch.optimal_blocksize(autoinit.device, dist_gpu1)
	block_i = launch.factor_partial(block)
	block_j = block / block_i
	print "block, BLOCK_I and _j",block, block_i, block_j

	###Need to detect exact memory use here!
	grid = launch.optimise_gridsize(15.0)
	grid_prelim_i = launch.round_down(sqrt(grid),block_i)
	grid_prelim_j = launch.round_down(grid/grid_prelim_i,block_j)
	print "PRELIM: grid, GRID_i and _j", grid, grid_prelim_i, grid_prelim_j

	if N1 < grid_prelim_i:
		grid_i = float(min(autoinit.device.max_grid_dim_x,N1))
		grid_j = float(min(autoinit.device.max_grid_dim_y, launch.round_down(grid/grid_i,block_j)))
	elif N2 < grid_prelim_j:
		grid_j = float(min(autoinit.device.max_grid_dim_y,N2))
		grid_i = float(min(autoinit.device.max_grid_dim_x, launch.round_down(grid/grid_j,block_i)))
	else:
		grid_i = float(min(autoinit.device.max_grid_dim_x, grid_prelim_i))
		grid_j = float(min(autoinit.device.max_grid_dim_y, grid_prelim_j))
	print "grid, GRID_i and _j", grid, grid_i, grid_j

	numRuns_i = int(ceil(N1/grid_i))
	numRuns_j = int(ceil(N2/grid_j))
	print "numRuns_i: ", numRuns_i
	print "numRuns_j: ", numRuns_j

	d1 = dataRef[0:N1,:,:].astype(float64)
	d2 = array(thetaRef)[N1:(N1+N2),:,:].astype(float64)
	d3 = dataMod[0:N1,:,:].astype(float64)
	d4 = array(thetaMod)[N1:(N1+N2),:,:].astype(float64)
	d6 = array(thetaRef)[(N1+N2):(N1+N2+N3),:,:].astype(float64)
	d8 = array(thetaMod)[(N1+N2+N3):(N1+N2+N3+N4),:,:].astype(float64)

	M_Ref=d1.shape[1]
	P_Ref=d1.shape[2]

	M_Mod=d3.shape[1]
	P_Mod=d3.shape[2]

	result1 = zeros([N1,numRuns_j])
	result2 = zeros([N1,numRuns_j])
	result3 = zeros([N1,numRuns_j])

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

			dist_gpu1(int32(Ni), int32(Nj), int32(M_Ref), int32(P_Ref), int32(M_Mod), int32(P_Mod), float32(sigma_ref), float32(sigma_mod), float64(scale), driver.In(data1), driver.In(data2), driver.In(data3), driver.In(data4), driver.Out(res1),block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))
			dist_gpu2(int32(Ni), int32(Nj), int32(M_Ref), int32(P_Ref), int32(M_Mod), int32(P_Mod), float32(sigma_ref), float32(sigma_mod), float64(scale), driver.In(data1), driver.In(data6), driver.In(data3), driver.In(data8), driver.Out(res2), driver.Out(res3),block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

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
			sum1 += log(sum(result1[i,:])) - log(sum(result2[i,:])) - log(sum(result3[i,:])) - log(float(N2)) + log(float(N3)) + log(float(N4))

			#a1 += log(sum(result1[i,:])) - log(N2) - M_Ref*P_Ref*log(2.0*pi*sigma_ref*sigma_ref) - M_Mod*P_Mod*log(2.0*pi*sigma_mod*sigma_mod) - M_Ref*P_Ref*scale - M_Mod*P_Mod*scale
			#a2 -= log(sum(result2[i,:])) + log(N3) + M_Ref*P_Ref*log(2.0*pi*sigma_ref*sigma_ref) + M_Ref*P_Ref*scale
			#a3 -= log(sum(result3[i,:])) + log(N4) + M_Mod*P_Mod*log(2.0*pi*sigma_mod*sigma_mod) + M_Mod*P_Mod*scale

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


def run_getEntropy3(model_obj, ref_obj):
	MutInfo3 = []
	for experiment in range(model_obj.nmodels):

		#pos = model_obj.pairParamsICS.values()[cudaorder.index(cudafile)].index([x[1] for x in model_obj.x0prior[model]])
		if model_obj.initialprior == False:
			pos = model_obj.pairParamsICS[model_obj.cuda[experiment]].index([x[1] for x in model_obj.x0prior[experiment]])
			N1 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][0]
			N2 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][1]
			N3 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][2]
			N4 = model_obj.cudaout_structure[model_obj.cuda[experiment]][pos][3]
		else:
			pos = model_obj.cudaout_structure[model_obj.cuda[experiment]][0]
			N1 = pos[0]
			N2 = pos[1]
			N3 = pos[2]
			N4 = pos[3]

		print "-----Calculating Mutual Information-----", experiment
		#print model_obj.trajectories[experiment].shape
		#print model_obj.cudaout[experiment].shape
		#print N1, N2

		MutInfo3.append(getEntropy3(ref_obj.trajectories[experiment],model_obj.trajectories[experiment],ref_obj.cudaout[experiment], model_obj.cudaout[experiment],N1,N2,N3,N4,ref_obj.sigma,model_obj.sigma,model_obj.scale[experiment]))

		print "Mutual Information:", MutInfo3[experiment]

	return MutInfo3
