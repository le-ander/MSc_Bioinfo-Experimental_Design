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

def factor_partial(N):
	for R in range(int(sqrt(N)),1,-1):
		if N%R == 0:
			return float(R)

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

def optimise_gridsize(gmem_per_thread):
	# DETERMINE TOTAL NUMBER OF THREADS LIMITED BY GLOBAL MEMORY
	# Note: need to manually check that max grid dimensions are not exceeded
	# Read total global memory of device
	avail_mem = driver.mem_get_info()[0]
	# Calculate maximum number of threads
	max_threads = floor(avail_mem / gmem_per_thread)

	##could change it to be a multiple of block size?
	return max_threads
