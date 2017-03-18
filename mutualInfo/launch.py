from numpy import *
from pycuda import driver


# A function to find the next lowest number from num that is a multiple of divisor
def round_down(num, divisor):
	return num - (num%divisor)


# A function to find the next highest number from num that is a multiple of divisor
def round_up(num, divisor):
	if num == divisor:
		return 1
	else:
		return num - (num%divisor) + divisor


# A function to find the divisor R of number N that closest to and smaller than sqrt(N)
def factor_partial(N):
	for R in range(int(sqrt(N)),1,-1):
		if N%R == 0:
			return float(R)


# A funtion to determine total number of threads limited by global memory
##Attention: user needs to manually check that max grid dimensions are not exceeded
##Argument: gmem_per_thread - number of bytes of global memory that is occupied by a single thread
def optimise_gridsize(gmem_per_thread):
	# Read total global memory of device
	avail_mem = driver.mem_get_info()[0]
	# Calculate maximum number of threads
	max_threads = floor(avail_mem / gmem_per_thread)

	return max_threads


# A function to calculate the minumum blocksize that achieves maximum occupancy of the GPU and hence minimises runtime
##Arguments:
##device - the pycuda.driver.Context instance for the device the kernel will be executing (e.g. pycuda.autoinit.device)
##function - a handle of the cuda function (kernel) that will be executed
def optimal_blocksize(device, function):

	# Check that compute capability of the GPU is compatible with this launch configurator
	print "Detected compute capability of device:", str(device.compute_capability()[0])+"."+str(device.compute_capability()[1])
	if device.compute_capability()[0] < 2 or device.compute_capability()[0] > 6:
		print "WARNING: GPU not supported. The launch configurator only supports compute capability 2.0 - 6.2. Configuration for CC 6.2 will be used. This might cause errors."
	elif device.compute_capability()[0] == 6 and device.compute_capability()[1] > 2:
		print "WARNING: GPU not supported. The launch configurator only supports compute capability 2.0 - 6.2. Configuration for CC 6.2 will be used. This might cause errors."

	# Start at smallest blocksize and iteratively increase it to find maximum occupancy
	max_blocksize = min(device.max_threads_per_block, function.max_threads_per_block)
	achieved_occupancy = 0
	blocksize = device.warp_size
	while blocksize <= max_blocksize:
		occupancy = blocksize * max_active_blocks_per_sm(device, function, blocksize)
		if occupancy > achieved_occupancy:
			optimal_blocksize = blocksize
			achieved_occupancy = occupancy
		# If maximum possible blocksize already achieved, exit
		if achieved_occupancy == device.max_threads_per_multiprocessor:
			break
		blocksize += device.warp_size

	print "Theoretically achieved GPU occupancy:", (float(achieved_occupancy)/device.max_threads_per_multiprocessor)*100, "%"

	return float(optimal_blocksize)


# A function to determine the maximum number of active blocks per multiprocessor,
##limited by register count, blocks/SM or threads/SM and shared memory (not used directly, but gets called by optimal_blocksize())
##Arguments:
##device - the pycuda.driver.Context instance for the device the kernel will be executing (e.g. pycuda.autoinit.device)
##function - a handle of the cuda function (kernel) that will be executed
##blocksize - number of threads per block for which the maximum numbers of blocks per SM should be calculated
##dyn_smem_per_block - number of bytes of dynamically allocated shared memory for a block of size blocksize
def max_active_blocks_per_sm(device, function, blocksize, dyn_smem_per_block=0):

	# Define variables based on device and function properties
	regs_per_thread = function.num_regs
	smem_per_function = function.shared_size_bytes
	warp_size = device.warp_size
	max_threads_per_block = min(device.max_threads_per_block, function.max_threads_per_block)
	max_threads_per_sm = device.max_threads_per_multiprocessor
	max_regs_per_block = device.max_registers_per_block
	max_smem_per_block = device.max_shared_memory_per_block

	# Define variables that cannot be read directly from the device based on compute capability
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

	#print "Max numbers, limited by [Registers, Threads per SM, Blocks per SM, Shared Memory]", [block_lim_regs, block_lim_tSM, block_lim_bSM, block_lim_smem]
	#print "Maximum number of blocks on this device (with", blocksize, "threads each):", block_lim

	return block_lim