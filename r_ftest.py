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
t3 = time.time()


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
	print "FmaxDistTraj:",FmaxDistTraj

	if(FmaxDistTraj<preci):
		scale = pow(1.79*pow(10,300),1.0/(ftheta[0].shape[1]*ftheta[0].shape[2]))
	else:
		scale = pow(preci,1.0/(ftheta[0].shape[1]*ftheta[0].shape[2]))*1.0/FmaxDistTraj

	return scale #*pow(10,-2)

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
	gridmax = 23000.0 # Define square root of maximum threads per grid
	blockmax = 16.0 # Maximum threads per block

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
			print "RES1", res1
			# First summation (could be done on GPU?)
			for k in range(Ni):
					res_t2[(i*int(gridmax)+k),j] = sum(res1[k,:])
			print res_t2
	sum1 = 0.0
	count_na = 0
	count_inf = 0

	for i in range(N1):
		if(isnan(sum(res_t2[i,:]))): count_na += 1
		elif(isinf(log(sum(res_t2[i,:])))): count_inf += 1
		else:
			sum1 += - log(sum(res_t2[i,:])) + log(float(N2)) + M*P*log(scale) +  M*P*log(2.0*pi*sigma*sigma)
	print count_na, count_inf
	Info = (sum1 / float(N1 - count_na - count_inf)) - M*P/2.0*(log(2.0*pi*sigma*sigma)+1)

	optimal_blocksize(autoinit.device, dist_gpu1)

	return(Info)




def getWeightedSample(weights):

	totals = []
	running_total = 0

	for w in weights:
		running_total = running_total + w[0]
		totals.append(running_total)

	rnd = random() * running_total
	for i, total in enumerate(totals):
		if rnd < total:
			return i

def printOptions():

	print "\nList of possible options:"

	print "\n Input options:"
	print "-i\t--infile\t declaration of the input file. This input file has to be provided to run the program!"
	print "-lc\t--localcode\t do not import model from sbml intead use a local .py, .hpp/.cpp or .cu file"

	print "\n Algorithmic options:"
	print "-sd\t--setseed\t seed the random number generator in numpy with an integer eg -sd=2, --setseed=2"
	print "-tm\t--timing\t print timing information"
	print "--c++\t\t\t use C++ implementation"
	print "-cu\t--cuda\t\t use CUDA implementation"

	print "\n Output options:"
	print "-of\t--outfolder\t write results to folder eg -of=/full/path/to/folder (default is _results_ in current directory)"
	print "-f\t--fulloutput\t print epsilon, sampling steps and acceptence rates after each population"
	print "-s\t--save\t\t no backup after each population"
	print "-db\t--debug\t set the debug mode"

	print "\n Simulate options:"
	print "-S\t--simulate\t simulate the model over the range of timepoints, using paramters sampled from the priors"

	print "\n Design options:"
	print "-D\t--design\t run ABC-SysBio in design mode"

	print "\n Plotting options:"
	print "-d\t--diagnostic\t no printing of diagnostic plots"
	print "-t\t--timeseries\t no plotting of simulation results after each population"
	print "-p\t--plotdata\t no plotting of given data points"
	print "\n-h\t--help\t\t print this list of options."

	print "\n"

def main():

	diagnostic=True
	pickling=True
	file_exist=False
	plot=True
	plotTimeSeries=True
	simulate=False
	design=False
	full=False
	usesbml=True
	seed = None
	timing = False
	fname = "_results_"
	custom_kernel = False
	custom_distance = False
	use_cuda = False
	use_c = False
	full_debug = False

	for i in range(1,len(sys.argv)):

		if sys.argv[i].startswith('--'):
			option = sys.argv[i][2:]

			if option == 'help':
				printOptions()
				sys.exit()
			elif option == 'diagnostic': diagnostic=False
			elif option == 'save': pickling=False
			elif option == 'timeseries': plotTimeSeries=False
			elif option == 'plotdata': plot=False
			elif option == 'simulate': simulate=True
			elif option == 'design': design=True
			elif option == 'debug': full_debug=True
			elif option == 'fulloutput': full=True
			elif option == 'localcode' : usesbml = False
			elif option[0:8] == 'setseed=' : seed = int( option[8:] )
			elif option[0:10] == 'outfolder=' : fname = option[10:]
			elif option[0:9] == 'cudacode=' : app_file = option[9:]
			elif option == 'timing' : timing = True
			elif option == 'custk' : custom_kernel = True
			elif option == 'custd' : custom_distance = True
			elif option == 'cuda' : use_cuda = True
			elif option == 'c++' : use_c = True
			elif option == 'infile':
				input_file=sys.argv[i+1]
				file_exist=True
			elif not(sys.argv[i-1][2:] == 'infile'):
				print "\nunknown option "+sys.argv[i]
				printOptions()
				sys.exit()


		elif sys.argv[i].startswith('-'):
			option = sys.argv[i][1:]
			if option == 'h':
				printOptions()
				sys.exit()
			elif option == 'd': diagnostic=False
			elif option == 's': pickling=False
			elif option == 't': plotTimeSeries=False
			elif option == 'p': plot=False
			elif option == 'S': simulate=True
			elif option == 'D': design=True
			elif option == 'db': full_debug=True
			elif option == 'f': full=True
			elif option == 'cu': use_cuda = True
			elif option == 'lc' : usesbml = False
			elif option[0:3] == 'sd=' : seed = int( option[3:] )
			elif option[0:3] == 'of=' : fname = option[3:]
			elif option == 'tm' : timing = True
			elif option == 'i':
				input_file=sys.argv[i+1]
				file_exist=True
			elif not(sys.argv[i-1][2:] == 'i'):
				print "\nunknown option "+sys.argv[i]
				printOptions()
				sys.exit()
		elif not((sys.argv[i-1][2:] == 'infile') or (sys.argv[i-1][1:] == 'i')):
			print "\nunknown expression \""+sys.argv[i]+"\""
			printOptions()
			sys.exit()

	if file_exist == False:
		print "\nNo input_file is given!\nUse: \n\t-i 'inputfile' \nor: \n\t--infile 'inputfile' \n"
		sys.exit()

	# python, C++ or CUDA
	if use_cuda == True and use_c == True:
		print "specified both c++ and CUDA "
		sys.exit()
	if design == True and simulate==True:
		print "specified both design and simulate "
		sys.exit()

	# parse the input file
	mode = 0
	if simulate == True: mode = 1
	if design == True: mode = 2

	info_new = parse_infoEnt.algorithm_info(input_file, mode)

	info_new.print_info()

	# Check that we have libSBML if it is requested
	if usesbml == True:
		try: import libsbml
		except ImportError:
			print "ABORT: libSBML required for SBML parsing. Please install libSBML"
			sys.exit()

	# Check that we can import scipy if we have ODE models
	o = re.compile('ODE')
	for m in range( info_new.nmodels ):
		if o.search(info_new.type[m]):
			try: from scipy.integrate.odepack import odeint
			except ImportError:
				print "ABORT: scipy required for ODE modelling. Please install scipy"
				sys.exit()
			break

	# Check that we have cuda-sim installed
	if use_cuda == True:
		try: import cudasim
		except ImportError:
			print "ABORT: cudasim required for running on CUDA GPUs. Please install cuda-sim"
			sys.exit()

	# set the random seeds
	if seed != None:
		print "#### Seeding random number generator : ", seed
		numpy.random.seed(seed)

	# Check the information is correct for simulation
	modelCorrect = False
	if usesbml == True :
		integrationType = []

		if use_cuda == True:
			# CUDA
			for i in range(len(info_new.type)):
				integrationType.append(info_new.type[i]+' CUDA')
		elif use_c == True :
			# C
			for i in range(len(info_new.type)):
				if info_new.type[i] == "SDE":
					info_new.type[i] = "EulerSDE"
				integrationType.append(info_new.type[i]+' C')
		else:
			# python
			for i in range(len(info_new.type)):
				integrationType.append(info_new.type[i]+' Python')


		ParseAndWrite.ParseAndWrite(info_new.source,integrationType,info_new.name,inputPath="",outputPath="",method=None)


	print("Parsing done")
	print("Starting Simulation...")
	modelTraj = []


	sampleFromPost = False
	referenceModel = True

	#loop over models

	#try:
	#    os.mkdir("acceptedParticles")
	#except:
		#print "\nThe folder acceptedParticles already exists!\n"
	#    sys.exit()

	#try:
	#    os.mkdir("rejectedParticles")
	#except:
		#print "\nThe folder rejectedParticles already exists!\n"
	#    sys.exit()


	saveIndex = []
	saveResult = []

	for mod in range(info_new.nmodels):

		print "Model:",mod+1


		accepted = 0
		while(accepted<info_new.particles):
			if(mod==0): parameters = zeros( [info_new.particles,len(info_new.prior[mod])] )
			species = zeros([info_new.particles,info_new.nspecies[mod]])

			for i in range(info_new.particles):
				for j in range(info_new.nspecies[mod]):
					species[i,j] = info_new.x0prior[mod][j][1]

			if(mod==0 and sampleFromPost==False):
				for j in range(len(info_new.prior[mod])):
					if(info_new.prior[mod][j][0]==0):
						for i in range(info_new.particles):
							parameters[i,j] = info_new.prior[mod][j][1]
					if(info_new.prior[mod][j][0]==2):
						parameters[:,j] = uniform(low=info_new.prior[mod][j][1], high=info_new.prior[mod][j][2],size=(1,1,info_new.particles))[0][0]
					if(info_new.prior[mod][j][0]==1):
						parameters[:,j] = normal(loc=info_new.prior[mod][j][1], scale=info_new.prior[mod][j][2],size=(1,1,info_new.particles))[0][0]
			if(mod==0 and sampleFromPost==True):
				infileName = "data"+`mod+1`+".txt"
				in_file=open(infileName, "r")
				matrix=[]
				param=[]
				counter=0
				for in_line in in_file.readlines():
					in_line=in_line.rstrip()
					matrix.append([])
					param.append([])
					matrix[counter]=in_line.split(" ")
					param[counter] = map(float, matrix[counter])
					counter=counter+1
				in_file.close

				infileName = "data_Weights"+`mod+1`+".txt"
				in_file=open(infileName, "r")
				matrix=[]
				weights=[]
				counter2=0
				for in_line in in_file.readlines():
					in_line=in_line.rstrip()
					matrix.append([])
					weights.append([])
					matrix[counter2]=in_line.split(" ")
					weights[counter2] = map(float, matrix[counter2])
					counter2=counter2+1
				in_file.close

				if not(counter == counter2):
					print ""
					print "Please provide equal number of particles and weights in model ", mod+1, "!"
					sys.exit()
				else:
					parameters = zeros( [info_new.particles,len(param[0])] )

					index = getWeightedSample(weights)

					for i in range(info_new.particles):
						index = getWeightedSample(weights)
						for j in range(len(param[0])):
							parameters[i,j] = param[index][j]

			accepted = 10000000

	N1 = 100
	N2 = 4500

	modelTraj = run_cudasim(info_new, parameters, species)
	#print "model traj SHAPE", shape(modelTraj[0])

	modelTraj = remove_na(info_new, modelTraj)

	ftheta = add_noise_to_traj(info_new, modelTraj, 5.0, N1)

	scale = scaling(modelTraj, ftheta, 5.0)

	t1=time.time()
	MutInfo1 = get_mutinf_all_param(info_new, ftheta, N1, N2, 5.0, modelTraj, scale)
	t2=time.time()
	print "TIME_gE1_call", t2-t1
	print "I(theta,X,",mod+1 ,") = ", MutInfo1
seed(123)
main()
t4=time.time()
print "TIME_total", t4-t3
