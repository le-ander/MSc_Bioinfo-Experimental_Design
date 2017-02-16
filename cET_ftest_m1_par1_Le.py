#!/usr/bin/python2.5

from numpy import *
from numpy.random import *
import abcsysbio
import sys
import re
import time, os

import cudasim
import cudasim.EulerMaruyama as EulerMaruyama
import cudasim.Gillespie as Gillespie
import cudasim.Lsoda as Lsoda

from pycuda import compiler, driver
from pycuda import autoinit

from abcsysbio import parse_infoEnt
from abcsysbio import model_py
from abcsysbio import model_cu
from abcsysbio import model_c
from abcsysbio import data
from abcsysbio import input_output

import abcsysbio_parser
from abcsysbio_parser import ParseAndWrite

import sys
sys.path.insert(0, ".")



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


def checkNAs(result):

	index = []
	for i in range(len(result)):
		# loop over species
		l = 0
		isok = True
		while (isok and (l<(len(result[i][0])-1))):
			# loop over timepoints
			for k in range(len(result[i][0][l])):
				if isnan(result[i][0][l][k]) ==True:
					index.append(i)
					isok = False
					break
			l = l+1
	return(index)

def removeNAs(result, parameter,index):
	p = parameter
	x = result
	xKeep = []
	pKeep = []
	xRemove = []
	pRemove = []

	for i in range(len(p)):
		rem = False
		for j in range(len(index)):
			if index[j] == i:
				pRemove.append(p[i])
				xRemove.append(x[i][0])
				rem = True
		if rem == False:
			pKeep.append(p[i])
			xKeep.append(x[i][0])

	return(xKeep,pKeep,xRemove,pRemove)


def print_results(result, outfile,timepoints):
	out = open(outfile,'w')
	print >>out, 0, 0, 0,
	for i in range(len(timepoints)):
		print >>out, timepoints[i],
	print >>out, ""
	# loop over threads
	for i in range(len(result)):
		# loop over species
		for l in range(len(result[i][0])):

			print >>out, i,"0",l,
			for k in range(len(timepoints)):
				print >>out, result[i][k][l],
			print >>out, ""

	out.close()



def print_parameters(param, outfile):

	out = open(outfile,'w')

	for i in range(len(param)):
		for j in range(len(param[i])):
			print >>out, param[i][j],
		print >>out, ""

	out.close()

def prod( iterable ):
	p= 1
	for n in iterable:
		p *= n
	return p


def saveResults(result,outfile):

	out = open(outfile,'w')

	for i in range(shape(result)[0]):
		for j in range(shape(result)[1]):
			print >>out, result[i,j],
		print >>out, ""

	out.close()









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

	__global__ void distance2(int Nj, int M, int P, float sigma, double scale, double *d1, double *d3, double *res2)
	{
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if(j>=Nj) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				x1 = x1 + log(scale) - ( d3[idx3d(j,k,l,M,P)]-d1[idx2d(k,l,P)])*(d3[idx3d(j,k,l,M,P)]-d1[idx2d(k,l,P)])/(2.0*sigma*sigma);
			}
	}

	res2[j] = exp(x1);
	}
	""")

	# Assigning main kernel function to a variable
	dist_gpu1 = mod.get_function("distance1")

	##should be defined as an int, can then clean up formulas further down
	gridmax = 256.0 # Define square root of maximum threads per grid
	blockmax = 16.0 # Maximum threads per block

	# Determine required number of runs for i and j
	##need float here?
	numRuns_i = int(ceil(N1/float(gridmax)))
	numRuns_j = int(ceil(N2/float(gridmax)))

	res_t2 = zeros([N1,numRuns_j])

	# Prepare data
	d1 = data[0:N1,:,:].astype(float64)
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
		print "LOGSCALE", log(scale)


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
			print "dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float64(scale),int(bi),int(bj),int(gi),int(gj)",int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float64(scale),int(bi),int(bj),int(gi),int(gj)
			# Invoke GPU calculations (takes data1 and data2 as input, outputs res1)
			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float64(scale), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))
			print res1
			# First summation (could be done on GPU?)
			for k in range(Ni):
				res_t2[(i*int(gridmax)+k),j] = sum(res1[k,:])
			#print res_t2
	sum1 = 0.0
	count_na = 0
	count_inf = 0

	for i in range(N1):
		if(isnan(sum(res_t2[i,:]))): count_na += 1
		elif(isinf(log(sum(res_t2[i,:])))): count_inf += 1
		else:
			sum1 += log(sum(res_t2[i,:])) - log(float(N2)) - M*P*log(scale) -  M*P*log(2.0*pi*sigma*sigma)
	print count_na, count_inf
	print "SUM1", sum1

######## part A finished with results saved in sum1

	dist_gpu2 = mod.get_function("distance2")

	##need this defined again here??
	gridmax = 256.0
	blockmax = 15.0

	numRuns_j2 = int(ceil(N3/gridmax))

	res_t2 = zeros([N1,numRuns_j2])

	d3 = array(theta)[(N1+N2):(N1+N2+N1*N3),:,:].astype(float64)

	print "DATA", shape(d1), shape(d3), gridmax, scale, shape(res_t2), numRuns_j2, N1

	for i in range(N1):

		data1 = d1[i,:,:]

		Nj = int(gridmax)

		for j in range(numRuns_j2):
			#print "runs left:", numRuns_j2 - j

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

			print int32(Nj), int32(M), int32(P), float32(sigma), float64(scale), int(bj),int(gj)

			dist_gpu2(int32(Nj), int32(M), int32(P), float32(sigma), float64(scale), driver.In(data1), driver.In(data3),  driver.Out(res2), block=(1,int(bj),1), grid=(1,int(gj)))

			res_t2[i,j] = sum(res2[:])

	sumstatic = 0.0
	sum2 = 0.0
	count2_na = 0
	count2_inf = 0

	for i in range(N1):
		if(isnan(sum(res_t2[i,:]))): count2_na += 1
		elif(isinf(log(sum(res_t2[i,:])))): count2_inf += 1
		else:
			sum2 += log(sum(res_t2[i,:])) - log(float(N3)) - M*P*log(scale) -  M*P*log(2.0*pi*sigma*sigma)
			sumstatic += - log(float(N3)) - M*P*log(scale) -  M*P*log(2.0*pi*sigma*sigma)
	print "COUNTER", count2_na, count2_inf
	print "SUM2", sum2
	print "sumstatic", sumstatic
	######## part B finished with results saved in sum2

	Info = (sum2 - sum1)/float(N1 - count_na - count_inf - count2_na - count2_inf)
	print "DIST_GPU2.NUM_REGS", dist_gpu2.num_regs
	return(Info)


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


def scaling(modelTraj, ftheta, sigma):
	maxDistTraj = max([math.fabs(amax(modelTraj) - amin(ftheta)),math.fabs(amax(ftheta) - amin(modelTraj))])

	preci = pow(10,-34)
	FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))

	if(FmaxDistTraj<preci):
		scale = pow(1.79*pow(10,300),1.0/(ftheta[0].shape[1]*ftheta[0].shape[2]))
	else:
		scale = pow(preci,1.0/(ftheta[0].shape[1]*ftheta[0].shape[2]))*1.0/FmaxDistTraj

	return scale



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

	try:
		os.mkdir("acceptedParticles")
	except:
		print "\nThe folder acceptedParticles already exists!\n"
		sys.exit()

	try:
		os.mkdir("rejectedParticles")
	except:
		print "\nThe folder rejectedParticles already exists!\n"
		sys.exit()

	for mod in range(info_new.nmodels):

		print "Model:",mod+1
		parametersKeepFinal = []
		parametersRemoveFinal = []
		resultKeepFinal = []
		resultRemoveFinal = []

		modelTraj.append([])
		accepted = 0


		N1 = 100
		N2 = 100
		N3 = 100
		indexTheta = 0

		while(accepted<(N1+N3+N1*N2)):
			if(mod==0):
				parametersN3 = zeros( [N3,len(info_new.prior[mod])] )
				parametersN2 = zeros( [N2,len(info_new.prior[mod])] )
				parametersN1 = zeros( [N1,len(info_new.prior[mod])] )



			if(mod==0 and sampleFromPost==False):
				# sample N3:

				for j in range(len(info_new.prior[mod])):
					if(info_new.prior[mod][j][0]==0):
						for i in range(N3):
							parametersN3[i,j] = info_new.prior[mod][j][1]
					if(info_new.prior[mod][j][0]==2):
						parametersN3[:,j] = uniform(low=info_new.prior[mod][j][1], high=info_new.prior[mod][j][2],size=(1,1,N3))[0][0]
					if(info_new.prior[mod][j][0]==1):
						parametersN3[:,j] = normal(loc=info_new.prior[mod][j][1], scale=info_new.prior[mod][j][2],size=(1,1,N3))[0][0]

				# sample N1:

				for j in range(len(info_new.prior[mod])):
					if(info_new.prior[mod][j][0]==0):
						for i in range(N1):
							parametersN1[i,j] = info_new.prior[mod][j][1]
					if(info_new.prior[mod][j][0]==2):
						parametersN1[:,j] = uniform(low=info_new.prior[mod][j][1], high=info_new.prior[mod][j][2],size=(1,1,N1))[0][0]
					if(info_new.prior[mod][j][0]==1):
						parametersN1[:,j] = normal(loc=info_new.prior[mod][j][1], scale=info_new.prior[mod][j][2],size=(1,1,N1))[0][0]

				# sample N2:

				for j in range(len(info_new.prior[mod])):
					if(info_new.prior[mod][j][0]==0):
						for i in range(N2):
							parametersN2[i,j] = info_new.prior[mod][j][1]
					if(info_new.prior[mod][j][0]==2):
						parametersN2[:,j] = uniform(low=info_new.prior[mod][j][1], high=info_new.prior[mod][j][2],size=(1,1,N2))[0][0]
					if(info_new.prior[mod][j][0]==1):
						parametersN2[:,j] = normal(loc=info_new.prior[mod][j][1], scale=info_new.prior[mod][j][2],size=(1,1,N2))[0][0]

				# generate parametersNX:

				parametersNX = zeros( [N1*N2,len(info_new.prior[mod])] )
				for i in range(N1):
					parametersNX[range((i*N2),((i+1)*N2)),:] = parametersN2[:,:]
					parametersNX[range((i*N2),((i+1)*N2)),indexTheta] = repeat(parametersN1[i,indexTheta],N2)


				# paste all parameters N1,N3,NX:
				NX = N1*N2
				parameters = zeros( [N1+N3+NX,len(info_new.prior[mod])] )
				parameters[range(0,N1),:] = parametersN1
				parameters[range(N1,N1+N3),:] = parametersN3
				parameters[range(N1+N3,N1+N3+NX),:] = parametersNX

			species = zeros([N1+N3+NX,info_new.nspecies[mod]])


			for i in range(N1+N3+NX):
				for j in range(info_new.nspecies[mod]):
					if(info_new.x0prior[mod][j][0]==0):
						species[i,j] = info_new.x0prior[mod][j][1]
					if(info_new.x0prior[mod][j][0]==2):
						species[i,j] = uniform(low=info_new.x0prior[mod][j][1], high=info_new.x0prior[mod][j][2],size=(1,1,1))[0][0]








			# simulate model mod
			cudaCode = info_new.name[mod] + '.cu'
			modelInstance = Lsoda.Lsoda(info_new.times, cudaCode, dt=info_new.dt)
			result0 = modelInstance.run(parameters, species)
			print "parameters: ", shape(parameters)

			n = shape(result0)[2]
			result = result0[:,:,0:n,:]
			print shape(result)

			# check for NA


			index = [p for p, i in enumerate(isnan(sum(result[:,0,:,2:3],axis=1))) if i==False]
			if(len(index)>0):
				for i in index:
					resultKeepFinal.append(result[i][0])
					parametersKeepFinal.append(parameters[i])


			accepted = len(parametersKeepFinal)
			print "accepted particles: ",accepted

		#use the first N1+N3+N1*N2 particles only
		del resultKeepFinal[(N1+N3+N1*N2):]
		del parametersKeepFinal[(N1+N3+N1*N2):]


		# create list that contains parameters theta and variables x
		modelTraj[mod].append(parametersKeepFinal)
		modelTraj[mod].append(resultKeepFinal)


		# write out sampled parameters and trajectories for accepted and rejected particles

		fileName =  "acceptedParticles/model"+`mod+1`+"_trajectories_accepted.txt"
#       print_results(resultKeepFinal,fileName,info_new.times)

		fileName =  "acceptedParticles/model"+`mod+1`+"_parameters_accepted.txt"
#       print_parameters(parametersKeepFinal,fileName)

		fileName =  "rejectedParticles/model"+`mod+1`+"_parameters_rejected.txt"
#       print_parameters(parametersRemoveFinal,fileName)



	sigma = 0.5
	print "%%%%%%%%%%%%%%%%%%%%"
	print "sigma: ",sigma

	ftheta = []
	for mod in range(info_new.nmodels):
		print shape(modelTraj[mod][1])
		trajTemp = array(modelTraj[mod][1])[:,:,0:1]
		print "shape traj:", shape(trajTemp)
		noise = normal(loc=0.0, scale=sigma,size=((N1+N3+N1*N2),len(info_new.times),shape(trajTemp)[2]))
		temp = trajTemp[:,:,:] + noise
		ftheta.append(temp)



	print("Simulation done")
	print "------------------------ "
	print " "
	print "SHAPE FTHETA[0]", ftheta[0].shape
	print "SHAPE modeltraj", array(modelTraj[0][1]).shape

	scale = scaling(array(modelTraj[0][1]),ftheta, sigma)

	# compute I(theta,x)
	print("Mutual information calculation 3... ")

	MutInfo2 = []
	for mod in range(info_new.nmodels):

		MutInfo2.append(getEntropy2(ftheta[mod],N1,N2,N3,sigma,array(modelTraj[mod][1])[:,:,0:1], scale))
		print "I(theta_i,X",mod+1,") = ", MutInfo2[mod]





seed(123)
main()
