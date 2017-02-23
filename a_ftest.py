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

	print result[0]
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




def getEntropy1(data,N,sigma,theta,maxDistTraj):

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

	__global__ void distance1(int Ni, int Nj, int M, int P, float sigma, float pi, double a, double *d1, double *d2, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x1;
	x1 = 0.0;
	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				   x1 = x1 +log(a) - ( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma);
			}
	}

	res1[idx2d(i,j,Nj)] = exp(x1);
	}
	""")

	# prepare data
#                          x1 = x1 * a * 1.0*exp(-( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma));




	N1 = 10000
	N2 = 90000

	d1 = data[0:N1,:,:].astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)
	d3 = array(theta)[0:N1,:,:].astype(float64)

	print "shape imp:", shape(d1)
	# split data to correct size to run on GPU
	Max = 256.0
	dist_gpu1 = mod.get_function("distance1")
	print "registers: ", dist_gpu1.num_regs

	preci = pow(10,-34)
	FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))
	print "FmaxDistTraj:",FmaxDistTraj
	if(FmaxDistTraj<preci):
		a = pow(1.79*pow(10,300),1.0/(d1.shape[1]*d1.shape[2]))
  #     a = 1.0
	else:
 #        a = pow(preci,1.0/(d1.shape[1]*d1.shape[2]))*1.0/FmaxDistTraj
		a = pow(1.79*pow(10,300),1.0/(d1.shape[1]*d1.shape[2]))
 #      a = 1.0
	print "preci:", preci, "a:",a

	numRuns = int(ceil(N1/Max))
	print "numRuns: ", numRuns

	result2 = zeros([N1,numRuns])
	countsi = 0
	for i in range(numRuns):
		print "runs left:", numRuns - i

		si = int(Max)
		sj = int(Max)

		s = int(Max)
		if((s*(i+1)) > N1):
			si = int(N1 - Max*i)
		countsj = 0
		for j in range(numRuns):

			if((s*(j+1)) > N1):
				sj = int(N1 - Max*j)

			data1 = d1[(i*int(Max)):(i*int(Max)+si),:,:]
			data2 = d2[(j*int(Max)):(j*int(Max)+sj),:,:]

			Ni=data1.shape[0]
			Nj=data2.shape[0]

			M=data1.shape[1]
			P=data1.shape[2]
			res1 = zeros([Ni,Nj]).astype(float64)

			# invoke kernel
			R = 15.0
#           print "Ni:",Ni,"Nj:",Nj
			if(Ni<R):
				gi = 1
				bi = Ni
			else:
				bi = R
				gi = ceil(Ni/R)
			if(Nj<R):
				gj = 1
				bj = Nj
			else:
				bj = R
				gj = ceil(Nj/R)

			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float32(pi), float64(a), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

#             print "shape:", shape(res1)
#           print "hello:", res1
#           saveResults(res1,'distances'+str(i)+'_'+str(j)+'.txt')
			for k in range(si):
				result2[(i*int(Max)+k),j] = sum(res1[k,:])
			countsj = countsj+sj
		countsi = countsi+si


	sum1 = 0.0
	counter = 0
	counter2 = 0
 #   print result2

	for i in range(N1):
		if(isnan(sum(result2[i,:]))): counter=counter+1
		if(isinf(log(sum(result2[i,:])))): counter2=counter2+1
		else:
#             print result2
			  sum1 = sum1 - log(sum(result2[i,:])) + log(float(N2)) + M*P*log(a) +  M*P*log(2.0*pi*sigma*sigma)
#             sum1 = sum1 - log(sum(result2[i,:])) + log(float(N2)) + M*P/2.0*log(2.0*pi*sigma*sigma) -  M*P*log(a) -log(float(N2))

	Info = sum1/float(N1)

	Info = Info - M*P/2.0*log(2.0*pi*sigma*sigma*exp(1))
  # Info = Info - M*P/2.0*log(2.0*pi*0.00001*0.00001*exp(1))

	print "counter: ",counter,"counter2: ",counter2

 #   saveResults(result2,"distances.txt")

	out = open('results','w')

	print >>out, "counter: ",counter2
	print >>out, "mutual info: ", Info

	out.close()

	return(Info)


def getEntropy2(dataRef,dataY,N,sigma,theta1,theta2,maxDistTraj):

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

	__global__ void distance1(int Ni, int Nj, int M, int P, int M2, int P2, float sigma, float pi, double a, double *d1, double *d2, double *d3, double *d4, double *res1)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x1;
	x1 = 0.0;
	double x3;
	x3 = 0.0;
	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				   x1 = x1 + a - ( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma);
			}
	}


	for(int k=0; k<M2; k++){
			for(int l=0; l<P2; l++){
				   x3 = x3 + a - ( d4[idx3d(j,k,l,M2,P2)]-d3[idx3d(i,k,l,M2,P2)])*( d4[idx3d(j,k,l,M2,P2)]-d3[idx3d(i,k,l,M2,P2)])/(2.0*sigma*sigma);
			}
	}


	res1[idx2d(i,j,Nj)] = exp(x1+x3);
	}


	__global__ void distance2(int Ni, int Nj, int M, int P, int M2, int P2, float sigma, float pi, double a, double *d1, double *d2, double *d3, double *d4, double *res2, double *res3)
	{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if((i>=Ni)||(j>=Nj)) return;

	double x2;
	double x3;
	x2 = 0.0;
	x3 = 0.0;

	for(int k=0; k<M; k++){
			for(int l=0; l<P; l++){
				   x2 = x2 + a - ( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*( d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma);
			}
	}
	for(int k=0; k<M2; k++){
			for(int l=0; l<P2; l++){
				   x3 = x3 + a - ( d4[idx3d(j,k,l,M2,P2)]-d3[idx3d(i,k,l,M2,P2)])*( d4[idx3d(j,k,l,M2,P2)]-d3[idx3d(i,k,l,M2,P2)])/(2.0*sigma*sigma);
			}
	}

	res2[idx2d(i,j,Nj)] = exp(x2);
	res3[idx2d(i,j,Nj)] = exp(x3);


	}


	""")

		 #         x1 = x1 + a - ( d4[idx3d(j,k,l,M2,P2)]-d3[idx3d(i,k,l,M2,P2)])*( d4[idx3d(j,k,l,M2,P2)]-d3[idx3d(i,k,l,M2,P2)])/(2.0*sigma*sigma);


	# prepare data

	N1 = 1000
	N2 = 2500
	N3 = 2500
	N4 = 2500

	d1 = dataRef[0:N1,:,:].astype(float64)
	d2 = array(theta1)[N1:(N1+N2),:,:].astype(float64)
	d3 = dataY[0:N1,:,:].astype(float64)
	d4 = array(theta2)[N1:(N1+N2),:,:].astype(float64)

	d5 = dataRef[0:N1,:,:].astype(float64)
	d6 = array(theta1)[(N1+N2):(N1+N2+N3),:,:].astype(float64)
	d7 = dataY[0:N1,:,:].astype(float64)
	d8 = array(theta2)[(N1+N2+N3):(N1+N2+N3+N4),:,:].astype(float64)


	# split data to correct size to run on GPU
	Max = 256.0
	dist_gpu1 = mod.get_function("distance1")
	dist_gpu2 = mod.get_function("distance2")


	preci = pow(10,-34)
	M = float(max(d1.shape[1],d3.shape[1]))
	P = float(max(d1.shape[2],d3.shape[2]))

	aa1 = log(preci)/(2.0*M*P) + (maxDistTraj*maxDistTraj)/(2.0*sigma*sigma)

	limMax = pow(10,300)

#    aa2 = 690.7755278982136/(2.0*M*P)

	aa2 = math.log(limMax)/(2.0*M*P)


	print "aa1: ",aa1, "aa2: ",aa2
	if(aa1<aa2): a = aa1
	else: a = 0.0

  #   FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))
 #    print "FmaxDistTraj:",FmaxDistTraj
 #    if(FmaxDistTraj<preci):
 #      a = pow(1.79*pow(10,300),1.0/(2.0*d1.shape[1]*d1.shape[2]*d3.shape[1]*d3.shape[2]))
  #   else:
 #      a = pow(1.79*pow(10,300),1.0/(2.0*d1.shape[1]*d1.shape[2]*d3.shape[1]*d3.shape[2]))


	print "preci:", preci,"a:",a


	numRuns = int(ceil(N1/Max))
	print "numRuns: ", numRuns
	numRuns2 = int(ceil(N2/Max))
	print "numRuns2: ", numRuns2

	result1 = zeros([N1,numRuns2])
	result2 = zeros([N1,numRuns2])
	result3 = zeros([N1,numRuns2])

	countsi = 0
	for i in range(numRuns):
		print "runs left:", numRuns - i

		si = int(Max)
		sj = int(Max)

		s = int(Max)
		if((s*(i+1)) > N1):
			si = int(N1 - Max*i)
		countsj = 0
		for j in range(numRuns2):
			if((s*(j+1)) > N2):
				sj = int(N2 - Max*j)

			data1 = d1[(i*int(Max)):(i*int(Max)+si),:,:]
			data2 = d2[(j*int(Max)):(j*int(Max)+sj),:,:]
			data3 = d3[(i*int(Max)):(i*int(Max)+si),:,:]
			data4 = d4[(j*int(Max)):(j*int(Max)+sj),:,:]

			data5 = d5[(i*int(Max)):(i*int(Max)+si),:,:]
			data6 = d6[(j*int(Max)):(j*int(Max)+sj),:,:]
			data7 = d7[(i*int(Max)):(i*int(Max)+si),:,:]
			data8 = d8[(j*int(Max)):(j*int(Max)+sj),:,:]

			Ni=data1.shape[0]
			Nj=data2.shape[0]

			M=data1.shape[1]
			P=data1.shape[2]

			M2=data3.shape[1]
			P2=data3.shape[2]

			res1 = zeros([Ni,Nj]).astype(float64)
			res2 = zeros([Ni,Nj]).astype(float64)
			res3 = zeros([Ni,Nj]).astype(float64)


			# invoke kernel
			R = 15.0
#           print "Ni:",Ni,"Nj:",Nj
			if(Ni<R):
				gi = 1
				bi = Ni
			else:
				bi = R
				gi = ceil(Ni/R)
			if(Nj<R):
				gj = 1
				bj = Nj
			else:
				bj = R
				gj = ceil(Nj/R)

			dist_gpu1(int32(Ni), int32(Nj), int32(M), int32(P), int32(M2), int32(P2), float32(sigma), float32(pi), float64(a), driver.In(data1), driver.In(data2), driver.In(data3), driver.In(data4), driver.Out(res1),block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			dist_gpu2(int32(Ni), int32(Nj), int32(M), int32(P), int32(M2), int32(P2), float32(sigma), float32(pi), float64(a), driver.In(data5), driver.In(data6), driver.In(data7), driver.In(data8), driver.Out(res2), driver.Out(res3),block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))


			for k in range(si):
				result1[(i*int(Max)+k),j] = sum(res1[k,:])
				result2[(i*int(Max)+k),j] = sum(res2[k,:])
				result3[(i*int(Max)+k),j] = sum(res3[k,:])
			countsj = countsj+sj
		countsi = countsi+si

	sum1 = 0.0
	a1 = 0.0
	a2 = 0.0
	a3 = 0.0

 #   print res1
 #   print result1
 #   print result2
 #   print result3

	counter1 = 0
	counter2 = 0
	counter3 = 0
	for i in range(N1):
		if(isinf(log(sum(result1[i,:])))):
			counter1=counter1+1
			print "inf 1:",i
		if(isinf(log(sum(result2[i,:])))):
			counter2=counter2+1
			print "inf 2:",i
		if(isinf(log(sum(result3[i,:])))):
			counter3=counter3+1
			print "inf 3:",i
		if(isinf(log(sum(result1[i,:]))) or isinf(log(sum(result2[i,:]))) or isinf(log(sum(result3[i,:])))):
			print "inf:",i
		else:
			sum1 = sum1 + log(sum(result1[i,:])) - log(sum(result2[i,:])) - log(sum(result3[i,:])) - log(float(N2)) + log(float(N3)) + log(float(N4))


		a1 = a1 + log(sum(result1[i,:])) - log(N2) - 2.0*M*P*log(2.0*pi*sigma*sigma) - 2*M*P*a
		a2 = a2 - log(sum(result2[i,:])) + log(N3) +  M*P*log(2.0*pi*sigma*sigma) + M*P*a
		a3 = a3 - log(sum(result3[i,:])) + log(N4) +  M*P*log(2.0*pi*sigma*sigma) + M*P*a


	print "a1: ", a1/float(N1) , "a2: ", a2/float(N1), "a3: ", a3/float(N1)
	print "all: ",  a1/float(N1) + a2/float(N1) + a3/float(N1)

	Info = sum1/float(N1)

	print "counter1: ", counter1
	print "counter2: ", counter2
	print "counter3: ", counter3
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


	sampleFromPost = True
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


	saveIndex = []
	saveResult = []

	for mod in range(info_new.nmodels):

		print "Model:",mod+1

		modelTraj.append([])
		accepted = 0
		while(accepted<info_new.particles):
			if(mod==0): parameters = zeros( [info_new.particles,len(info_new.prior[mod])] )
			species = zeros([info_new.particles,info_new.nspecies[mod]])

			for i in range(info_new.particles):
				for j in range(info_new.nspecies[mod]):
					if(info_new.x0prior[mod][j][0]==0):
						species[i,j] = info_new.x0prior[mod][j][1]
					if(info_new.x0prior[mod][j][0]==2):
						species[i,j] = uniform(low=info_new.x0prior[mod][j][1], high=info_new.x0prior[mod][j][2],size=(1,1,1))[0][0]

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
				infileName1 = "param.txt"
				infileName2 = "init.txt"
				in_file1=open(infileName1, "r")
				matrix1=[]
				param=[]
				counter=0
				for in_line in in_file1.readlines():
					in_line=in_line.rstrip()
					matrix1.append([])
					param.append([])
					matrix1[counter]=in_line.split(" ")
					param[counter] = map(float, matrix1[counter])
					counter=counter+1
				in_file1.close

				parameters = zeros( [info_new.particles,len(param[0])] )
				for i in range(info_new.particles):
					for j in range(len(param[0])):
						parameters[i,j] = param[i][j]

			if(sampleFromPost==True):

				in_file2=open(infileName2, "r")
				matrix2=[]
				init=[]
				counter=0
				for in_line in in_file2.readlines():
					in_line=in_line.rstrip()
					matrix2.append([])
					init.append([])
					matrix2[counter]=in_line.split(" ")
					init[counter] = map(float, matrix2[counter])
					counter=counter+1
				in_file2.close

				for i in range(info_new.particles):
					for j in range(len(init[0])):
						species[i,j] = init[i][j]




			timesMod2 = range(0,3600,100)
 #          timesMod2 = [0,   50,  100,  110,  120,  130,  140,  150,  160 , 200,  250,  300,  350,  400,  780, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1260, 1310, 1360, 1410, 1460, 1840, 2220, 2230, 2240, 2250, 2260, 2270, 2280, 2320, 2370, 2420, 2470, 2520, 3000]
 #          timesMod2 = info_new.times

			# simulate model mod
			cudaCode = info_new.name[mod] + '.cu'
			print "cuda code:", cudaCode
			if mod==0: modelInstance = EulerMaruyama.EulerMaruyama(timesMod2, cudaCode, beta=1.0, dt=0.001)
			if mod==1: modelInstance = EulerMaruyama.EulerMaruyama(info_new.times, cudaCode, beta=1.0, dt=0.001)
			result0 = modelInstance.run(parameters, species)

			n = shape(result0)[2]
			result = result0[:,:,0:n,:]
			print shape(result)

			# merge result and parameters

			# check for NA

			if mod==0: index = [p for p, i in enumerate(isnan(sum(result[:,0,35:36,:],axis=2))) if i==False]
			if mod==1: index = [p for p, i in enumerate(isnan(sum(result[:,0,7:8,:],axis=2))) if i==False]
			print len(index)
			saveIndex.append(index)

			maxLimit = 3000
#           print (result[:,0,7:8,2:3]+result[:,0,7:8,3:4])
			if mod==0:
				index1 = [p for p, i in enumerate((result[:,0,35:36,5:6]+result[:,0,35:36,7:8])) if i<maxLimit ]
				index2 = [p for p, i in enumerate(result[:,0,35:36,8:9]) if i<maxLimit ]

			if mod==1:
				index1 = [p for p, i in enumerate((result[:,0,7:8,5:6]+result[:,0,7:8,7:8])) if i<maxLimit ]
				index2 = [p for p, i in enumerate(result[:,0,7:8,8:9]) if i<maxLimit ]

			print len(index1)
			print len(index2)
			saveIndex.append(index1)
			saveIndex.append(index2)


			saveResult.append(result)
			accepted = 400000



	for mod in range(info_new.nmodels):

		parametersKeepFinal = []
		parametersRemoveFinal = []
		resultKeepFinal = []
		resultRemoveFinal = []


		if(len(saveIndex[0])>0):
			print len(set(saveIndex[0]) & set(saveIndex[1]) & set(saveIndex[2]) & set(saveIndex[3]) & set(saveIndex[4]) & set(saveIndex[5]))

			for i in (set(saveIndex[0]) & set(saveIndex[1]) & set(saveIndex[2]) & set(saveIndex[3])& set(saveIndex[4]) & set(saveIndex[5])):
				resultKeepFinal.append(saveResult[mod][i][0])
				parametersKeepFinal.append(parameters[i])

		#use the first N particles only
		del resultKeepFinal[info_new.particles:]
		del parametersKeepFinal[info_new.particles:]


		# create list that contains parameters theta and variables x
		modelTraj[mod].append(parametersKeepFinal)
		modelTraj[mod].append(resultKeepFinal)

		print "shape important: ", shape(modelTraj[mod][1])

		print "model ",mod,": simulation done!"
		# write out sampled parameters and trajectories for accepted and rejected particles

		fileName =  "acceptedParticles/model"+`mod+1`+"_trajectories_accepted.txt"
		if mod==1: print_results(resultKeepFinal,fileName,info_new.times)
		if mod==0: print_results(resultKeepFinal,fileName,timesMod2)

		fileName =  "acceptedParticles/model"+`mod+1`+"_parameters_accepted.txt"
#      print_parameters(parametersKeepFinal,fileName)

		fileName =  "rejectedParticles/model"+`mod+1`+"_parameters_rejected.txt"
		#       print_parameters(parametersRemoveFinal,fileName)



	sigma = 100.0
	print "Sigma: ", sigma

	ftheta = []
	maxDistTraj = []
	for mod in range(info_new.nmodels):
		print shape(modelTraj[mod][1])
		trajTemp = array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8]
		print "shape traj:", shape(trajTemp)
		noise = normal(loc=0.0, scale=sigma,size=(shape(trajTemp)[0],shape(trajTemp)[1],shape(trajTemp)[2]))
		print "s trajTemp: ",shape(trajTemp)
		print "s noise: ", shape(noise)
		temp = trajTemp[:,:,:] + noise

		distance = []
		for tp in range(shape(trajTemp)[1]):
			distance.append( max(amax(temp[:,tp,:]),amax(trajTemp[:,tp,:])) - min(amin(temp[:,tp,:]),amin(trajTemp[:,tp,:]))   )


		maxDistTraj.append(amax(array(distance)))
		print "maxDistTraj:", maxDistTraj
		ftheta.append(temp)


#    ftheta[1] = ftheta[0]
#    modelTraj[1][1] = modelTraj[0][1]

	print("Simulation done")
	print "------------------------ "
	print " "

	# compute I(x,y) for reference model

	if referenceModel == True:
		MutInfo2 = []
		print("Mutual information calculation 2... ")
		MutInfo2.append("NA")
		for mod in range(1,info_new.nmodels):
			N = info_new.particles
			MutInfo2.append(getEntropy2(ftheta[0],ftheta[mod],N,sigma,array(modelTraj[0][1])[:,:,5:6]+array(modelTraj[0][1])[:,:,7:8],array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8],max(maxDistTraj)))
			print "I(X(reference model),X(model",mod+1,") = ", MutInfo2[mod]


	print "------------------------------------------------------------------------ "


	sigma = 50.0
	print "Sigma: ", sigma

	ftheta = []
	maxDistTraj = []
	for mod in range(info_new.nmodels):
		print shape(modelTraj[mod][1])
		trajTemp = array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8]
		print "shape traj:", shape(trajTemp)
		noise = normal(loc=0.0, scale=sigma,size=(shape(trajTemp)[0],shape(trajTemp)[1],shape(trajTemp)[2]))
		print "s trajTemp: ",shape(trajTemp)
		print "s noise: ", shape(noise)
		temp = trajTemp[:,:,:] + noise

		distance = []
		for tp in range(shape(trajTemp)[1]):
			distance.append( max(amax(temp[:,tp,:]),amax(trajTemp[:,tp,:])) - min(amin(temp[:,tp,:]),amin(trajTemp[:,tp,:]))   )


		maxDistTraj.append(amax(array(distance)))
		print "maxDistTraj:", maxDistTraj
		ftheta.append(temp)


#    ftheta[1] = ftheta[0]
#    modelTraj[1][1] = modelTraj[0][1]

	print("Simulation done")
	print "------------------------ "
	print " "

	# compute I(x,y) for reference model

	if referenceModel == True:
		MutInfo2 = []
		print("Mutual information calculation 2... ")
		MutInfo2.append("NA")
		for mod in range(1,info_new.nmodels):
			N = info_new.particles
			MutInfo2.append(getEntropy2(ftheta[0],ftheta[mod],N,sigma,array(modelTraj[0][1])[:,:,5:6]+array(modelTraj[0][1])[:,:,7:8],array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8],max(maxDistTraj)))
			print "I(X(reference model),X(model",mod+1,") = ", MutInfo2[mod]


	print "------------------------------------------------------------------------ "


	sigma = 40.0
	print "Sigma: ", sigma

	ftheta = []
	maxDistTraj = []
	for mod in range(info_new.nmodels):
		print shape(modelTraj[mod][1])
		trajTemp = array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8]
		print "shape traj:", shape(trajTemp)
		noise = normal(loc=0.0, scale=sigma,size=(shape(trajTemp)[0],shape(trajTemp)[1],shape(trajTemp)[2]))
		print "s trajTemp: ",shape(trajTemp)
		print "s noise: ", shape(noise)
		temp = trajTemp[:,:,:] + noise

		distance = []
		for tp in range(shape(trajTemp)[1]):
			distance.append( max(amax(temp[:,tp,:]),amax(trajTemp[:,tp,:])) - min(amin(temp[:,tp,:]),amin(trajTemp[:,tp,:]))   )


		maxDistTraj.append(amax(array(distance)))
		print "maxDistTraj:", maxDistTraj
		ftheta.append(temp)


#    ftheta[1] = ftheta[0]
#    modelTraj[1][1] = modelTraj[0][1]

	print("Simulation done")
	print "------------------------ "
	print " "

	# compute I(x,y) for reference model

	if referenceModel == True:
		MutInfo2 = []
		print("Mutual information calculation 2... ")
		MutInfo2.append("NA")
		for mod in range(1,info_new.nmodels):
			N = info_new.particles
			MutInfo2.append(getEntropy2(ftheta[0],ftheta[mod],N,sigma,array(modelTraj[0][1])[:,:,5:6]+array(modelTraj[0][1])[:,:,7:8],array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8],max(maxDistTraj)))
			print "I(X(reference model),X(model",mod+1,") = ", MutInfo2[mod]


	print "------------------------------------------------------------------------ "


	sigma = 30.0
	print "Sigma: ", sigma

	ftheta = []
	maxDistTraj = []
	for mod in range(info_new.nmodels):
		print shape(modelTraj[mod][1])
		trajTemp = array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8]
		print "shape traj:", shape(trajTemp)
		noise = normal(loc=0.0, scale=sigma,size=(shape(trajTemp)[0],shape(trajTemp)[1],shape(trajTemp)[2]))
		print "s trajTemp: ",shape(trajTemp)
		print "s noise: ", shape(noise)
		temp = trajTemp[:,:,:] + noise

		distance = []
		for tp in range(shape(trajTemp)[1]):
			distance.append( max(amax(temp[:,tp,:]),amax(trajTemp[:,tp,:])) - min(amin(temp[:,tp,:]),amin(trajTemp[:,tp,:]))   )


		maxDistTraj.append(amax(array(distance)))
		print "maxDistTraj:", maxDistTraj
		ftheta.append(temp)


#    ftheta[1] = ftheta[0]
#    modelTraj[1][1] = modelTraj[0][1]

	print("Simulation done")
	print "------------------------ "
	print " "

	# compute I(x,y) for reference model

	if referenceModel == True:
		MutInfo2 = []
		print("Mutual information calculation 2... ")
		MutInfo2.append("NA")
		for mod in range(1,info_new.nmodels):
			N = info_new.particles
			MutInfo2.append(getEntropy2(ftheta[0],ftheta[mod],N,sigma,array(modelTraj[0][1])[:,:,5:6]+array(modelTraj[0][1])[:,:,7:8],array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8],max(maxDistTraj)))
			print "I(X(reference model),X(model",mod+1,") = ", MutInfo2[mod]


	print "------------------------------------------------------------------------ "



	sigma = 20.0
	print "Sigma: ", sigma

	ftheta = []
	maxDistTraj = []
	for mod in range(info_new.nmodels):
		print shape(modelTraj[mod][1])
		trajTemp = array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8]
		print "shape traj:", shape(trajTemp)
		noise = normal(loc=0.0, scale=sigma,size=(shape(trajTemp)[0],shape(trajTemp)[1],shape(trajTemp)[2]))
		print "s trajTemp: ",shape(trajTemp)
		print "s noise: ", shape(noise)
		temp = trajTemp[:,:,:] + noise

		distance = []
		for tp in range(shape(trajTemp)[1]):
			distance.append( max(amax(temp[:,tp,:]),amax(trajTemp[:,tp,:])) - min(amin(temp[:,tp,:]),amin(trajTemp[:,tp,:]))   )


		maxDistTraj.append(amax(array(distance)))
		print "maxDistTraj:", maxDistTraj
		ftheta.append(temp)


#    ftheta[1] = ftheta[0]
#    modelTraj[1][1] = modelTraj[0][1]

	print("Simulation done")
	print "------------------------ "
	print " "

	# compute I(x,y) for reference model

	if referenceModel == True:
		MutInfo2 = []
		print("Mutual information calculation 2... ")
		MutInfo2.append("NA")
		for mod in range(1,info_new.nmodels):
			N = info_new.particles
			MutInfo2.append(getEntropy2(ftheta[0],ftheta[mod],N,sigma,array(modelTraj[0][1])[:,:,5:6]+array(modelTraj[0][1])[:,:,7:8],array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8],max(maxDistTraj)))
			print "I(X(reference model),X(model",mod+1,") = ", MutInfo2[mod]


	print "------------------------------------------------------------------------ "


	sigma = 15.0
	print "Sigma: ", sigma

	ftheta = []
	maxDistTraj = []
	for mod in range(info_new.nmodels):
		print shape(modelTraj[mod][1])
		trajTemp = array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8]
		print "shape traj:", shape(trajTemp)
		noise = normal(loc=0.0, scale=sigma,size=(shape(trajTemp)[0],shape(trajTemp)[1],shape(trajTemp)[2]))
		print "s trajTemp: ",shape(trajTemp)
		print "s noise: ", shape(noise)
		temp = trajTemp[:,:,:] + noise

		distance = []
		for tp in range(shape(trajTemp)[1]):
			distance.append( max(amax(temp[:,tp,:]),amax(trajTemp[:,tp,:])) - min(amin(temp[:,tp,:]),amin(trajTemp[:,tp,:]))   )


		maxDistTraj.append(amax(array(distance)))
		print "maxDistTraj:", maxDistTraj
		ftheta.append(temp)


#    ftheta[1] = ftheta[0]
#    modelTraj[1][1] = modelTraj[0][1]

	print("Simulation done")
	print "------------------------ "
	print " "

	# compute I(x,y) for reference model

	if referenceModel == True:
		MutInfo2 = []
		print("Mutual information calculation 2... ")
		MutInfo2.append("NA")
		for mod in range(1,info_new.nmodels):
			N = info_new.particles
			MutInfo2.append(getEntropy2(ftheta[0],ftheta[mod],N,sigma,array(modelTraj[0][1])[:,:,5:6]+array(modelTraj[0][1])[:,:,7:8],array(modelTraj[mod][1])[:,:,5:6]+array(modelTraj[mod][1])[:,:,7:8],max(maxDistTraj)))
			print "I(X(reference model),X(model",mod+1,") = ", MutInfo2[mod]



main()
