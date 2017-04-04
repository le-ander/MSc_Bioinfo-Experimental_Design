#!/usr/bin/python2.5

from numpy import *
from numpy.random import *
import math
import re

#import Lsoda
import peitho.simulations.cudasim.Lsoda as Lsoda

try:
	import cPickle as pickle
except:
	import pickle

import time
import sys



# A function to run cudasim
##(method gets called by sorting_files)
##Arguments:
##inpath - input path for cuda code files
def run_cudasim(m_object,inpath=""):

	#Makes instance of Lsoda object
	modelInstance = Lsoda.Lsoda(m_object.times, list(set(m_object.cuda)), dt=m_object.dt, inpath=inpath)

	#Detects if compartments are used as they are treated as parameters
	if m_object.ncompparams_all > 0:
		parameters=concatenate((m_object.compsSample,m_object.parameterSample),axis=1)
	else:
		parameters = m_object.parameterSample

	#Sets species
	species = m_object.speciesSample

	#Runs cuda-sim
	result = modelInstance.run(parameters, species, constant_sets = not(m_object.initialprior), pairings=m_object.pairParamsICS)

	#Converts output of cuda-sim to a list
	if type(result)==list:
		result = [x[:,0] for x in result]
	else:
		result = [result[:,0]]

	#Sorts the output from cuda-sim
	print "-----Sorting NaNs from CUDA-Sim output-----"
	m_object.sortCUDASimoutput(list(set(m_object.cuda)),result)

# A function to pickle object
##(method gets called when required)
##Arguments:
##object - thing to be pickled
def pickle_object(object,name="savepoint.pkl"):
	pickle.dump(object, open(name, "wb"))

# A function to unpickle object
##(method gets called when required)
##Arguments:
##filename - file to be unpickled
def unpickle_object(filename="savepoint.pkl"):
	object = pickle.load(open(filename, "rb"))
	#Returns unpickled object
	return object
