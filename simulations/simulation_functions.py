#!/usr/bin/python2.5

from numpy import *
from numpy.random import *
import math
import re

import cudasim.Lsoda as Lsoda
import cudasim.EulerMaruyama as EulerMaruyama

try:
	import cPickle as pickle
except:
	import pickle

import time
import sys
sys.path.insert(1, '../error_checks')
sys.path.insert(1, ".")
import parse_infoEnt_new_2
import covariance_sort


# A function to run cudasim
##(method gets called by sorting_files)
##Arguments:
##inpath - input path for cuda code files
def run_cudasim(m_object,inpath="", intType="ODE",usesbml=False):

	if intType=="ODE":
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
	
	elif intType=="SDE":

		if usesbml == True:
			inpath_LNA = inpath+"/LNA"
		else:
			inpath_LNA = inpath

		modelInstance = EulerMaruyama.EulerMaruyama(m_object.times, list(set(m_object.cuda)), dt=m_object.dt, inpath=inpath, beta = m_object.beta) 
		LNAInstance = Lsoda.Lsoda(m_object.times, list(set(m_object.cuda)), dt=m_object.dt, inpath=inpath_LNA)
		
		if m_object.ncompparams_all > 0:
			parameters=concatenate((m_object.compsSample,m_object.parameterSample),axis=1)
		else:
			parameters = m_object.parameterSample

		#Sets species
		species = m_object.speciesSample

		try:
			nspecies = len(species[0][0,:])
		except:
			nspecies = len(species[0,:])

		var_IC = [""]*(nspecies*(nspecies+1)/2)
		pos = 0
		for i in range(nspecies):
			for j in range(i,nspecies):
				if i==j:
					var_IC[pos] = 1.0
				else:
					var_IC[pos] = 0.0
				pos+=1
		var_IC = array(var_IC)
		var_IC = tile((var_IC,)*parameters.shape[0],1)
		
		try:
			species_var = [concatenate((x,var_IC),axis=1) for x in species]
		except:
			species_var = concatenate((species,var_IC),axis=1)

		result = modelInstance.run(parameters, species, constant_sets = not(m_object.initialprior), pairings=m_object.pairParamsICS)
		
		result_var = LNAInstance.run(parameters, species_var, constant_sets = not(m_object.initialprior), pairings=m_object.pairParamsICS)
		
		cuda_order = list(set(m_object.cuda))

		m_object.sort_mu_covariance(cuda_order, [x[:,0,:,:] for x in result_var],nspecies)

		m_object.measured_species()
		sys.exit()

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
