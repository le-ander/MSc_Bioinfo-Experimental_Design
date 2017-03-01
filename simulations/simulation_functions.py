#!/usr/bin/python2.5

from numpy import *
from numpy.random import *
import math
import re

import cudasim.Lsoda as Lsoda



try:
	import cPickle as pickle
except:
	import pickle

import time
import sys
sys.path.insert(1, '../error_checks')
sys.path.insert(1, ".")
import parse_infoEnt_new_2


def run_cudasim(m_object,inpath="",analysis=0):
	modelTraj = []
	modelInstance = Lsoda.Lsoda(m_object.times, list(set(m_object.cuda)), dt=m_object.dt, inpath=inpath)
	if analysis == 0:
		if m_object.ncompparams_all > 0:
			parameters=concatenate((m_object.compsSample,m_object.parameterSample),axis=1)
		else:
			parameters = m_object.parameterSample

		species = m_object.speciesSample
	elif analysis == 1:
		if m_object.ncompparams_all > 0:
			parameters = concatenate((m_object.compsSample[:m_object.N1sample+m_object.N2sample+m_object.N3sample,:],m_object.parameterSample[:m_object.N1sample+m_object.N2sample+m_object.N3sample,:]), axis=1)
		else:
			parameters = m_object.parameterSample[:m_object.N1sample+m_object.N2sample+m_object.N3sample,:]

		species = [x[:m_object.N1sample+m_object.N2sample+m_object.N3sample,:] for x in m_object.speciesSample]

	elif analysis == 2:
		if m_object.ncompparams_all > 0:
			param_temp = concatenate((m_object.parameterSample[:m_object.N1sample+m_object.N2sample,:],m_object.parameterSample[m_object.N1sample+m_object.N2sample+m_object.N3sample:m_object.N1sample+m_object.N2sample+m_object.N3sample+m_object.N4sample,:]),axis=0)
			comps_temp = concatenate((m_object.compsSample[:m_object.N1sample+m_object.N2sample,:],m_object.compsSample[m_object.N1sample+m_object.N2sample+m_object.N3sample:m_object.N1sample+m_object.N2sample+m_object.N3sample+m_object.N4sample,:]),axis=0)	
			parameters = concatenate((comps_temp,param_temp),axis = 1)
		else:
			parameters = concatenate((m_object.parameterSample[:m_object.N1sample+m_object.N2sample,:],m_object.parameterSample[m_object.N1sample+m_object.N2sample+m_object.N3sample:m_object.N1sample+m_object.N2sample+m_object.N3sample+m_object.N4sample,:]),axis=0)
		
		species = [concatenate((x[:m_object.N1sample+m_object.N2sample,:],x[m_object.N1sample+m_object.N2sample+m_object.N3sample:m_object.N1sample+m_object.N2sample+m_object.N3sample:m_object.N1sample+m_object.N2sample+m_object.N3sample+m_object.N4sample,:]),axis=0) for x in m_object.speciesSample] 

	result = modelInstance.run(parameters, species, constant_sets = not(m_object.initialprior), pairings=m_object.pairParamsICS)
	
	if type(result)==list:
		result = [x[:,0] for x in result]
	else:
		result = [result[:,0]]

	print "-----Sorting NaNs from CUDA-Sim output-----"

	m_object.sortCUDASimoutput(list(set(m_object.cuda)),result, control = analysis)

def pickle_object(object):
	pickle.dump(object, open("save_point.pkl", "wb"))

def unpickle_object(filename="savepoint.pkl"):
	object = pickle.load(open(filename, "rb"))

	return object
