#!/usr/bin/python2.5

from numpy import *
from numpy.random import *
import math
import re

import cudasim.Lsoda as Lsoda

from abcsysbio import parse_infoEnt
from abcsysbio_parser import ParseAndWrite

try:
	import cPickle as pickle
except:
	import pickle

import time
import sys
sys.path.insert(0, '/cluster/home/saw112/work/git_group_project/error_checks')
sys.path.insert(0, ".")
import parse_infoEnt_new_2


def run_cudasim(m_object,inpath=""):
	modelTraj = []
	modelInstance = Lsoda.Lsoda(m_object.times, list(set(m_object.cuda)), dt=m_object.dt, inpath=inpath)
	if m_object.ncompparams_all > 0:
		parameters=concatenate((m_object.compsSample,m_object.parameterSample),axis=1)
	else:
		parameters = m_object.parameterSample


	result = modelInstance.run(parameters, m_object.speciesSample, constant_sets = not(m_object.initialprior), pairings=m_object.pairParamsICS)
	
	if type(result)==list:
		result = [x[:,0] for x in result]
	else:
		result = [result[:,0]]

	print "-----Sorting NaNs from CUDA-Sim output-----"

	m_object.sortCUDASimoutput(list(set(m_object.cuda)),result)

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

	if(FmaxDistTraj<preci):
		scale = pow(1.79*pow(10,300),1.0/(ftheta[0].shape[1]*ftheta[0].shape[2]))
	else:
		scale = pow(preci,1.0/(ftheta[0].shape[1]*ftheta[0].shape[2]))*1.0/FmaxDistTraj

	return scale

def pickle_object(object):
	pickle.dump(object, open("save_point.pkl", "wb"))

def unpickle_object(filename="savepoint.pkl"):
	object = pickle.load(open(filename, "rb"))

	return object
