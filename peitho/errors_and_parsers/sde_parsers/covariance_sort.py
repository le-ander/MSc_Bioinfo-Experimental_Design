import numpy, sys, copy

def sort_mu_covariance(sbml_object,covariance_matrix,nspecies):
	
	nparams = covariance_matrix.shape[0]
	ntimes = covariance_matrix.shape[1]
	ncov = covariance_matrix.shape[2]-nspecies

	mus = covariance_matrix[:,:,0:nspecies]
	variances_old = covariance_matrix[:,:,nspecies:]

	covariances = numpy.zeros((nparams,ntimes*nspecies,nspecies))

	for param in range(0,nparams):
		count_1 = -1
		for time in range(0,ntimes):
			count_1 += 1
			count_2 = -nspecies
			for i in range(0, nspecies):
				count_2 += nspecies-i
				for j in range(i, nspecies):
					if i == j:
						covariances[param,i+count_1*nspecies,j] = variances_old[param, time, count_2 + j]
					else:
						covariances[param,i+count_1*nspecies,j] = variances_old[param, time, count_2 + j]
						covariances[param,j+count_1*nspecies,i] = variances_old[param, time, count_2 + j]

	'''
	variances = covariance_matrix[:,:,nspecies:]

	#means_list = [[mu[i,j,:] for j in range(0,mu.shape[1])] for i in range(0,mu.shape[0])]
	covariance_list = [[""]*ntimes]*nparams

	temp_cov = numpy.zeros((nspecies,nspecies))

	for param in range(0,nparams):
		for time in range(0,ntimes):
			cov_mat = copy.deepcopy(temp_cov)
			count = -nspecies
			for i in range(0,nspecies):
				count += nspecies-i
				for j in range(i,nspecies):
					if j == i:
						cov_mat[i,i] = variances[param, time, count + i]
					else:
						cov_mat[i,j] = variances[param, time, count + j]
						cov_mat[j,i] = variances[param, time, count + j]
			covariance_list[param][time] = cov_mat
	'''
	
	return mus, covariances

def measured_species(sbml_object):
	print sbml_object.fitSpecies
	print "here213"
