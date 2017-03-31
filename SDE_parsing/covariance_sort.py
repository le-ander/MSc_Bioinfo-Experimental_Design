import numpy, sys, copy

def sort_mu_covariance(covariance_matrix,nspecies):
	nparams = covariance_matrix.shape[0]
	ntimes = covariance_matrix.shape[1]
	ncov = covariance_matrix.shape[2]-nspecies

	mu = covariance_matrix[:,:,0:nspecies]
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

	return mu, covariance_list
