import numpy, sys, copy

def sort_mu_covariance(covariance_matrix,nspecies):
	nparams = covariance_matrix.shape[0]
	ntimes = covariance_matrix.shape[1]
	ncov = covariance_matrix.shape[2]-nspecies

	mu = covariance_matrix[:,:,0:nspecies]
	variances = covariance_matrix[:,:,nspecies:]

	covariance_list = [[""]*ntimes]*nparams
	eigenvalues = [[""]*ntimes]*nparams

	temp_cov = numpy.zeros((nspecies,nspecies))

	for param in range(0,nparams):
		for time in range(0,ntimes):
			cov_mat = copy.deepcopy(temp_cov)
			count = 0
			for i in range(0,nspecies):
				count += 1
				for j in range(i,nspecies):
					if i == j:
						cov_mat[i,i] = variances[param, time, i*(nspecies-count) + i]
					else:
						cov_mat[i,j] = variances[param, time, i*(nspecies-count) + j]
						cov_mat[j,i] = variances[param, time, i*(nspecies-count) + j]
			covariance_list[param][time] = cov_mat
			eigenvalues[param][time] = numpy.linalg.eigvals(cov_mat)
			print eigenvalues[param][time]

	

	return mu, covariance_list
