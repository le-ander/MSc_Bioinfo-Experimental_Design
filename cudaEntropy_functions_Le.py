def run_cudasim(m_object, parameters, species):
	modelTraj = []
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Define CUDA filename for cudasim
		cudaCode = m_object.name[mod] + '.cu'
		# Create ODEProblem object
		modelInstance = Lsoda.Lsoda(m_object.times, cudaCode, dt=m_object.dt)
		# Solve ODEs using Lsoda algorithm
		##Different parameters and species matrices for i in nmodels?
		result = modelInstance.run(parameters, species)
		modelTraj.append(result[:,0])
		
	return modelTraj

def remove_na(m_object, modelTraj):
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Create a list of indices of particles that have an NA in their row
		##Why using 7:8 when summing?
		index = [p for p, i in enumerate(isnan(sum(asarray(modelTraj[mod])[:,7:8,:],axis=2))) if i==True]
		# Delete row of 1. results and 2. parameters from the output array for which an index exists
		for i in index:
			delete(modelTraj[mod], (i), axis=0)

	return modelTraj

def add_noise_to_traj(m_object, modelTraj, sigma):
	ftheta = []
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Create array with noise of same size as the trajectory array
		noise = normal(loc=0.0, scale=sigma,size=shape(modelTraj[mod]))
		# Add noise to trajectories and output new 'noisy' trajectories
		traj = array(modelTraj[mod]) + noise
		ftheta.append(traj)

	return ftheta

def get_max_dist(m_object, ftheta):
	maxDistTraj = []
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Calculate the maximum distance for each model and store it in array
		maxDistTraj.append(amax(ftheta[mod]) - amin(ftheta[mod]))

	return maxDistTraj

def get_mutinf_all_param(m_object, ftheta, modelTraj, maxDistTraj, sigma):
	MutInfo1 = []
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Run function to get the mutual information for all parameters inference
		MutInfo1.append(getEntropy1(ftheta[mod],shape(modelTraj[mod])[0],sigma,array(modelTraj[mod]),maxDistTraj[mod]))
	
	return MutInfo1