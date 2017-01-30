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

def add_noise_to_traj(m_object, modelTraj, sigma, N1):
	ftheta = []
	# For each model in turn...
	for mod in range(m_object.nmodels):
		# Create array with noise of same size as the trajectory array (only the first N1 particles)
		noise = normal(loc=0.0, scale=sigma,size=shape(modelTraj[mod][0:N1,:,:]))
		# Add noise to trajectories and output new 'noisy' trajectories
		traj = array(modelTraj[mod][0:N1,:,:]) + noise
		ftheta.append(traj)
	# Return final trajectories for 0:N1 particles
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
	# For each model in turn....
	for mod in range(m_object.nmodels):
		# Run function to get the mutual information for all parameters inference
		MutInfo1.append(getEntropy1(ftheta[mod],sigma,array(modelTraj[mod]),maxDistTraj[mod]))

	return MutInfo1

def getEntropy1(data,sigma,theta,maxDistTraj):

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
				   x1 = x1 +log(a) - (d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])*(d2[idx3d(j,k,l,M,P)]-d1[idx3d(i,k,l,M,P)])/(2.0*sigma*sigma);
			}
	}

	res1[idx2d(i,j,Nj)] = exp(x1);
	}
	""")

	# Prepare data
	N1 = 10
	N2 = 90

	d1 = data.astype(float64)
	d2 = array(theta)[N1:(N1+N2),:,:].astype(float64)

	#print "shape d1:", shape(d1)
	#print "shape d2:", shape(d2)

	# Split data to correct size to run on GPU
	Max = 10.0 # max number of threads on whole gpu

	dist_gpu1 = mod.get_function("distance1")

	# Set up scaling factor to avoid working with too small numbers
	preci = pow(10,-34)
	FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))
	print "FmaxDistTraj:",FmaxDistTraj
	if(FmaxDistTraj<preci):
		a = pow(1.79*pow(10,300),1.0/(d1.shape[1]*d1.shape[2]))
	else:
		a = pow(1.79*pow(10,300),1.0/(d1.shape[1]*d1.shape[2]))
	print "preci:", preci, "a:",a

	# Determine required number of runs for i and j
	numRuns = int(ceil(N1/Max))
	numRuns2 = int(ceil(N2/Max))

	result2 = zeros([N1,numRuns2])

	countsi = 0

	for i in range(numRuns):
		#print "Runs left:", numRuns - i

		countsj = 0

		si = int(Max)
		sj = int(Max)
		s = int(Max)

		if((s*(i+1)) > N1): # If last run with less that max remaining trajectories
			si = int(N1 - Max*i) # Set si to remaining number of particels

		for j in range(numRuns2):

			if((s*(j+1)) > N2): # If last run with less that max remaining trajectories
				sj = int(N2 - Max*j) # Set sj to remaining number of particels

			data1 = d1[(i*int(Max)):(i*int(Max)+si),:,:] # d1 subunit for this run (same vector 9 times)
			data2 = d2[(j*int(Max)):(j*int(Max)+sj),:,:] # d2 subunit for this run (9 different verctors)
			#print shape(data1), shape(data2)

			Ni = data1.shape[0] # Number of particels in data1 (<= Max)
			Nj = data2.shape[0] # Number of particels in data2 (<= Max)

			M = data1.shape[1] # number of timepoints in d1 subunit
			P = data1.shape[2] # number of species in d1 subunit

			res1 = zeros([Ni,Nj]).astype(float64) # results vector [shape(data1)*shape(data2)]

			# Define square root of maximum threads per block
			R = sqrt(driver.Device(0).max_threads_per_block)

			if(Ni<R):
				gi = 1  # grid width  (no of blocks in i direction, i.e. gi * gj gives number of blocks)
				bi = Ni # block width (no of threads in i direction, i.e. bi * bj gives size of each block (max. R*R))
			else:
				gi = ceil(Ni/R)
				bi = R
			if(Nj<R):
				gj = 1  # grid length
				bj = Nj # block length
			else:
				gj = ceil(Nj/R)
				bj = R

			# Invoke GPU calculations (takes data1 and data2 as input, outputs res1)
			dist_gpu1(int32(Ni),int32(Nj), int32(M), int32(P), float32(sigma), float32(pi), float64(a), driver.In(data1), driver.In(data2),  driver.Out(res1), block=(int(bi),int(bj),1), grid=(int(gi),int(gj)))

			#print "SHAPE RES1", shape(res1)
			for k in range(si):
				result2[(i*int(Max)+k),j] = sum(res1[k,:])
			countsj = countsj+sj

		countsi = countsi+si


	sum1 = 0.0
	counter = 0  # counts number of nan in matrix
	counter2 = 0 # counts number of inf sums in matrix


	for i in range(N1):
		if(isnan(sum(result2[i,:]))): counter=counter+1
		if(isinf(log(sum(result2[i,:])))): counter2=counter2+1
		else:
			sum1 = sum1 - log(sum(result2[i,:])) + log(float(N2)) + M*P*log(a) +  M*P*log(2.0*pi*sigma*sigma)

	Info = sum1/float(N1)

	Info = Info - M*P/2.0*log(2.0*pi*sigma*sigma*exp(1))

	print "counter: ",counter,"counter2: ",counter2

	out = open('results','w')

	print >>out, "counter: ",counter2
	print >>out, "mutual info: ", Info

	out.close()

	return(Info)
