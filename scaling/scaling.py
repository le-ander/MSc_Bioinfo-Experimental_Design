import math

def max_dist(data, theta):
    maxDistTraj = [math.fabs(amax(data) - amin(theta)),math.fabs(amax(theta) - amin(data))]
    return max(maxDistTraj)
    
def scaling(sigma, data, maxDistTraj):

	preci = pow(10,-34)
	FmaxDistTraj = 1.0*exp(-(maxDistTraj*maxDistTraj)/(2.0*sigma*sigma))
	
	if(FmaxDistTraj<preci):
		a = pow(1.79*pow(10,300),1.0/( data.shape[1]*data.shape[2]))
	else:
		a = pow(preci,1.0/(data.shape[1]*data.shape[2]))*1.0/FmaxDistTraj
	
	return a