from numpy import *
from numpy.random import *

def getWeightedSample(weights):

    totals = []
    running_total = 0

    for w in weights:
        running_total = running_total + w[0]
        totals.append(running_total)

    rnd = random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i

def THETAS(model_object, sampleGiven = False, sampleFromPost="", weight="", inputpath="", analysisType = 0, N1 = 0, N3 = 0, parameter_i = "", specie_i = "",usesbml = False):
	#create array which holds parameters
	if sampleGiven==False:
		parameters = zeros([model_object.particles,model_object.globalnparameters]) #we might  want to change prior[0] to a globally defined prior in the object

		#obtain Thetas from prior distributions, wich are either constant, uniform, normal or lognormal

		for j in range(len(model_object.prior[0])): # loop through number of parameter
			
			#####Constant prior#####
			if(model_object.prior[0][j][0]==0):  # j paramater index
				parameters[:,j] = model_object.prior[0][j][1]
			
			#####Uniform prior#####
			elif(model_object.prior[0][j][0]==2):   
				parameters[:,j] = uniform(low=model_object.prior[0][j][1], high=model_object.prior[0][j][2], size=(model_object.particles))
			
			#####Normal prior#####
			elif(model_object.prior[0][j][0]==1):       
				parameters[:,j] = normal(loc=model_object.prior[0][j][1], scale=model_object.prior[0][j][2], size=(model_object.particles))

			#####Lognormal prior#####
			elif(model_object.prior[0][j][0]==3):       
				parameters[:,j] = lognormal(mean=model_object.prior[0][j][1], sigma=model_object.prior[0][j][2], size=(model_object.particles))

			####
			else:
				print " Prior distribution not defined for parameters"
				sys.exit()
		
		species = zeros([model_object.particles,model_object.nspecies])  # number of repeats x species in system

		for j in range(len(model_object.x0prior[0])): # loop through number of parameter
				
			#####Constant prior#####
			if(model_object.x0prior[0][j][0]==0):  # j paramater index
				species[:,j] = model_object.x0prior[0][j][1]
			
			#####Uniform prior#####
			elif(model_object.x0prior[0][j][0]==2):   
				species[:,j] = uniform(low=model_object.x0prior[0][j][1], high=model_object.x0prior[0][j][2], size=(model_object.particles))
			
			#####Normal prior#####
			elif(model_object.x0prior[0][j][0]==1):       
				species[:,j] = normal(loc=model_object.x0prior[0][j][1], scale=model_object.x0prior[0][j][2], size=(model_object.particles))

			#####Lognormal prior#####
			elif(model_object.x0prior[0][j][0]==3):       
				species[:,j] = lognormal(mean=model_object.x0prior[0][j][1], sigma=model_object.x0prior[0][j][2], size=(model_object.particles))

			####
			else:
				print " Prior distribution not defined on initial conditions"
				sys.exit()

		if usesbml == True:

			compartments = zeros([model_object.particles,model_object.ncompartments])

			for j in range(len(model_object.compartment[0])): # loop through number of parameter
				
				#####Constant prior#####
				if(model_object.compartment[0][j][0]==0):  # j paramater index
					compartments[:,j] = model_object.compartment[0][j][1]
				
				#####Uniform prior#####
				elif(model_object.compartment[0][j][0]==2):   
					compartments[:,j] = uniform(low=model_object.compartment[0][j][1], high=model_object.compartment[0][j][2], size=(model_object.particles))
				
				#####Normal prior#####
				elif(model_object.compartment[0][j][0]==1):       
					compartments[:,j] = normal(loc=model_object.compartment[0][j][1], scale=model_object.compartment[0][j][2], size=(model_object.particles))

				#####Lognormal prior#####
				elif(model_object.compartment[0][j][0]==3):       
					compartments[:,j] = lognormal(mean=model_object.compartment[0][j][1], sigma=model_object.compartment[0][j][2], size=(model_object.particles))

				####
				else:
					print " Prior distribution not defined on compartments"
					sys.exit()

	elif sampleGiven==True:
		#obtain Thetas from posterior sample and associated weights
		
		######Reading in sample from posterior#####
		infileName = inputpath+sampleFromPost
		in_file=open(infileName, "r")
		param=[]
		counter=0
		for in_line in in_file.readlines():
			in_line=in_line.rstrip()
			param.append([])
			param[counter]=in_line.split(" ")
			param[counter] = map(float, param[counter])
			counter=counter+1
		in_file.close

		######Reading in weigths associated to sample from posterior#####
		infileName = inputpath+weight
		in_file=open(infileName, "r")
		weights=[]
		counter2=0
		for in_line in in_file.readlines():
			in_line=in_line.rstrip()
			weights.append([])
			weights[counter2]=in_line.split(" ")
			weights[counter2] = map(float, weights[counter2])
			counter2=counter2+1
		in_file.close

		if usesbml == False:
			####Obtain Theta from posterior samples through weigths####
			if(counter==counter2):#and len(model_object.nparameters[0])==len(param[0])): ### model object needs to include nparameters information
				parameters = zeros( [model_object.particles,model_object.globalnparameters] )
				species = zeros([model_object.particles,model_object.nspecies])	
				for i in range(model_object.particles): #repeats
					index = getWeightedSample(weights)  #manually defined function
					print index
					parameters[i,:] = param[index][:model_object.globalnparameters] #index indefies list which is used to assign parameter value.  j corresponds to different parameters defines column 
					species[i,:] = param[index][-model_object.nspecies:]
			else:
				print "Please provide equal number of particles and weights in model!"
				sys.exit()
		elif usesbml == True:
			####Obtain Theta from posterior samples through weigths####
			if(counter==counter2):#and len(model_object.nparameters[0])==len(param[0])): ### model object needs to include nparameters information
				compartments = zeros([model_object.particles,model_object.ncompartments])
				parameters = zeros( [model_object.particles,model_object.globalnparameters] )
				species = zeros([model_object.particles,model_object.nspecies])	
				for i in range(model_object.particles): #repeats
					index = getWeightedSample(weights)  #manually defined function
					print index
					compartments[i,:] = param[index][:model_object.globalnparameters]
					parameters[i,:] = param[index][model_object.globalnparameters:model_object.globalnparameters] #index indefies list which is used to assign parameter value.  j corresponds to different parameters defines column 
					species[i,:] = param[index][-model_object.nspecies:]
			else:
				print "Please provide equal number of particles and weights in model!"
				sys.exit()


	if analysisType == 1:
		paramsN3 = parameters[(model_object.particles-N3):,:]
		speciesN3 = species[(model_object.particles-N3):,:]
		params_final = concatenate((paramsN3,)*N1,axis=0)
		species_final = concatenate((speciesN3,)*N1,axis=0)
		
		for j in range(0,N1):
			for i in parameter_i:
				params_final[range((j*N3),((j+1)*N3)),i] = parameters[j,i]

		for j in range(0,N1):
			for i in specie_i:
				species_final[range((j*N3),((j+1)*N3)),i] = species[j,i]

		parameters = concatenate((parameters[range(model_object.particles-N3),:],params_final),axis=0)
		species = concatenate((species[range(model_object.particles-N3),:],species_final),axis=0)

		if usesbml == True:
			compsN3 = compartments[(model_object.particles-N3):,:]
			comp_final = concatenate((compsN3,)*N1,axis=0)
			for j in range(0,N1):
				for i in comps_i:
					comp_final[range((j*N3),((j+1)*N3)),i] = compartments[j,i]

			compartments = concatenate((compartments[range(model_object.particles-N3),:],comp_final),axis=0)

	print parameters
	print ""
	print species

	return parameters, species

def SPECIES(model_object):
	#creates an array with species
	species = zeros([model_object.particles,model_object.nspecies])  # number of repeats x species in system
	'''
	for i in range(model_object.particles): 
		for j in range(model_object.nspecies[0]): #species in repeat
			species[i,j] = model_object.x0prior[0][j][1] #initial state of species added to array
	'''
	for j in range(len(model_object.x0prior[0])): # loop through number of parameter
			
			#####Constant prior#####
			if(model_object.x0prior[0][j][0]==0):  # j paramater index
				species[:,j] = model_object.x0prior[0][j][1]
			

			#####Uniform prior#####
			elif(model_object.x0prior[0][j][0]==2):   
				species[:,j] = uniform(low=model_object.x0prior[0][j][1], high=model_object.x0prior[0][j][2], size=(model_object.particles))
			

			#####Normal prior#####
			elif(model_object.x0prior[0][j][0]==1):       
				species[:,j] = normal(loc=model_object.x0prior[0][j][1], scale=model_object.x0prior[0][j][2], size=(model_object.particles))


			#####Lognormal prior#####
			elif(model_object.x0prior[0][j][0]==3):       
				species[:,j] = lognormal(mean=model_object.x0prior[mod][j][1], sigma=model_object.x0prior[mod][j][2], size=(model_object.particles))

			####
			else:
				print " Prior distribution not defined on initial conditions"
				sys.exit()

	return species

def ThetasGivenI(model_object, parameters, species, parameter_i, specie_i, N1, N3):
	
	params_to_sample = [i for i in range(parameters.shape[1])]
	params_to_sample = [i for i in params_to_sample if i not in parameter_i]
	ic_to_sample = [i for i in range(species.shape[1])]
	ic_to_sample = [i for i in ic_to_sample if i not in specie_i]

	paramsN3 = zeros([N3,parameters.shape[1]])

	for j in params_to_sample:
		#####Constant prior#####
		if(model_object.prior[0][j][0]==0):  # j paramater index
			paramsN3[:,j] = model_object.prior[0][j][1]
		#####Uniform prior#####
		elif(model_object.prior[0][j][0]==2):   
			paramsN3[:,j] = uniform(low=model_object.prior[0][j][1], high=model_object.prior[0][j][2], size=(N3))
		#####Normal prior#####
		elif(model_object.prior[0][j][0]==1):       
			paramsN3[:,j] = normal(loc=model_object.prior[0][j][1], scale=model_object.prior[0][j][2], size=(N3))
		#####Lognormal prior#####
		elif(model_object.prior[0][j][0]==3):       
			paramsN3[:,j] = lognormal(mean=model_object.prior[mod][j][1], sigma=model_object.prior[mod][j][2], size=(N3))

	params_final=concatenate((paramsN3,)*N1,axis=0)

	for j in range(0,N1):
		for i in parameter_i:
			params_final[range((j*N3),((j+1)*N3)),i] = parameters[j,i]

	speciesN3 = zeros([N3,species.shape[1]])

	for j in ic_to_sample:
		#####Constant prior#####
		if(model_object.x0prior[0][j][0]==0):  # j paramater index
			speciesN3[:,j] = model_object.x0prior[0][j][1]
		#####Uniform prior#####
		elif(model_object.x0prior[0][j][0]==2):   
			speciesN3[:,j] = uniform(low=model_object.x0prior[0][j][1], high=model_object.x0prior[0][j][2], size=(N3))
		#####Normal prior#####
		elif(model_object.x0prior[0][j][0]==1):       
			speciesN3[:,j] = normal(loc=model_object.x0prior[0][j][1], scale=model_object.x0prior[0][j][2], size=(N3))
		#####Lognormal prior#####
		elif(model_object.x0prior[0][j][0]==3):       
			speciesN3[:,j] = lognormal(mean=model_object.x0prior[mod][j][1], sigma=model_object.x0prior[mod][j][2], size=(N3))

	species_final=concatenate((speciesN3,)*N1,axis=0)

	for j in range(0,N1):
		for i in specie_i:
			species_final[range((j*N3),((j+1)*N3)),i] = species[j,i]

	return params_final, species_final



	