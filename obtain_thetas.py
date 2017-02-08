def THETAS(model_object, sampleFromPost):
	#create array which holds parameters
	


	if sampleFromPost==False:
		parameters = zeros( [model_object.particles,len(model_object.globalnparameters)] ) #we might  want to change prior[0] to a globally defined prior in the object

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
				parameters[:,j] = lognormal(mean=model_object.prior[mod][j][1], sigma=model_object.prior[mod][j][2], size=(model_object.particles))

			####
			else:
				print " Prior distribution not defined "
				sys.exit()






	elif sampleFromPost==True:
		#obtain Thetas from posterior sample and associated weights
		
		######Reading in sample from posterior#####
		infileName = "../input/sample/post_sample_data.txt"
		in_file=open(infileName, "r")
		param=[]
		counter=0
		for in_line in in_file.readlines():
			in_line=in_line.rstrip()
			param.append([])
			matrix[counter]=in_line.split(" ")
			param[counter] = map(float, param[counter])
			counter=counter+1
		in_file.close


		######Reading in weigths associated to sample from posterior#####
		infileName = "../input/sample/post_sample_data_Weights.txt"
		in_file=open(infileName, "r")
		weights=[]
		counter2=0
		for in_line in in_file.readlines():
			in_line=in_line.rstrip()
			weights.append([])
			matrix[counter2]=in_line.split(" ")
			weights[counter2] = map(float, weights[counter2])
			counter2=counter2+1
		in_file.close



		####Obtain Theta from posterior samples through weigths + compartments####
		if(counter==counter2) # and len(model_object.nparameters[0])==len(param[0])): ### model object needs to include nparameters information
			parameters = zeros( [model_object.particles,len(param[0])+model_object.globalnparameters] )
			
			####Fill in compartment parameters
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
					parameters[:,j] = lognormal(mean=model_object.prior[mod][j][1], sigma=model_object.prior[mod][j][2], size=(model_object.particles))

				####
				else:
					print " Prior distribution not defined "
					sys.exit()


			####Fill in the rest through sample from posterior					
			for i in range(model_object.particles): #repeats
				index = getWeightedSample(weights)  #manually defined function
				parameters[i,model_object.globalnparameters:] = param[index] #index indefies list which is used to assign parameter value.  j corresponds to different parameters defines column 
		
		else:
			print "Please provide equal number of particles and weights in model!"
			sys.exit()

	
	return parameters




def SPECIES(model_object):
	#creates an array with species
	species = zeros([model_object.particles,model_object.nspecies[0]])  # number of repeats x species in system

	for i in range(model_object.particles): 
		for j in range(model_object.nspecies[0]): #species in repeat
			species[i,j] = model_object.x0prior[0][j][1] #initial state of species added to array


	return species
