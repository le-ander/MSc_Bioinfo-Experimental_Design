    
def main():
	
    diagnostic=True
    pickling=True
    file_exist=False
    plot=True
    plotTimeSeries=True
    simulate=False
    design=False
    full=False
    usesbml=True
    seed = None
    timing = False
    fname = "_results_"
    custom_kernel = False
    custom_distance = False
    use_cuda = False
    use_c = False
    full_debug = False

    for i in range(1,len(sys.argv)):

        if sys.argv[i].startswith('--'):
            option = sys.argv[i][2:]
          
            if option == 'help':
                printOptions()
                sys.exit()
            elif option == 'diagnostic': diagnostic=False
            elif option == 'save': pickling=False
            elif option == 'timeseries': plotTimeSeries=False
            elif option == 'plotdata': plot=False
            elif option == 'simulate': simulate=True
            elif option == 'design': design=True
            elif option == 'debug': full_debug=True
            elif option == 'fulloutput': full=True
	    elif option == 'localcode' : usesbml = False 
            elif option[0:8] == 'setseed=' : seed = int( option[8:] )
            elif option[0:10] == 'outfolder=' : fname = option[10:]
            elif option[0:9] == 'cudacode=' : app_file = option[9:]
	    elif option == 'timing' : timing = True 
	    elif option == 'custk' : custom_kernel = True
	    elif option == 'custd' : custom_distance = True
            elif option == 'cuda' : use_cuda = True
            elif option == 'c++' : use_c = True
	    elif option == 'infile': 
                input_file=sys.argv[i+1]
                file_exist=True
	    elif not(sys.argv[i-1][2:] == 'infile'): 
                print "\nunknown option "+sys.argv[i]
                printOptions()
                sys.exit()
            
        
        elif sys.argv[i].startswith('-'):
            option = sys.argv[i][1:]
            if option == 'h':
                printOptions()
                sys.exit()
            elif option == 'd': diagnostic=False
            elif option == 's': pickling=False
            elif option == 't': plotTimeSeries=False
            elif option == 'p': plot=False
            elif option == 'S': simulate=True
            elif option == 'D': design=True
            elif option == 'db': full_debug=True
            elif option == 'f': full=True
            elif option == 'cu': use_cuda = True
	    elif option == 'lc' : usesbml = False
            elif option[0:3] == 'sd=' : seed = int( option[3:] )
            elif option[0:3] == 'of=' : fname = option[3:]
            elif option == 'tm' : timing = True
            elif option == 'i': 
                input_file=sys.argv[i+1]
                file_exist=True
            elif not(sys.argv[i-1][2:] == 'i'): 
                print "\nunknown option "+sys.argv[i]
                printOptions()
                sys.exit()
        elif not((sys.argv[i-1][2:] == 'infile') or (sys.argv[i-1][1:] == 'i')):
            print "\nunknown expression \""+sys.argv[i]+"\""
            printOptions()
            sys.exit()

    if file_exist == False:
        print "\nNo input_file is given!\nUse: \n\t-i 'inputfile' \nor: \n\t--infile 'inputfile' \n"
        sys.exit()

    # python, C++ or CUDA 
    if use_cuda == True and use_c == True:
        print "specified both c++ and CUDA "
        sys.exit()
    if design == True and simulate==True:
        print "specified both design and simulate "
        sys.exit()

    # parse the input file
    mode = 0
    if simulate == True: mode = 1
    if design == True: mode = 2

    info_new = parse_infoEnt.algorithm_info(input_file, mode) 

    info_new.print_info()

    # Check that we have libSBML if it is requested
    if usesbml == True:
        try: import libsbml
        except ImportError:
            print "ABORT: libSBML required for SBML parsing. Please install libSBML"
            sys.exit()

    # Check that we can import scipy if we have ODE models
    o = re.compile('ODE')
    for m in range( info_new.nmodels ):
        if o.search(info_new.type[m]):
            try: from scipy.integrate.odepack import odeint
            except ImportError:
                print "ABORT: scipy required for ODE modelling. Please install scipy"
                sys.exit()
            break

    # Check that we havex.y cuda-sim installed
    if use_cuda == True:
        try: import cudasim
        except ImportError:
            print "ABORT: cudasim required for running on CUDA GPUs. Please install cuda-sim"
            sys.exit()
    
    # set the random seeds
    if seed != None:
	print "#### Seeding random number generator : ", seed
        numpy.random.seed(seed)

    # Check the information is correct for simulation
    modelCorrect = False
    if usesbml == True :
        integrationType = []
        
        if use_cuda == True:
            # CUDA
            for i in range(len(info_new.type)):
                integrationType.append(info_new.type[i]+' CUDA')
        elif use_c == True :
            # C
            for i in range(len(info_new.type)):
                if info_new.type[i] == "SDE":
                    info_new.type[i] = "EulerSDE"
                integrationType.append(info_new.type[i]+' C')
        else:
            # python 
            for i in range(len(info_new.type)):
                integrationType.append(info_new.type[i]+' Python')
            

        ParseAndWrite.ParseAndWrite(info_new.source,integrationType,info_new.name,inputPath="",outputPath="",method=None)


    print("Parsing done")
    print("Starting Simulation...")
    modelTraj = []


    sampleFromPost = False
    referenceModel = True

    #loop over models

    try:
        os.mkdir("acceptedParticles")
    except:
	print "\nThe folder acceptedParticles already exists!\n"
        sys.exit()

    try:
        os.mkdir("rejectedParticles")
    except:
	print "\nThe folder rejectedParticles already exists!\n"
        sys.exit()


    saveIndex = []
    saveResult = []

    for mod in range(info_new.nmodels):

    	print "Model:",mod+1

    	modelTraj.append([])
	accepted = 0
	while(accepted<info_new.particles):
	    if(mod==0): parameters = zeros( [info_new.particles,len(info_new.prior[mod])] )	
    	    species = zeros([info_new.particles,info_new.nspecies[mod]])

            for i in range(info_new.particles):
            	for j in range(info_new.nspecies[mod]):
                    species[i,j] = info_new.x0prior[mod][j][1]

       	    if(mod==0 and sampleFromPost==False):
		for j in range(len(info_new.prior[mod])):
                    if(info_new.prior[mod][j][0]==0):
			for i in range(info_new.particles):
                            parameters[i,j] = info_new.prior[mod][j][1]
                    if(info_new.prior[mod][j][0]==2):
			parameters[:,j] = uniform(low=info_new.prior[mod][j][1], high=info_new.prior[mod][j][2],size=(1,1,info_new.particles))[0][0]
        	    if(info_new.prior[mod][j][0]==1):
			parameters[:,j] = normal(loc=info_new.prior[mod][j][1], scale=info_new.prior[mod][j][2],size=(1,1,info_new.particles))[0][0]
            if(mod==0 and sampleFromPost==True):
		infileName = "data"+`mod+1`+".txt"
		in_file=open(infileName, "r")
	        matrix=[]
	        param=[]
		counter=0
		for in_line in in_file.readlines():
		    in_line=in_line.rstrip()
		    matrix.append([])
		    param.append([])
    		    matrix[counter]=in_line.split(" ")
		    param[counter] = map(float, matrix[counter])
    		    counter=counter+1
		in_file.close

		infileName = "data_Weights"+`mod+1`+".txt"
		in_file=open(infileName, "r")
	        matrix=[]
	        weights=[]
		counter2=0
		for in_line in in_file.readlines():
		    in_line=in_line.rstrip()
		    matrix.append([])
		    weights.append([])
    		    matrix[counter2]=in_line.split(" ")
		    weights[counter2] = map(float, matrix[counter2])
    		    counter2=counter2+1
		in_file.close

		if not(counter == counter2):
		    print ""
		    print "Please provide equal number of particles and weights in model ", mod+1, "!"
		    sys.exit()
		else:
		    parameters = zeros( [info_new.particles,len(param[0])] )
		    		    
		    index = getWeightedSample(weights)

		    for i in range(info_new.particles):
		        index = getWeightedSample(weights)
		    	for j in range(len(param[0])):
			    parameters[i,j] = param[index][j]
 
    
		
	    # simulate model mod
            cudaCode = info_new.name[mod] + '.cu'
	    modelInstance = Lsoda.Lsoda(info_new.times, cudaCode, dt=info_new.dt)
            result0 = modelInstance.run(parameters, species)	
            
############ASK ABOUT########################################################
	    n = shape(result0)[2]
       	    result = result0[:,:,0:n,:]
	    print shape(result)


	    # merge result and parameters
	    
	    # check for NA
	    
	    #for i in enumerate(isnan(sum(result[:,0,7:8,:],axis=2))):
	#	print i
#	    print isnan(sum(result[:,0,7:8,:],axis=2))
	    index = [p for p, i in enumerate(isnan(sum(result[:,0,7:8,:],axis=2))) if i==False]
	    print len(index)
	    saveIndex.append(index)


            saveResult.append(result)
#	    accepted = len(parametersKeepFinal)
#	    print "accepted particles: ",accepted
	    accepted = 10000000

    for mod in range(info_new.nmodels):

    	parametersKeepFinal = []
	parametersRemoveFinal = []
	resultKeepFinal = []
	resultRemoveFinal = []

        if(len(saveIndex[0])>0):
    	    for i in (set(saveIndex[0])):
       	        resultKeepFinal.append(saveResult[mod][i][0])
	    	parametersKeepFinal.append(parameters[i])

	#use the first N particles only
	del resultKeepFinal[info_new.particles:]
	del parametersKeepFinal[info_new.particles:]

	# create list that contains parameters theta and variables x
	modelTraj[mod].append(parametersKeepFinal)
	modelTraj[mod].append(resultKeepFinal)
	
	print "shape important: ", shape(modelTraj[mod][1])
	
	print "model ",mod,": simulation done!"
	# write out sampled parameters and trajectories for accepted and rejected particles

	fileName =  "acceptedParticles/model"+`mod+1`+"_trajectories_accepted.txt"
	#print_results(resultKeepFinal,fileName,info_new.times)

	fileName =  "acceptedParticles/model"+`mod+1`+"_parameters_accepted.txt"
 	#print_parameters(parametersKeepFinal,fileName)

	fileName =  "rejectedParticles/model"+`mod+1`+"_parameters_rejected.txt"
 	#print_parameters(parametersRemoveFinal,fileName)

##############################################################################

    sigma = 5.0

    ftheta = []
    maxDistTraj = []
    for mod in range(info_new.nmodels):
    	print shape(modelTraj[mod][1])
    	trajTemp = array(modelTraj[mod][1])[:,:,0:6]
	print "shape traj:", shape(trajTemp)
	noise = normal(loc=0.0, scale=sigma,size=(info_new.particles,len(info_new.times),shape(trajTemp)[2]))
	temp = trajTemp[:,:,:] + noise
	maxDistTraj.append(amax(temp) - amin(temp))
	print "maxDistTraj:", maxDistTraj
        ftheta.append(temp)
	    


    print("Simulation done")
    print "------------------------ "
    print " "

    MutInfo1 = []
    for mod in range(info_new.nmodels):
 	N = info_new.particles

     	MutInfo1.append(getEntropy1(ftheta[mod],N,sigma,array(modelTraj[mod][1])[:,:,0:6],maxDistTraj[mod]))
     	print "I(theta,X",mod+1,") = ", MutInfo1[mod]

    # compute I(x,y) for reference model

    if referenceModel == True:
        MutInfo2 = []
        print("Mutual information calculation 2... ")
	MutInfo2.append("NA")
	for mod in range(1,info_new.nmodels):
	    N = info_new.particles
   	    MutInfo2.append(getEntropy2(ftheta[0],ftheta[mod],N,sigma,modelTraj[0][1],modelTraj[mod][1]))
   	    print "I(X(reference model),X(model",mod+1,") = ", MutInfo2[mod]



main()