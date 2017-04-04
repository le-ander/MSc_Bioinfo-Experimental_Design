# Algorithm information

import re, sys, numpy, copy, math
from numpy.random import *
from xml.dom import minidom



#### Regular expression for implemented priors or posterior placeholder#############################
re_prior_const=re.compile('constant')
re_prior_uni=re.compile('uniform')
re_prior_normal=re.compile('normal')
re_prior_logn=re.compile('lognormal')
re_prior_posterior=re.compile('posterior')


#### Regular expression for implemented models (ODE/SDE)#############################
re_ODE=re.compile('ODE')
re_SDE=re.compile('SDE')


#### Regular Expression for True/None ##############################################################
re_true=re.compile('True')
re_none=re.compile('None')






#### Function to obtain weighted sample from posterior sample (taken from ABC Sys Bio package)######
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




#### Function to obtain a single value from given node and tag in input xml file####################
def parse_required_single_value( node, tagname, message, cast ):
	try:
		data = node.getElementsByTagName(tagname)[0].firstChild.data
	except:
		print message
		sys.exit()

	ret = 0
	try:
		ret = cast( data )
	except:
		print message
		sys.exit()

	return(ret)




#### Function to obtain a vector of value from given node and tag in input xml file################
def parse_required_vector_value(self, node, tagname, message, cast ):
	try:
		data = node.getElementsByTagName(tagname)[0].firstChild.data
	except:
		print message
		sys.exit()

	tmp = str( data ).split()
	ret = []
	try:
		ret = [ cast(i) for i in tmp ]
	except:
		print message
		sys.exit()

	if len(ret) == 0:
		print message
		sys.exit()

	return(ret)


def type_parser (type, filename):
	
	if re_ODE.match(type):
		return 0
	elif re_SDE.match(type):
		print "\n\nERROR: Type of the model in " ,filename,  self.name[self.nmodels-1], " is not supported!\n\n"
		sys.exit()

	else:
		print "\n\nERROR: Type of the model ", self.name[self.nmodels-1], " is not supported in " +filename+ "!\n\n"
		sys.exit()


#### Function to identify prior the user given prior and obtain its values#########################
#####
def process_prior(self, tmp ,filename):
	prior_tmp = [0,0,0]

	if re_prior_const.match( tmp[0] ):
		prior_tmp[0] = 0
		try:
			prior_tmp[1] = float( tmp[1] )
		except:
			print "\n\nERROR: Value of the prior for experiment ", self.name[self.nmodels-1], "has the wrong format in" +filename+ "!\n\n"
			sys.exit()

	elif re_prior_normal.match( tmp[0] ):
		prior_tmp[0] = 1
		try:
			prior_tmp[1] = float( tmp[1] )
			prior_tmp[2] = float( tmp[2] )
		except:
			print "\n\nERROR: Value of the prior for experiment ", self.name[self.nmodels-1], "has the wrong format in" +filename+ "!\n\n"
			sys.exit()

	elif re_prior_uni.match( tmp[0] ):
		prior_tmp[0] = 2
		try:
			prior_tmp[1] = float( tmp[1] )
			prior_tmp[2] = float( tmp[2] )
		except:
			print "\n\nERROR: Value of the prior for experiment ", self.name[self.nmodels-1], "has the wrong format in" +filename+ "!\n\n"
			sys.exit()

	elif re_prior_logn.match( tmp[0] ):
		prior_tmp[0] = 3
		try:
			prior_tmp[1] = float( tmp[1] )
			prior_tmp[2] = float( tmp[2] )
		except:
			print "\n\nERROR: Value of the prior for experiment ", self.name[self.nmodels-1], "has the wrong format in" +filename+ "!\n\n"
			sys.exit()

	elif re_prior_posterior.match( tmp[0] ):
		prior_tmp[0] = 4

	else:
		print "\n\nERROR: Supplied parameter prior  in", filename , tmp[0], " is unsupported!\n\n"
		sys.exit()

	return prior_tmp




#### Function to parse a string into an integer#####################################################
def parseint(str):
	try:
		return int(str)
	except ValueError:
		return str




#### Function to parse a string into an integer and offset it by -1 to obtain the right index########
def parseint_index(str):
	try:
		out=int(str)-1
		return out
	except ValueError:
		return str




#### Function to parse fitting information about species###########################################
#####
def parse_fitting_information( mod_str, node, species_number ):
	fitref = node.getElementsByTagName(mod_str)[0]
	tmp = str( fitref.firstChild.data ).split()
	ret1 = []

	if (tmp[0]=="All") and (len(tmp)==1):
		ret_temp = []

		for i in range(species_number):
			ret_temp.append([i])

		ret1.extend(ret_temp)

	else:
		for i in tmp:
			ttmp = re.sub('species','', i )
			ttmp = re.sub(r'\+', ' + ', ttmp)
			ttmp = re.sub(r'\-', ' - ', ttmp)
			ttmp = ttmp.split(" ")

			ttmp_int = [ parseint_index(y) for y in ttmp]

			ret1.append(ttmp_int)

	return( ret1 )




#### Function to parse fitting information about other model parameter###############################
#####
def parse_fitting_information_parameters(mod_str, node, item, parameter_number):
	fitref = node.getElementsByTagName(mod_str)[0]
	tmp = str( fitref.firstChild.data ).split()
	ret1 = []

	if (tmp[0]=="All") and (len(tmp)==1):

		for i in range(0,parameter_number):
			ret1.append(i)

	elif(tmp[0]=="None") and (len(tmp)==1):
		ret1=[]

	else:
		for i in tmp:
			ttmp = re.sub(item,'', i )
			ret1.append(int(ttmp)-1)

	return( ret1 )



class algorithm_info:
	"""
	A class to parse the user-provided input file, return all information required to run cuda-sim and hold associated data.

	"""

	def __init__(self, filename, combination_list):
		xmldoc = minidom.parse(filename)
		### mode is 0  inference, 1 simulate, 2 design
		#self.mode = mode
		
		### initialises attributes of the object
		
		#### Global model/experiment attributes
		self.modelnumber = 0
		self.particles = 0
		self.beta = 0
		self.dt = 0
		self.times = []
		self.ntimes = 0
		self.nspecies_all=0
		self.nparameters_all = 0
		self.sigma= 0
		self.ncompparams = []

		#### SDE specific attributes
		#self.beta = 
		#self.cuda_SDE =

		#### Posterior sample attributes
		self.sampleFromPost = False
		self.post_sample_file = ""
		self.post_weight_file = ""
		
		#### Attribute indicating if initial conditions are defined by priors
		self.initialprior = False

		#### Fit attributes
		self.comp_fit= []
		self.init_fit= []
		self.param_fit = []

		#### Sample attributes
		self.N1sample = 0
		self.N2sample = 0
		self.N3sample = 0
		self.N4sample = 0

		#### Attribute holding combination of initial condtion, paramter changes which define models/experiments
		self.combination = combination_list

		#### Model/Experiment specific attributes
		self.nmodels = 0
		self.nparameters = []
		self.nspecies = []
		self.name = []
		self.cuda = []
		self.source = []
		self.type = ""
		self.prior = []
		self.x0prior = []
		self.compprior = []
		self.fitSpecies = []


		##################################################
		## Required arguments

		### get number of models
		self.modelnumber = parse_required_single_value( xmldoc, "experimentnumber", "\n\nERROR: Please provide an integer value for <experimentnumber> in"+ filename+ "!", int )

		#### gets type of model (ODE/SDE)#####################
		self.type=type_parser (parse_required_single_value( xmldoc, "type", "\n\nERROR: Type of the model in"+ filename+ "is not supported!", str ).strip(),filename) ### convert type into integer and check for correct definition test


		### get number of samples
		self.particles = parse_required_single_value( xmldoc, "particles", "\n\nERROR: Please provide an integer value for <particles> in"+ filename+ "!", int )


		### get dt
		self.dt = parse_required_single_value( xmldoc, "dt", "\n\nERROR: Please provide an float value for <dt> in"+ filename+ "!", float )


		### indetifies data node in input xml file
		dataref = xmldoc.getElementsByTagName('data')[0]


		### get timepoints
		self.times = parse_required_vector_value(self, dataref, "times", "\n\nERROR: Please provide a white space separated list of values for <data><times> in"+ filename+ "!" , float )
		self.ntimes = len(self.times)
		#### check if times are in ascending order

		### get global number of parameters
		self.nparameters_all = parse_required_single_value(dataref, "nparameters_all", "\n\nERROR: Please provide an integer value for <data><nparameters_all> in"+ filename+ "!", int)


		### get sigma
		self.sigma = parse_required_single_value(dataref, "sigma", "\n\nERROR: Please provide a float value for <data><sigma> in"+ filename+ "!", float)


		### get information about sample from posterior
		if parse_required_single_value( dataref, "samplefrompost", "\n\nERROR: Please provide a boolean value for <samplefrompost> in"+ filename+ "!", str ).strip()=="True":
			self.sampleFromPost = True
			self.post_sample_file = parse_required_single_value( dataref, "samplefrompost_file", "\n\nERROR: Please provide a file name for <samplefrompost_file> in"+ filename+ "!", str ).strip()
			self.post_weight_file = parse_required_single_value( dataref, "samplefrompost_weights", "\n\nERROR: Please provide a file name for <samplefrompost_weights> in"+ filename+ "!", str ).strip()
		else:
			self.sampleFromPost = False

		if parse_required_single_value( dataref, "initialprior", "\n\nERROR: Please provide a boolean value for <initialprior> in"+ filename+ "!", str ).strip()=="True":
			self.initialprior = True
		else:
			self.initialprior = False


		#### get sizes of N1, N2, N3 and N4 samples
		nsampleref = xmldoc.getElementsByTagName('nsamples')[0]

		self.N1sample = parse_required_single_value( dataref, "N1", "\n\nERROR: Please provide an integer value for <nsamples><N1> in"+ filename+ "!", int )
		self.N2sample = parse_required_single_value( dataref, "N2", "\n\nERROR: Please provide an integer value for <nsamples><N2> in"+ filename+ "!", int )
		self.N3sample = parse_required_single_value( dataref, "N3", "\n\nERROR: Please provide an integer value for <nsamples><N3> in"+ filename+ "!", int )
		self.N4sample = parse_required_single_value( dataref, "N4", "\n\nERROR: Please provide an integer value for <nsamples><N4> in"+ filename+ "!", int )

		if (self.N1sample+self.N2sample+self.N3sample+self.N4sample)>self.particles:
			print "\n\nERROR: The sum of N1, N2, N3 and N4 is bigger than given particle number  in"+ filename+ "!"
			sys.exit()


		### get model attributes
		modelref = xmldoc.getElementsByTagName('experiments')[0]
		for m in modelref.childNodes:
			if m.nodeType == m.ELEMENT_NODE:
				self.nmodels += 1
				self.prior.append([])
				self.x0prior.append([])
				self.compprior.append([])

				#### gets name of experiment##################
				try:
					self.name.append( str(m.getElementsByTagName('name')[0].firstChild.data).strip() )
				except:
					print "\n\nERROR: Please provide a string value for <name> for experiment ", self.nmodels, " in ", filename, "!"
					sys.exit()
				
				#### gets name of associated SBML file###############
				try:
					self.source.append( str(m.getElementsByTagName('source')[0].firstChild.data).strip() )
				except:
					print "\n\nERROR: Please provide an string value for <source> for experiment ", self.nmodels, " in ", filename, "!"
					sys.exit()
				
				#### gets name of associated CUDA file#############
				try:
					self.cuda.append( str(m.getElementsByTagName('cuda')[0].firstChild.data).strip() )
				except:
					print "\n\nERROR: Please provide an string value for <cuda> for experiment ", self.nmodels, " in ", filename, "!"
					sys.exit()
				

				nparameter = 0
				ncompparam = 0

				#### gets priors of compartment if defined ##########
				try:
					compref = m.getElementsByTagName('compartments')[0]
					for p in compref.childNodes:
						if p.nodeType == p.ELEMENT_NODE:
							ncompparam += 1
							prior_tmp = [0,0,0]
							tmp = str( p.firstChild.data ).split()
							self.compprior[self.nmodels-1].append( process_prior(self, tmp ,filename) )
				except:
					ncompparam = 0

				#### get priors of model parameters##################
				paramref = m.getElementsByTagName('parameters')[0]
				for p in paramref.childNodes:
					if p.nodeType == p.ELEMENT_NODE:
						nparameter += 1
						prior_tmp = [0,0,0]
						tmp = str( p.firstChild.data ).split()
						self.prior[self.nmodels-1].append( process_prior(self, tmp, filename) )


				#### get priors/constants for initial conditions###
				ninit = 0
				initref = m.getElementsByTagName('initial')[0]
				for inn in initref.childNodes:
					if inn.nodeType == inn.ELEMENT_NODE:
						ninit += 1
						prior_tmp = [0,0,0]
						tmp = str( inn.firstChild.data ).split()
						self.x0prior[self.nmodels-1].append( process_prior(self, tmp, filename) )


				#### get measurable species of experiment###############
				try:
					self.fitSpecies.append( parse_fitting_information('measuredspecies', m, ninit )  )
				except:
					print "\n\nERROR: Measurable species are not defined properly with <measuredspecies> ... </measuredspecies> for experiment", self.nmodels, " in ", filename, "!"
					sys.exit()

				#### Error checking for parameter prior, initial condtion prior and measurable species
				if nparameter == 0:
					print "\n\nERROR: No parameters specified in experiment ", self.nmodels, " in ", filename, "!"
					sys.exit()
				if ninit == 0:
					print "\n\nERROR: No initial conditions specified in experiment ", self.nmodels, " in ", filename, "!"
					sys.exit()
				if len(self.fitSpecies[self.nmodels-1])==0:
					print "\n\nERROR: No measurable species specified in experiment ", self.nmodels, " in ", filename, "!"
					sys.exit()

				self.nparameters.append( nparameter )
				self.nspecies.append( ninit )
				self.ncompparams.append( ncompparam )


		if self.nmodels == 0:
			print "\n\nERROR: No experiments specified in "+ filename+ "!"
			sys.exit()

		### checks if all experiments have the same number of species
		if len(set(self.nspecies))==1:
			self.nspecies_all = list(set(self.nspecies))[0]
		else:
			print "\n\nERROR: Models don't have the same number of species in " +filename+ "!"
			sys.exit()

		### checks if all experiments have the same number of compartments
		if len(set(self.ncompparams))==1: #and list(set(self.ncompparams))[0]!=0:
			self.ncompparams_all = list(set(self.ncompparams))[0]
		elif len(set(self.ncompparams))!=1:
			print "\n\nERROR: Experimental models don't have the same number of compartments in" +filename+ "!"
			sys.exit()

		### checks if all experiments have the same number of parameters
		if (len(set(self.nparameters))!=1) or (self.nparameters_all != list(set(self.nparameters))[0]):
			print "\n\nERROR: Experimental models don't have the same number of parameters in" +filename+ "!"
			sys.exit()


		### get paramter fit
		try:
			self.param_fit =( parse_fitting_information_parameters('paramfit', dataref, 'parameter' ,self.nparameters_all )  )
		except:
			print "\n\nERROR: Parameters to be fitted are not defined properly in <paramfit> ... </paramfit> in" +filename+ "!"
			sys.exit()
		###

		### get initial fit
		try:
			self.init_fit =( parse_fitting_information_parameters('initfit', dataref, 'initial' ,self.nspecies_all )  )
		except:
			print "\n\nERROR: Initial conditions to be fitted are not defined properly in <initfit> ... </initfit> in" +filename+ "!"
			sys.exit()
		###

		### get compartment fit
		try:
			self.comp_fit=( parse_fitting_information_parameters('compfit', dataref, 'compartment' ,self.ncompparams_all )  )
		except:
			print "\n\nERROR: Compartments to be fitted are not defined properly in <compfit> ... </compfit> in" +filename+ "!"
			sys.exit()
		###

	# A method which prints out details about the models/experiments defined in the object
	def print_info(self):
		print "\nALGORITHM INFO"
		print "experimentnumber:", self.modelnumber
		print "samples:", self.particles
		print "dt:", self.dt
		print "type:", self.type
		print "parameters:", self.nparameters_all
		print "nspecies:", self.nspecies_all
		print "ncompparams:", self.ncompparams_all
		print "sample from posterior:", bool(self.sampleFromPost)
		print "sample file:", self.post_sample_file
		print "weight file:", self.post_weight_file
		print "parameter fit:", self.param_fit
		print "initial condition fit:", self.init_fit
		print "compartment fit:", self.comp_fit
		print "initial prior:", self.initialprior
		print "sigma:", self.sigma
		print "N1:", self.N1sample
		print "N2:", self.N2sample
		print "N3:", self.N3sample
		print "N4:", self.N4sample




		print "times:", self.times


		print "EXPERIMENTS:", self.nmodels
		for i in range(self.nmodels):
			print "\t", "npar:", self.nparameters[i]
			print "\t", "nspecies:", self.nspecies[i]
			print "\t", "ncompparams:", self.ncompparams[i]
			print "\t", "name:", self.name[i]
			print "\t", "source:", self.source[i]
			print "\t", "measured species:", self.fitSpecies[i]

			print "\t", "init:", self.x0prior[i]
			print "\t", "prior:", self.prior[i]
			print "\t", "comp_prior:", self.compprior[i]
			print "\n"

	# A method to sample from the priors
	##(method gets called by sorting_files)
	##Arguments: 
	##inputpath - if a sample for the prior is given then inputpath is where this file is
	##usesbml - input indicating whether an SBML file is used or not
	def THETAS(self, inputpath="", usesbml = False):
		
		#Compartments are always defined in SBML file and so if local code used but number of compartments > 0 then change usesbml to TRUE in order to sample from compartment prior
		if self.ncompparams_all!=0:
			usesbml = True

		#If statement determines whether to sample from a given sample or use an in-built one
		if self.sampleFromPost==False:

			#Sets up matrix for parameters
			parameters = numpy.zeros([self.particles,self.nparameters_all]) 

			#Loop through number of parameters
			for j in range(len(self.prior[0])): 

				#####Constant prior#####
				if(self.prior[0][j][0]==0):
					parameters[:,j] = self.prior[0][j][1]

				#####Uniform prior#####
				elif(self.prior[0][j][0]==2):
					parameters[:,j] = uniform(low=self.prior[0][j][1], high=self.prior[0][j][2], size=(self.particles))

				#####Normal prior#####
				elif(self.prior[0][j][0]==1):
					parameters[:,j] = normal(loc=self.prior[0][j][1], scale=self.prior[0][j][2], size=(self.particles))

				#####Lognormal prior#####
				elif(self.prior[0][j][0]==3):
					parameters[:,j] = lognormal(mean=self.prior[0][j][1], sigma=self.prior[0][j][2], size=(self.particles))

				####If prior not defined####
				else:
					print " Prior distribution not defined for parameters"
					sys.exit()

			#If the initial conditions have a prior distribution over them then sample
			if self.initialprior == True:

				#Sets up initial conditions array
				species = numpy.zeros([self.particles,self.nspecies_all])  
				
				#Loop through number of species
				for j in range(len(self.x0prior[0])): 

					#####Constant prior#####
					if(self.x0prior[0][j][0]==0):
						species[:,j] = self.x0prior[0][j][1]

					#####Uniform prior#####
					elif(self.x0prior[0][j][0]==2):
						species[:,j] = uniform(low=self.x0prior[0][j][1], high=self.x0prior[0][j][2], size=(self.particles))

					#####Normal prior#####
					elif(self.x0prior[0][j][0]==1):
						species[:,j] = normal(loc=self.x0prior[0][j][1], scale=self.x0prior[0][j][2], size=(self.particles))

					#####Lognormal prior#####
					elif(self.x0prior[0][j][0]==3):
						species[:,j] = lognormal(mean=self.x0prior[0][j][1], sigma=self.x0prior[0][j][2], size=(self.particles))

					####If prior not defined####
					else:
						print " Prior distribution not defined on initial conditions"
						sys.exit()

			#If using constant initial conditions then create species matrix
			else:
				#Finds the set of initial conditions (no repeated elements)
				x0prior_uniq = [self.x0prior[0]]
				for ic in self.x0prior[1:]:
					if ic not in x0prior_uniq:
						x0prior_uniq.append(ic)

				#Sets up the species array
				species = [numpy.zeros([self.particles,self.nspecies_all]) for x in range(len(x0prior_uniq))] 

				#Loop over each initial condition
				for ic in range(len(x0prior_uniq)):
					#Loop through number of species
					for j in range(len(x0prior_uniq[ic])):

						#####Constant prior#####
						if(x0prior_uniq[ic][j][0]==0): 
							species[ic][:,j] = x0prior_uniq[ic][j][1]

						####If prior not defined####
						else:
							print " Prior distribution not defined on initial conditions"
							sys.exit()

			#If usesbml ==True then means that compartments are used
			if usesbml == True:

				#Sets up compartment matrix
				compartments = numpy.zeros([self.particles,self.ncompparams_all])

				#Loop through number of compartments
				for j in range(len(self.compprior[0])): 

					#####Constant prior#####
					if(self.compprior[0][j][0]==0):  # j paramater self.index
						compartments[:,j] = self.compprior[0][j][1]

					#####Uniform prior#####
					elif(self.compprior[0][j][0]==2):
						compartments[:,j] = uniform(low=self.compprior[0][j][1], high=self.compprior[0][j][2], size=(self.particles))

					#####Normal prior#####
					elif(self.compprior[0][j][0]==1):
						compartments[:,j] = normal(loc=self.compprior[0][j][1], scale=self.compprior[0][j][2], size=(self.particles))

					#####Lognormal prior#####
					elif(self.compprior[0][j][0]==3):
						compartments[:,j] = lognormal(mean=self.compprior[0][j][1], sigma=self.compprior[0][j][2], size=(self.particles))

					####If prior not defined#####
					else:
						print " Prior distribution not defined on compartments"
						sys.exit()

		#If a sample from the prior is given then go here
		#obtain Thetas from posterior sample and associated weights
		elif self.sampleFromPost==True:
			######Reading in sample from posterior#####
			infileName = inputpath+"/"+self.post_sample_file
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
			infileName = inputpath+"/"+self.post_weight_file
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

			#If no compartments then fall here
			if usesbml == False:
				####Obtain Theta from posterior samples through weigths####
				#Checks to make sure the number of samples is equal to the number of weights
				if(counter==counter2):
					#Sets up arrays for parameters and initial conditions
					parameters = numpy.zeros( [self.particles,self.nparameters_all] )
					species = numpy.zeros([self.particles,self.nspecies_all])
					#For loop draws samples one at a time
					for i in range(self.particles): 
						#Function that gives sample
						index = getWeightedSample(weights)  
						parameters[i,:] = param[index][:self.nparameters_all] 
						species[i,:] = param[index][-self.nspecies_all:]
				else:
					print "Please provide equal number of particles and weights in model!"
					sys.exit()

			#If compartments used then fall here
			elif usesbml == True:
				####Obtain Theta from posterior samples through weigths####
				#Checks to make sure the number of samples is equal to the number of weights
				if(counter==counter2):
					#Sets up arrays for compartments, parameters and initial conditions
					compartments = numpy.zeros([self.particles,self.ncompparams_all])
					parameters = numpy.zeros( [self.particles,self.nparameters_all] )
					species = numpy.zeros([self.particles,self.nspecies_all])
					#For loop draws samples one at a time
					for i in range(self.particles): 
						#Function that gives sample
						index = getWeightedSample(weights)  
						compartments[i,:] = param[index][:self.ncompparams_all]
						parameters[i,:] = param[index][self.ncompparams_all:self.ncompparams_all+self.nparameters_all] #self.index indefies list which is used to assign parameter value.  j corresponds to different parameters defines column
						species[i,:] = param[index][-self.nspecies_all:]
				else:
					print "Please provide equal number of particles and weights in model!"
					sys.exit()

		#Approach 2 requires a different way to sampling from the prior and is dealt with here
		if self.analysisType == 1:
			#Extracts the N3 sample from the parameter matrix
			paramsN3 = parameters[(self.particles-self.N3sample):,:]

			#Sets up the N1xN3 parameter sample
			params_final = numpy.concatenate((paramsN3,)*self.N1sample,axis=0)

			#For loop changes parameters in N1xN3 sample to be the same as the parameter that is to be estimated in the N1 sample
			for j in range(0,self.N1sample):
				for i in self.param_fit:
					params_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = parameters[j,i]

			#Concatenates the N1 and N2 sample with the N1xN3 sample
			parameters = numpy.concatenate((parameters[range(self.particles-self.N3sample),:],params_final),axis=0)

			#Has to do the same if the initial conditions were drawn from a prior
			if self.initialprior == True:
				#Extracts the N3 sample from the initial values matrix
				speciesN3 = species[(self.particles-self.N3sample):,:]

				#Sets up the N1xN3 initial values sample
				species_final = numpy.concatenate((speciesN3,)*self.N1sample,axis=0)

				#For loop changes initial conditions in N1xN3 sample to be the same as the initial conditions that is to be estimated in the N1 sample
				for j in range(0,self.N1sample):
					for i in self.init_fit:
						species_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = species[j,i]

				#Concatenates the N1 and N2 sample with the N1xN3 sample
				species = numpy.concatenate((species[range(self.particles-self.N3sample),:],species_final),axis=0)
			
			else:
				#Just creates a new intial values matrix that is the same but of size (N1+N2+N1xN3)x(number of species) since initial values are constant
				for ic in range(len(species)):
					species[ic] = numpy.tile(species[ic][0,:],(self.N1sample*self.N3sample+self.N1sample+self.N2sample,1))

			#Has to do the same if compartments are used
			if usesbml == True:
				#Extracts the N3 sample from the initial values matrix
				compsN3 = compartments[(self.particles-self.N3sample):,:]

				#Sets up the N1xN3 initial values sample
				comp_final = numpy.concatenate((compsN3,)*self.N1sample,axis=0)

				#For loop changes compartments in N1xN3 sample to be the same as the compartmetns that is to be estimated in the N1 sample
				for j in range(0,self.N1sample):
					for i in self.comp_fit:
						comp_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = compartments[j,i]

				#Concatenates the N1 and N2 sample with the N1xN3 sample
				compartments = numpy.concatenate((compartments[range(self.particles-self.N3sample),:],comp_final),axis=0)

		#Creates attribute for object with compartment sample
		if usesbml == True:
			self.compsSample = compartments

		#Need to treat each approach differently
		if self.analysisType !=2:
			#For approach 1 and 2 just simply make attributes
			self.parameterSample = parameters
			self.speciesSample = species

		#For approach 3 need split up samples between those used by reference and those used by the experiments
		elif self.analysisType == 2:
			#Extract N4 sample
			self.N4parameterSample = parameters[self.N1sample+self.N2sample+self.N3sample:self.N1sample+self.N2sample+self.N3sample+self.N4sample,:]
			#Set attribute in reference model to N1, N2 and N3 sample
			self.parameterSample = parameters[:self.N1sample+self.N2sample+self.N3sample,:]
			#If prior used for initial conditions then do the same as parameters before
			if self.initialprior == True:
				#Extract the N4 sampel
				self.N4speciesSample = species[self.N1sample+self.N2sample+self.N3sample:self.N1sample+self.N2sample+self.N3sample+self.N4sample,:]
				#Set attribute in reference model to N1, N2 and N3 sample
				self.speciesSample = species[:self.N1sample+self.N2sample+self.N3sample,:]
			else:
				#Otherwise just set the constant initial values to attribute
				self.speciesSample = [x[:self.N1sample+self.N2sample+self.N3sample,:] for x in species]

			#If compartments used
			if usesbml == True:
				#Extract N4 sample
				self.N4compsSample = compartments[self.N1sample+self.N2sample+self.N3sample:self.N1sample+self.N2sample+self.N3sample+self.N4sample,:]
				#Create attribute for reference object
				self.compsSample = compartments[:self.N1sample+self.N2sample+self.N3sample,:]
			#Reset the total number of particles
			self.particles -= self.N4sample
			#Set N4 to 0 as it is not used for the reference model
			self.N4sample = 0


	# A method to obtain the approach
	##(method gets called by sorting_files)
	##Arguments: 
	##analysisType - type of approach
	def getAnalysisType(self,analysisType):
		self.analysisType = analysisType

	# A method to match each cudacode file to a list of initial conditions
	##(method gets called by sorting_files)
	##No arguments
	def getpairingCudaICs(self):
		#Initialises dictionary
		self.pairParamsICS = {}

		#First detect whether using given prior sample
		if self.sampleFromPost == False:
			#If using prior over initials then simply create dictionary with key of cuda code file and values with the prior distribution of the initial conditions
			if self.initialprior == True:
				for Cfile in set(self.cuda):
					self.pairParamsICS[Cfile]  = [self.x0prior[j] for j in [i for i, x in enumerate(self.cuda) if x == Cfile]][0]
			#If using constant initials then keys are cuda code and values is a list of the initial conditions used with the cuda code file
			elif self.initialprior == False:
				for Cfile in set(self.cuda):
					temp = [[l[1] for l in self.x0prior[j]] for j in [i for i, x in enumerate(self.cuda) if x == Cfile]]

					temp_uniq = [temp[0]]
					for ic in temp[1:]:
						if ic not in temp_uniq:
							temp_uniq.append(ic)

					self.pairParamsICS[Cfile] = temp_uniq
		
	# A method to that sorts the output from CUDA-sim
	##(method gets called by run_cudasim)
	##Arguments: 
	##cudaorder - order in which cudacode files are passed in CUDA-sim
	##cudaout - output from cudasim
	def sortCUDASimoutput(self,cudaorder,cudaout):
		#Sets up attribute for output of CUDA-sim
		self.cudaout=[""]*len(self.cuda)

		#If approach is of type 2 then total particles is N1+N2+N1xN3 otherwise just sum of Ni
		if self.analysisType == 1:
			Nparticles = self.N1sample+self.N2sample+self.N1sample*self.N3sample
		else:
			Nparticles = self.particles

		#cuda_NAs holds the number of particles which don't result in NAs from cuda-sim
		##Format of cuda_NAs:
		##For approach 1 and 3: {"Exp_1.cu":[N1,N2,N3,N4],"Exp_2.cu:[N1,N2,N3,N4],...etc"}
		##For approach 2: {"Exp_1.cu":[N1,N2,[N3_1,...,N3_N1]],"Exp_2.cu:[N1,N2,[N3_1,...,N3_N1]],...etc"}
		cuda_NAs = dict((k, []) for k in cudaorder)

		#For loop iterates over distinct cuda codes
		for i, cudafile in enumerate(cudaorder):
			#Index_NA finds the particle indices for which there are NAs
			index_NA = [p for p, e in enumerate(numpy.isnan(numpy.sum(numpy.sum(cudaout[i][:,:,:],axis=2),axis=1))) if e==True]
			
			#Detects whether we have priors over the initial conditions or not and sets pairings_ICs for iteration over in the next part
			if self.initialprior == False:
				pairing_ICs = enumerate(self.pairParamsICS.values()[i])
			else:
				pairing_ICs = enumerate(range(1))

			#Iteration over different initial conditions
			for j, IC in pairing_ICs:
				#Splits up the NAs belong to each specific initial condition
				index_NA_IC = [s for s in index_NA if s < (j+1)*Nparticles  and s >= j*Nparticles]

				#Need to distinguish between approach 2 and the others as NAs handled differently
				if self.analysisType !=1:
					#For approach 1 and 3 simply count the NAs and remove
					N1_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample]
					N2_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample and x >= j*Nparticles + self.N1sample]
					N3_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample+self.N3sample and x >= j*Nparticles + self.N1sample + self.N2sample]
					N4_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample+self.N3sample+self.N4sample and x >= j*Nparticles + self.N1sample + self.N2sample+self.N3sample]
					#Key is the cuda code file string and the values is a list corresponding to the remaining samples after removing NAs
					cuda_NAs[cudafile].append([self.N1sample-len(N1_NA),self.N2sample-len(N2_NA),self.N3sample-len(N3_NA),self.N4sample-len(N4_NA)])
				
				#Approach 2 needs to be considered more carefully
				#If there is an NA for N1 then need to remove all of the corresponding N3
				#If there is an NA in an N3 then the N1 which this corresponds to has a different N3, i.e. after removing NAs
				elif self.analysisType == 1:
					#start and end are the position for the N1xN3 sample for a specific initial condition
					start = j*Nparticles + self.N1sample+self.N2sample
					end = j*Nparticles + self.N1sample+self.N2sample + self.N1sample*self.N3sample
					
					#Counts NAs in N1 and N2
					N1_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample]
					N2_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample and x >= j*Nparticles + self.N1sample]
					new_N2 = self.N2sample-len(N2_NA)
					
					#Calculates the number of NAs in N1xN3 sample but seperates out between each N3 belonging to a specific N1
					additional_N1N3_NAs = [range(int(j*Nparticles + self.N1sample + self.N2sample + x*self.N3sample),int(j*Nparticles + self.N1sample + self.N2sample + (x+1)*self.N3sample)) for x in N1_NA - j*Nparticles*numpy.ones([len(N1_NA)])]
					y = []

					for temp in additional_N1N3_NAs:
						y+=temp

					additional_N1N3_NAs = y

					#Collects all detected NAs into one list
					index_NA_IC = list(set().union(index_NA_IC,additional_N1N3_NAs))

					#Finds the NAs in the N1xN3 sample
					N1N3_NA = [x for x in index_NA_IC if x < end and x >= start]
					
					#Finds the remaining particles in N1xN3 sample
					remaining_N1N3 = [item for item in range(start,end) if item not in N1N3_NA]



					#Splits up the N1xN3 into a list for each N1
					keep_N1N3 = [[z for z in y if z not in index_NA_IC] for y in [list(range(x,x+self.N3sample)) for x in range(start,end,self.N3sample)]]

					#Finds the number remaining in the N1xN3 sample
					new_N1N3 = [len(x) for x in keep_N1N3 if len(x)!=0]

					#Adds to indicies to remove
					index_NA_IC = set().union(index_NA_IC, [x+j*Nparticles for x,y in enumerate(new_N1N3) if y == 0])
					
					#Calculates new NAs in N1 sample
					N1_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample]
					new_N1 = self.N1sample-len(N1_NA)
					
					#Sets the number remaining in the sample
					cuda_NAs[cudafile].append([new_N1,new_N2,new_N1N3])

					#Adds NAs to entire list index_NA
					index_NA = list(set().union(index_NA,index_NA_IC))
			
			#Removes the NAs in the cuda output
			cudaout[i] = numpy.delete(cudaout[i], index_NA, axis=0)

		#Sets the size of the sample to attribute after NAs removed
		self.cudaout_structure = cuda_NAs

		#Copies the correct cuda output to the correct experiment
		if self.initialprior == False:
			#For loop over the cuda codes in the experiments
			for model, cudafile in enumerate(self.cuda):
				#Picks corresponding sample for each cuda code
				cudaout_temp = cudaout[cudaorder.index(cudafile)]
			
				#Finds which initial condition this experiment has
				pos = self.pairParamsICS[cudafile].index([x[1] for x in self.x0prior[model]])
				
				#Find total number of particles associated with cuda file
				if self.analysisType!=1:
					size_cudaout_start = [sum(cuda_NAs[cudafile][x]) for x in range(pos-1)]
					size_cudaout_start = sum(size_cudaout_start)
					size_cudaout_end = size_cudaout_start + sum(cuda_NAs[cudafile][pos])
				else:
					size_cudaout_start  = [sum([cuda_NAs[cudafile][x][0]]+[cuda_NAs[cudafile][x][1]]+cuda_NAs[cudafile][x][2]) for x in range(pos-1)]
					size_cudaout_start = sum(size_cudaout_start)
					size_cudaout_end = size_cudaout_start + sum([cuda_NAs[cudafile][pos][0]]+[cuda_NAs[cudafile][pos][1]]+cuda_NAs[cudafile][pos][2])

				#Copies associated cudaoutput to corresponding experiement
				self.cudaout[model] = cudaout_temp[size_cudaout_start:size_cudaout_end,:,:]
		#If priors over initial conditions then assign just corresponding cuda code
		else:
			for model, cudafile in enumerate(self.cuda):
				self.cudaout[model] = cudaout[cudaorder.index(cudafile)]

		#Calls fitSort() which accounts for which species are measurable
		print "-----Sorting out measurable species-----"
		self.fitSort()

		#Adds noise to the N1 trajectories if ODEs used
		if self.type == 0:
			print "-----Adding noise to CUDA-Sim outputs-----"
			self.addNoise(cudaorder)

	# A method to add noise to the N1 sample
	##(method gets called by sortCUDASimoutput)
	##Arguments: 
	##cudaorder - order in which cudacode files are passed in CUDA-sim
	def addNoise(self,cudaorder):

		#Sets attribute to hold trajectories
		self.trajectories=[""]*len(self.cuda)


		if self.initialprior == False:
			#Iterates over cuda code files for each experiment
			for model, cudafile in enumerate(self.cuda):
				
				#Finds position of initial condition used for a cuda code file
				pos = self.pairParamsICS[cudafile].index([x[1] for x in self.x0prior[model]])
				
				#Finds the corresponding N1 to that cuda file
				N1_temp = self.cudaout_structure[cudafile][pos][0]
				
				#Samples the noise
				noise = normal(loc=0.0, scale=self.sigma, size=(N1_temp,len(self.times),len(self.fitSpecies[model])))
				
				#Adds the noise to the trajectories
				self.trajectories[model] = self.cudaout[model][:N1_temp,:,:] + noise
		else:
			#Iterates over the cuda files for each experiment
			for model, cudafile in enumerate(self.cuda):
				
				#Finds N1 corresponding to that cuda file
				N1_temp = self.cudaout_structure[cudafile][0][0]
				
				#Samples noise
				noise = normal(loc=0.0, scale=self.sigma, size=(N1_temp,len(self.times),len(self.fitSpecies[model])))
				
				#Adds noise to trajectories
				self.trajectories[model] = self.cudaout[model][:N1_temp,:,:] + noise

	# A method to sort out measurable species
	##(method gets called by sortCUDASimoutput)
	##No arguments
	def fitSort(self):

		#Iterates over the potential measurable species for each experiment
		for i, exp_n in enumerate(self.cudaout):
			#Initiates an array to store changes to cuda output depending upon the measurable species
			cudaout_temp = numpy.zeros((exp_n.shape[0],exp_n.shape[1],len(self.fitSpecies[i])))

			#Iterates over each fittable species to the experiment
			for j, fit in enumerate(self.fitSpecies[i]):
				#Picks out measurable species if not combined with other
				if len(fit) == 1:
					cudaout_temp[:,:,j] = exp_n[:,:,fit[0]]
				
				#Calculates the fits if combinations required e.g. species1+species2-species3
				else:
					if fit[1] == "+":
						cudaout_temp[:,:,j] = exp_n[:,:,fit[0]] + exp_n[:,:,fit[2]]
					elif fit[1] == "-":
						cudaout_temp[:,:,j] = exp_n[:,:,fit[0]] - exp_n[:,:,fit[2]]
					for k, part in enumerate(fit[3::2]):
						if part == "+":
							cudaout_temp[:,:,j] += exp_n[:,:,fit[2*k+4]]
						elif part == "-":
							cudaout_temp[:,:,j] -= exp_n[:,:,fit[2*k+4]]

			#Replaces the old attribute assigned to cudaout with new one
			self.cudaout[i] = cudaout_temp

	#Scaling method for approach 1 and 2
	##(method gets called by sorting_files)
	##No arguments
	def scaling(self):
		#Initiates list for each experiment
		self.scale = [""]*self.nmodels

		#Iterates over each experiment
		for model in range(self.nmodels):
			#Calculates the maximum difference between trajectores and output from cuda sim
			maxDistTraj = max([math.fabs(numpy.amax(self.trajectories[model]) - numpy.amin(self.cudaout[model])),math.fabs(numpy.amax(self.cudaout[model]) - numpy.amin(self.trajectories[model]))])
			
			#Sets an arbitray precision
			preci = pow(10,-34)

			#Calculates number proportional to normal distribution based on maximum distance
			FmaxDistTraj = 1.0*math.exp(-(maxDistTraj*maxDistTraj)/(2.0*self.sigma*self.sigma))

			#Depending on how FmaxDistTraj compares to precision set an appropriate scale
			if FmaxDistTraj<preci:
#				self.scale[model] = math.log(pow(1.79*pow(10,300),1.0/(len(self.fitSpecies[model])*len(self.times))),math.e)
				self.scale[model] = pow(1.79*pow(10,300),1.0/(len(self.fitSpecies[model])*len(self.times)))		
			else:
				self.scale[model] = pow(preci,1.0/(len(self.fitSpecies[model])*len(self.times)))*1.0/FmaxDistTraj
	
	#Scaling method for approach 3
	##(method gets called by sorting_files)
	##M_Ref - number of timepoints in reference model
	##P_Ref - number of measurable species in reference model
	##Note if scaling_ge3 called for reference model then M_ref and P_ref take default values
	def scaling_ge3(self,M_Ref = 0,P_Ref = 0):
		#Initiates list of maximum distances between trajectories
		maxDistList =[]

		#Iterates over experiments
		for model in range(self.nmodels):
			#Initiates list for distances
			distance = []
			
			#Calculates distances between trajectories at timepoints
			for tp in range(len(self.times)):
				distance.append(max([math.fabs(numpy.amax(self.trajectories[model][:,tp,:]) - numpy.amin(self.cudaout[model][:,tp,:])),math.fabs(numpy.amax(self.cudaout[model][:,tp,:]) - numpy.amin(self.trajectories[model][:,tp,:]))]))
			
			#Assigns maximum of distance to maxDistList
			maxDistList.append(numpy.amax(numpy.array(distance)))
		
		#MIGHT NOT BE NEEDED
		#maxDistTraj = max(maxDistList)

		#Initiates scales list
		self.scale = [""]*self.nmodels
		
		#Sets arbitrary precision level
		preci = pow(10,-34)

		#Iterates over experiments
		for model in range(self.nmodels):

			#Extracts timepoints and measurable species for each experiment
			M_Alt = len(self.times)
			P_Alt = len(self.fitSpecies[model])
			
			#Finds maximum between reference and experiment
			M_Max = float(max(M_Ref,M_Alt))
			P_Max = float(max(P_Ref,P_Alt))

			#Calculates proportional to normal based on maximum distance between trajectories on log scale
			scale1 = math.log(preci)/(2.0*M_Max*P_Max) + (maxDistList[model]*maxDistList[model])/(2.0*self.sigma*self.sigma)
			
			#Calculates comparative scale
			scale2 = math.log(pow(10,300))/(2.0*M_Max*P_Max)

			#Compares scales and assigns appropriate one
			if(scale1<scale2): self.scale[model] = scale1
			else: self.scale[model] = 0.0

	#A method to copy prior sample from reference model for approach 3
	##(method gets called by sorting_files)
	##refmod - object for the reference model
	def copyTHETAS(self,refmod):
		#Sets the N3 sample for experiments
		
		self.particles -= self.N3sample
		self.N3sample = 0
		
		#Sets parameter sample for experiment
		self.parameterSample = numpy.concatenate((refmod.parameterSample[:self.N1sample+self.N2sample,:],refmod.N4parameterSample),axis = 0)

		#Sets sample for initial conditions
		if self.initialprior == True and refmod.initialprior == True:
			self.speciesSample = numpy.concatenate((refmod.speciesSample[:self.N1sample+self.N2sample,:],refmod.N4speciesSample),axis = 0)
		
		#For when constant initial conditions are used
		elif self.initialprior == False and refmod.initialprior == False:
			
			#Finds set of constant initial conditions
			x0prior_uniq = [self.x0prior[0]]
			for ic in self.x0prior[1:]:
				if ic not in x0prior_uniq:
					x0prior_uniq.append(ic)

			#Initiates list of arrays for species sample
			species = [numpy.zeros([self.particles,self.nspecies_all]) for x in range(len(x0prior_uniq))]  # number of repeats x species in system

			#Copies the initial conditions into species list of arrays
			for ic in range(len(x0prior_uniq)):
				#Loop through number of parameter
				for j in range(len(x0prior_uniq[ic])): 
					#####Constant prior#####
					if(x0prior_uniq[ic][j][0]==0):  # j paramater self.index
						species[ic][:,j] = x0prior_uniq[ic][j][1]
					else:
						print " Prior distribution not defined on initial conditions"
						sys.exit()

			#Sets attribute for species sample
			self.speciesSample = species
		else:
			print "Not sampling from the same prior between reference and experimental models"
			sys.exit()

		#For when compartments are used
		if refmod.ncompparams_all > 0:	
			self.compsSample = numpy.concatenate((refmod.compsSample[:self.N1sample+self.N2sample,:],refmod.N4compsSample),axis = 0)

