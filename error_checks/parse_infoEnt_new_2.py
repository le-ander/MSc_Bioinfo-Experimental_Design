# Algorithm information

import re, sys, numpy, copy, math
from numpy.random import *
from xml.dom import minidom

# implemented priors
re_prior_const=re.compile('constant')
re_prior_uni=re.compile('uniform')
re_prior_normal=re.compile('normal')
re_prior_logn=re.compile('lognormal')

# True/False
re_true=re.compile('True')
re_none=re.compile('None')


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

def parse_required_vector_value( node, tagname, message, cast ):
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

def process_prior( tmp ):
	prior_tmp = [0,0,0]

	if re_prior_const.match( tmp[0] ):
		prior_tmp[0] = 0
		try:
			prior_tmp[1] = float( tmp[1] )
		except:
			print "\nValue of the prior for model ", self.name[self.nmodels-1], "has the wrong format:", tmp[1]
			sys.exit()

	elif re_prior_normal.match( tmp[0] ):
		prior_tmp[0] = 1
		try:
			prior_tmp[1] = float( tmp[1] )
			prior_tmp[2] = float( tmp[2] )
		except:
			print "\nValue of the prior for model ", self.name[self.nmodels-1], "has the wrong format:", tmp[1]
			sys.exit()

	elif re_prior_uni.match( tmp[0] ):
		prior_tmp[0] = 2
		try:
			prior_tmp[1] = float( tmp[1] )
			prior_tmp[2] = float( tmp[2] )
		except:
			print "\nValue of the prior for model ", self.name[self.nmodels-1], "has the wrong format:", tmp[1]
			sys.exit()

	elif re_prior_logn.match( tmp[0] ):
		prior_tmp[0] = 3
		try:
			prior_tmp[1] = float( tmp[1] )
			prior_tmp[2] = float( tmp[2] )
		except:
			print "\nValue of the prior for model ", self.name[self.nmodels-1], "has the wrong format:", tmp[1]
			sys.exit()
	else:
		print "\nSupplied parameter prior ", tmp[0], " unsupported"
		sys.exit()

	return prior_tmp



def parseint(str):
    try:
        return int(str)
    except ValueError:
        return str

def parseint_index(str):
    try:
    	out=int(str)-1
        return out
    except ValueError:
        return str



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
	A class to parse the user-provided input file and return all information required to run the abc-SMC algorithm.

	"""

	def __init__(self, filename, mode, combination_list):
		xmldoc = minidom.parse(filename)
		self.mode = mode
		### mode is 0  inference, 1 simulate, 2 design

		self.modelnumber = 0
		self.particles = 0
		self.beta = 0
		self.dt = 0
		self.times = []
		self.ntimes = 0
		self.post_sample_file = ""
		self.post_weight_file = ""
		self.comp_fit= []
		self.init_fit= []
		self.param_fit = []
		self.sigma= 0
		self.N1sample = 0
		self.N2sample = 0
		self.N3sample = 0
		self.N4sample = 0
		self.combination = combination_list



		self.nspecies_all=0
		self.nparameters_all = 0
		self.sampleFromPost = False
		self.initialprior = False


		self.nmodels = 0
		self.nparameters = []
		self.nspecies = []
		self.name = []
		self.cuda = []
		self.source = []
		self.type = []
		self.prior = []
		self.x0prior = []
		self.compprior = []
		self.fitSpecies = []

		self.ncompparams = []


		##################################################
		## Required arguments

		### get number of models
		self.modelnumber = parse_required_single_value( xmldoc, "modelnumber", "Please provide an integer value for <modelnumber>", int )

		### get number of samples
		self.particles = parse_required_single_value( xmldoc, "particles", "Please provide an integer value for <particles>", int )


		### get dt
		self.dt = parse_required_single_value( xmldoc, "dt", "Please provide an float value for <dt>", float )


		### get data attributes
		dataref = xmldoc.getElementsByTagName('data')[0]


		# times
		self.times = parse_required_vector_value( dataref, "times", "Please provide a whitespace separated list of values for <data><times>" , float )
		self.ntimes = len(self.times)

		### get global number of parameters
		self.nparameters_all = parse_required_single_value(dataref, "nparameters_all", "Please provide an integer value for <data><nparameters_all>", int)

		### sigma
		self.sigma = parse_required_single_value(dataref, "sigma", "Please provide an integer value for <data><sigma>", float)

		### get information about sample from posterior
		if parse_required_single_value( dataref, "samplefrompost", "Please provide a boolean value for <samplefrompost>", str ).strip()=="True":

			self.sampleFromPost = True
			self.post_sample_file = parse_required_single_value( dataref, "samplefrompost_file", "Please provide a file name for <samplefrompost_file>", str ).strip()
			self.post_weight_file = parse_required_single_value( dataref, "samplefrompost_weights", "Please provide a file name for <samplefrompost_weights>", str ).strip()
		else:
			self.sampleFromPost = False

		if parse_required_single_value( dataref, "initialprior", "Please provide a boolean value for <initialprior>", str ).strip()=="True":
			self.initialprior = True
		else:
			self.initialprior = False

		####nsamples
		nsampleref = xmldoc.getElementsByTagName('nsamples')[0]

		self.N1sample = parse_required_single_value( dataref, "N1", "Please provide an integer value for <nsamples><N1>", int )
		self.N2sample = parse_required_single_value( dataref, "N2", "Please provide an integer value for <nsamples><N2>", int )
		self.N3sample = parse_required_single_value( dataref, "N3", "Please provide an integer value for <nsamples><N3>", int )
		self.N4sample = parse_required_single_value( dataref, "N4", "Please provide an integer value for <nsamples><N4>", int )

		if (self.N1sample+self.N2sample+self.N3sample+self.N4sample)>self.particles:
			print "The sum of N1, N2, N3 and N4 is bigger than given particle number"
			sys.exit()






		### get model attributes
		modelref = xmldoc.getElementsByTagName('models')[0]
		for m in modelref.childNodes:
			if m.nodeType == m.ELEMENT_NODE:
				self.nmodels += 1
				self.prior.append([])
				self.x0prior.append([])
				self.compprior.append([])

				try:
					self.name.append( str(m.getElementsByTagName('name')[0].firstChild.data).strip() )
				except:
					print "Please provide a string value for <name> for model ", self.nmodels
					sys.exit()
				try:
					self.source.append( str(m.getElementsByTagName('source')[0].firstChild.data).strip() )
				except:
					print "Please provide an string value for <source> for model ", self.nmodels
					sys.exit()
				try:
					self.cuda.append( str(m.getElementsByTagName('cuda')[0].firstChild.data).strip() )
				except:
					print "Please provide an string value for <cuda> for model ", self.nmodels
					sys.exit()
				try:
					self.type.append( str(m.getElementsByTagName('type')[0].firstChild.data).strip() )
				except:
					print "Please provide an string value for <type> for model ", self.nmodels
					sys.exit()

				#initref = m.getElementsByTagName('initialvalues')[0]
				#tmp = str( initref.firstChild.data ).split()
				#self.init.append( [ float(i) for i in tmp ] )
				#self.nspecies.append( len( self.init[self.nmodels-1] ) )


#				self.fit.append( parse_fitting_information( m )  )

#				nfitSpecies = 0
#				fitSpeciesref = m.getElementsByTagName('fit')[0]
#				for s in fitSpeciesref.childNodes:
#					if s.nodeType == s.ELEMENT_NODE:
#						nfitSpecies += 1
#						tmp = str(s.firstChild.data).split()
#						self.fitSpecies[self.nmodels-1].append(tmp)


				nparameter = 0
				ncompparam = 0

				try:
					compref = m.getElementsByTagName('compartments')[0]
					for p in compref.childNodes:
						if p.nodeType == p.ELEMENT_NODE:
							ncompparam += 1
							prior_tmp = [0,0,0]
							tmp = str( p.firstChild.data ).split()
							self.compprior[self.nmodels-1].append( process_prior( tmp ) )
				except:
					ncompparam = 0


				paramref = m.getElementsByTagName('parameters')[0]

				for p in paramref.childNodes:
					if p.nodeType == p.ELEMENT_NODE:
						nparameter += 1
						prior_tmp = [0,0,0]
						tmp = str( p.firstChild.data ).split()
						self.prior[self.nmodels-1].append( process_prior( tmp ) )

				ninit = 0
				initref = m.getElementsByTagName('initial')[0]
				for inn in initref.childNodes:
					if inn.nodeType == inn.ELEMENT_NODE:
						ninit += 1
						prior_tmp = [0,0,0]
						tmp = str( inn.firstChild.data ).split()
						self.x0prior[self.nmodels-1].append( process_prior( tmp ) )

#				if nfitSpecies == 0:
#					print "\nNo measurable species specified in model ", self.name[self.nmodels-1]
#					sys.exit()
#				if nfitParams == 0:
#					print "\nNo parameters to fit specified in model ", self.name[self.nmodels-1]
#					sys.exit()
				if nparameter == 0:
					print "\nNo parameters specified in model ", self.name[self.nmodels-1]
					sys.exit()
				if ninit == 0:
					print "\nNo initial conditions specified in model ", self.name[self.nmodels-1]
					sys.exit()

				self.fitSpecies.append( parse_fitting_information('fit', m, ninit )  )
				self.nparameters.append( nparameter )
				self.nspecies.append( ninit )
				self.ncompparams.append( ncompparam )

		if self.nmodels == 0:
			print "\nNo models specified"
			sys.exit()


		if len(set(self.nspecies))==1:
			self.nspecies_all = list(set(self.nspecies))[0]
		else:
			print "Models don't have the same number of species"
			sys.exit()


		if len(set(self.ncompparams))==1: #and list(set(self.ncompparams))[0]!=0:
			self.ncompparams_all = list(set(self.ncompparams))[0]
		elif len(set(self.ncompparams))!=1:
			print "Models don't have the same number of compartments"
			sys.exit()


		if (len(set(self.nparameters))!=1) or (self.nparameters_all != list(set(self.nparameters))[0]):
			print "Models don't have the same number of parameters"
			sys.exit()


		### paramter fit
		self.param_fit =( parse_fitting_information_parameters('paramfit', dataref, 'parameter' ,self.nparameters_all )  )
		###

		### initial fit
		self.init_fit =( parse_fitting_information_parameters('initfit', dataref, 'initial' ,self.nspecies_all )  )
		###

		### compartment fit
		self.comp_fit=( parse_fitting_information_parameters('compfit', dataref, 'compartment' ,self.ncompparams_all )  )
		###



#	def post_cudasim(self, array):
#		self.maxDist =
#		self.scalefactor =
#		self.fTheta =
#		self.m =
#



	def print_info(self):
		print "\nALGORITHM INFO"
		print "modelnumber:", self.modelnumber
		print "samples:", self.particles
		print "dt:", self.dt
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


		print "MODELS:", self.nmodels
		for i in range(self.nmodels):
			print "\t", "npar:", self.nparameters[i]
			print "\t", "nspecies:", self.nspecies[i]
			print "\t", "ncompparams:", self.ncompparams[i]
			print "\t", "name:", self.name[i]
			print "\t", "source:", self.source[i]
			print "\t", "type:", self.type[i]
			print "\t", "fitSpecies:", self.fitSpecies[i]

			print "\t", "init:", self.x0prior[i]
			print "\t", "prior:", self.prior[i]
			print "\t", "comp_prior:", self.compprior[i]
			print "\n"


	def THETAS(self, inputpath="", usesbml = False):
		#create array which holds parameters
		if self.ncompparams_all!=0:
			usesbml = True

		if self.sampleFromPost==False:
			parameters = numpy.zeros([self.particles,self.nparameters_all]) #we might  want to change prior[0] to a globally defined prior in the object

			#obtain Thetas from prior distributions, wich are either constant, uniform, normal or lognormal

			for j in range(len(self.prior[0])): # loop through number of parameter

				#####Constant prior#####
				if(self.prior[0][j][0]==0):  # j paramater self.index
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

				####
				else:
					print " Prior distribution not defined for parameters"
					sys.exit()

			if self.initialprior == True:

				species = numpy.zeros([self.particles,self.nspecies_all])  # number of repeats x species in system

				for j in range(len(self.x0prior[0])): # loop through number of parameter

					#####Constant prior#####
					if(self.x0prior[0][j][0]==0):  # j paramater self.index
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

					####
					else:
						print " Prior distribution not defined on initial conditions"
						sys.exit()

			else:

				x0prior_uniq = [self.x0prior[0]]
				for ic in self.x0prior[1:]:
					if ic not in x0prior_uniq:
						x0prior_uniq.append(ic)


				species = [numpy.zeros([self.particles,self.nspecies_all]) for x in range(len(x0prior_uniq))]  # number of repeats x species in system

				for ic in range(len(x0prior_uniq)):
					for j in range(len(x0prior_uniq[ic])): # loop through number of parameter
						#####Constant prior#####
						if(x0prior_uniq[ic][j][0]==0):  # j paramater self.index
							species[ic][:,j] = x0prior_uniq[ic][j][1]
						else:
							print " Prior distribution not defined on initial conditions"
							sys.exit()

			if usesbml == True:

				compartments = numpy.zeros([self.particles,self.ncompparams_all])

				for j in range(len(self.compprior[0])): # loop through number of parameter

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

					####
					else:
						print " Prior distribution not defined on compartments"
						sys.exit()

		elif self.sampleFromPost==True:
			#obtain Thetas from posterior sample and associated weights
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

			if usesbml == False:
				####Obtain Theta from posterior samples through weigths####
				if(counter==counter2):#and len(self.nparameters[0])==len(param[0])): ### model object needs to include nparameters information
					parameters = numpy.zeros( [self.particles,self.nparameters_all] )
					species = numpy.zeros([self.particles,self.nspecies_all])
					for i in range(self.particles): #repeats
						index = getWeightedSample(weights)  #manually defined function
						parameters[i,:] = param[index][:self.nparameters_all] #self.index indefies list which is used to assign parameter value.  j corresponds to different parameters defines column
						species[i,:] = param[index][-self.nspecies_all:]
				else:
					print "Please provide equal number of particles and weights in model!"
					sys.exit()

			elif usesbml == True:
				####Obtain Theta from posterior samples through weigths####
				if(counter==counter2):#and len(self.nparameters[0])==len(param[0])): ### model object needs to include nparameters information
					compartments = numpy.zeros([self.particles,self.ncompparams_all])
					parameters = numpy.zeros( [self.particles,self.nparameters_all] )
					species = numpy.zeros([self.particles,self.nspecies_all])
					for i in range(self.particles): #repeats
						index = getWeightedSample(weights)  #manually defined function
						compartments[i,:] = param[index][:self.ncompparams_all]
						parameters[i,:] = param[index][self.ncompparams_all:self.ncompparams_all+self.nparameters_all] #self.index indefies list which is used to assign parameter value.  j corresponds to different parameters defines column
						species[i,:] = param[index][-self.nspecies_all:]
				else:
					print "Please provide equal number of particles and weights in model!"
					sys.exit()

		if self.analysisType == 1:
			paramsN3 = parameters[(self.particles-self.N3sample):,:]
			params_final = numpy.concatenate((paramsN3,)*self.N1sample,axis=0)

			for j in range(0,self.N1sample):
				for i in self.param_fit:
					params_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = parameters[j,i]

			parameters = numpy.concatenate((parameters[range(self.particles-self.N3sample),:],params_final),axis=0)

			if self.initialprior == True:
				speciesN3 = species[(self.particles-self.N3sample):,:]
				species_final = numpy.concatenate((speciesN3,)*self.N1sample,axis=0)

				for j in range(0,self.N1sample):
					for i in self.init_fit:
						species_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = species[j,i]

				species = numpy.concatenate((species[range(self.particles-self.N3sample),:],species_final),axis=0)
			else:
				for ic in range(len(species)):
					species[ic] = numpy.tile(species[ic][0,:],(self.N1sample*self.N3sample+self.N1sample+self.N2sample,1))

			if usesbml == True:
				compsN3 = compartments[(self.particles-self.N3sample):,:]
				comp_final = numpy.concatenate((compsN3,)*self.N1sample,axis=0)
				for j in range(0,self.N1sample):
					for i in self.comp_fit:
						comp_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = compartments[j,i]

				compartments = numpy.concatenate((compartments[range(self.particles-self.N3sample),:],comp_final),axis=0)

		if usesbml == True:
			self.compsSample = compartments

		if self.analysisType !=2:
			self.parameterSample = parameters
			self.speciesSample = species
		elif self.analysisType == 2:
			self.N4parameterSample = parameters[self.N1sample+self.N2sample+self.N3sample:self.N1sample+self.N2sample+self.N3sample+self.N4sample,:]
			self.parameterSample = parameters[:self.N1sample+self.N2sample+self.N3sample,:]

			if self.initialprior == True:
				self.N4speciesSample = species[self.N1sample+self.N2sample+self.N3sample:self.N1sample+self.N2sample+self.N3sample+self.N4sample,:]
				self.speciesSample = species[:self.N1sample+self.N2sample+self.N3sample,:]
			else:
				self.speciesSample = [x[:self.N1sample+self.N2sample+self.N3sample,:] for x in species]

			if usesbml == True:
				self.N4compsSample = compartments[self.N1sample+self.N2sample+self.N3sample:self.N1sample+self.N2sample+self.N3sample+self.N4sample,:]
				self.compsSample = compartments[:self.N1sample+self.N2sample+self.N3sample,:]

			self.particles -= self.N4sample
			self.N4sample = 0


	def getAnalysisType(self,analysisType):
		self.analysisType = analysisType

#	def getCombination(self, combination_list):
#		self.combination = combination_list

	'''
	def getSampleSizes(self,N1=0,N2=0,N3=0,N4=0):
		if N1+N2+N3+N4==self.particles:
			self.N1sample = N1
			self.N2sample = N2
			self.N3sample = N3
			self.N4sample = N4
		else:initset2 paramexp3 fit3
			print "Sum of N1, N2, N3, and N4 is not the number of particles given in the input XML file"
			sys.exit()
	'''
	def getpairingCudaICs(self):
		self.pairParamsICS = {}
		if self.sampleFromPost == False:
			if self.initialprior == True:
				for Cfile in set(self.cuda):
					self.pairParamsICS[Cfile]  = [self.x0prior[j] for j in [i for i, x in enumerate(self.cuda) if x == Cfile]][0]
			elif self.initialprior == False:
				for Cfile in set(self.cuda):
					temp = [[l[1] for l in self.x0prior[j]] for j in [i for i, x in enumerate(self.cuda) if x == Cfile]]

					temp_uniq = [temp[0]]
					for ic in temp[1:]:
						if ic not in temp_uniq:
							temp_uniq.append(ic)

					self.pairParamsICS[Cfile] = temp_uniq

	def sortCUDASimoutput(self,cudaorder,cudaout,control = 0):

		self.cudaout=[""]*len(self.cuda)

		if self.analysisType == 1:
			Nparticles = self.N1sample+self.N2sample+self.N1sample*self.N3sample
		else:
			Nparticles = self.particles

		cuda_NAs = dict((k, []) for k in cudaorder)
		for i, cudafile in enumerate(cudaorder):
			index_NA = [p for p, e in enumerate(numpy.isnan(numpy.sum(numpy.sum(cudaout[i][:,:,:],axis=2),axis=1))) if e==True]
			#print self.pairParamsICS.values()[i]
			if self.initialprior == False:
				pairing_ICs = enumerate(self.pairParamsICS.values()[i])
			else:
				pairing_ICs = enumerate(range(1))

			for j, IC in pairing_ICs:
				index_NA_IC = [s for s in index_NA if s < (j+1)*Nparticles  and s >= j*Nparticles]
				#index_NA_IC = [p for p, e in enumerate(numpy.isnan(numpy.sum(numpy.sum(cudaout[i][j*Nparticles:(j+1)*Nparticles,:,:],axis=2),axis=1))) if e==True]
				#print index_NA_IC
				if self.analysisType !=1:
					N1_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample]
					N2_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample and x >= j*Nparticles + self.N1sample]
					N3_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample+self.N3sample and x >= j*Nparticles + self.N1sample + self.N2sample]
					N4_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample+self.N3sample+self.N4sample and x >= j*Nparticles + self.N1sample + self.N2sample+self.N3sample]
					cuda_NAs[cudafile].append([self.N1sample-len(N1_NA),self.N2sample-len(N2_NA),self.N3sample-len(N3_NA),self.N4sample-len(N4_NA)])
				#elif self.analysisType == 2:
				#	N1_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample]
				#	N2_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample and x >= j*Nparticles + self.N1sample]
				#	N3_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample+self.N3sample and x >= j*Nparticles + self.N1sample + self.N2sample]
				#	N4_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample+self.N3sample+self.N4sample and x >= j*Nparticles + self.N1sample + self.N2sample+self.N3sample]
				#	cuda_NAs[cudafile].append([self.N1sample-len(N1_NA),self.N2sample-len(N2_NA),self.N3sample-len(N3_NA),self.N4sample-len(N4_NA)])
				#	sys.exit()
				elif self.analysisType == 1:
					start = j*Nparticles + self.N1sample+self.N2sample
					end = j*Nparticles + self.N1sample+self.N2sample + self.N1sample*self.N3sample
					N1_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample]
					N2_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample+self.N2sample and x >= j*Nparticles + self.N1sample]
					new_N2 = self.N2sample-len(N2_NA)
					additional_N1N3_NAs = [range(int(j*Nparticles + self.N1sample + self.N2sample + x*self.N3sample),int(j*Nparticles + self.N1sample + self.N2sample + (x+1)*self.N3sample)) for x in N1_NA - j*Nparticles*numpy.ones([len(N1_NA)])]

					y = []

					for temp in additional_N1N3_NAs:
						y+=temp

					additional_N1N3_NAs = y

					index_NA_IC = list(set().union(index_NA_IC,additional_N1N3_NAs))

					N1N3_NA = [x for x in index_NA_IC if x < end and x >= start]
					#new_N1N3 = self.N1sample*self.N3sample-len(N1N3_NA)
					remaining_N1N3 = [item for item in range(start,end) if item not in N1N3_NA]

					keep_N1N3 = [[z for z in y if z not in index_NA_IC] for y in [list(range(x,x+self.N3sample)) for x in range(start,end,self.N3sample)]]
					new_N1N3 = [len(x) for x in keep_N1N3 if len(x)!=0]

					index_NA_IC = set().union(index_NA_IC, [x+j*Nparticles for x,y in enumerate(new_N1N3) if y == 0])
					#print remaining_N1N3
					N1_NA = [x for x in index_NA_IC if x < j*Nparticles + self.N1sample]
					new_N1 = self.N1sample-len(N1_NA)
					#print new_N1N3%new_N1


					#print range(j*Nparticles + self.N1sample+self.N2sample,j*Nparticles + self.N1sample+self.N2sample+ self.N1sample*self.N3sample)
					#print ""
					#print [item for item in range(j*Nparticles + self.N1sample+self.N2sample,j*Nparticles + self.N1sample+self.N2sample+ self.N1sample*self.N3sample) if item not in N1N3_NA]
					#print (self.N1sample*self.N3sample-len(N1N3_NA))%(self.N1sample-len(N1_NA))
					#print remaining_N1N3
					cuda_NAs[cudafile].append([new_N1,new_N2,new_N1N3])
					index_NA = list(set().union(index_NA,index_NA_IC))
					#cuda_NAs[cudafile].append([self.N1sample-len(N1_NA),self.N2sample-len(N2_NA),self.N1sample*self.N3sample-len(N1N3_NA)])
			cudaout[i] = numpy.delete(cudaout[i], index_NA, axis=0)

		self.cudaout_structure = cuda_NAs
		#print "-----Sorting out measurable species-----"
		#self.fitSort()

		#if self.type[0] == "ODE":
		#	print "-----Adding noise to CUDA-Sim outputs-----"
		#	self.addNoise(cudaorder, cudaout)



		#print ""
		#print cuda_NAs
		#print ""
		#print self.pairParamsICS
		#print ""
		#print cudaorder
		#print ""
		#print self.cudaout_structure
		#print [x.shape for x in cudaout]
		#print ""
		#print self.cuda
		#print ""
		#print [numpy.isnan(numpy.sum(numpy.sum(cudaout[i][:,:,:],axis=2),axis=1)) for i in range(2)]
		#print cudaout[1][4,:,:]
		#np.sum(np.sum(a, axis=-2), axis=1)
		#[p for p, i in enumerate(isnan(sum(asarray(modelTraj[0])[:,7:8,:],axis=2))) if i==True]
		#print numpy.sum(cudaout[0][:,0:cudaout[0].shape[1],:],axis=2)


		if self.initialprior == False:
			for model, cudafile in enumerate(self.cuda):
				cudaout_temp = cudaout[cudaorder.index(cudafile)]
				#print cudaout_temp.shape

				pos = self.pairParamsICS[cudafile].index([x[1] for x in self.x0prior[model]])
				#print cudaout_temp[pos*sum(cuda_NAs[cudafile][pos]):(pos+1)*sum(cuda_NAs[cudafile][pos]),:,:]
				if self.analysisType!=1:
					size_cudaout_start = [sum(cuda_NAs[cudafile][x]) for x in range(pos-1)]
					size_cudaout_start = sum(size_cudaout_start)
					size_cudaout_end = size_cudaout_start + sum(cuda_NAs[cudafile][pos])
				else:
					size_cudaout_start  = [sum([cuda_NAs[cudafile][x][0]]+[cuda_NAs[cudafile][x][1]]+cuda_NAs[cudafile][x][2]) for x in range(pos-1)]
					size_cudaout_start = sum(size_cudaout_start)
					size_cudaout_end = size_cudaout_start + sum([cuda_NAs[cudafile][pos][0]]+[cuda_NAs[cudafile][pos][1]]+cuda_NAs[cudafile][pos][2])

				self.cudaout[model] = cudaout_temp[size_cudaout_start:size_cudaout_end,:,:]

			#print cudaout
		else:
			for model, cudafile in enumerate(self.cuda):
				self.cudaout[model] = cudaout[cudaorder.index(cudafile)]

		#print ""
		#print [x.shape for x in self.cudaout]
		#print self.cudaout_structure
	#def removeNAs(self):
		#for

		#print [x for x in self.cudaout]
		print "-----Sorting out measurable species-----"
		self.fitSort()


		if self.type[0] == "ODE":
			print "-----Adding noise to CUDA-Sim outputs-----"
			self.addNoise(cudaorder)


	def addNoise(self,cudaorder):
		#print self.sigma
		#print [x.shape for x in cudaout_red]
		#print self.cudaout_structure
		'''
		model_trajs = dict((k, []) for k in cudaorder)
		for i, cudafile in enumerate(cudaorder):

			for j, IC in enumerate(self.pairParamsICS.values()[i]):
				noise = normal(loc=0.0, scale=self.sigma, size=(self.cudaout_structure[cudafile][j][0],len(self.times),self.nspecies_all))
				#model_trajs[cudafile] =  + noise
				pos = self.pairParamsICS.values()[cudaorder.index(cudafile)].index(IC)
				if self.analysisType!=1:
					size_cudaout = sum(self.cudaout_structure[cudafile][pos])
				else:
					size_cudaout = sum([self.cudaout_structure[cudafile][pos][0]]+[self.cudaout_structure[cudafile][pos][1]]+self.cudaout_structure[cudafile][pos][2])

				model_trajs[cudafile].append(cudaout_red[i][j*size_cudaout:j*size_cudaout+self.cudaout_structure[cudafile][j][0],:,:]+noise)
		'''
		#for m in model_trajs.values():
		#	print [x.shape for x in m]

		self.trajectories=[""]*len(self.cuda)

		if self.initialprior == False:
			for model, cudafile in enumerate(self.cuda):
				#cudaout_temp = cudaout_red[cudaorder.index(cudafile)]
				#print cudaout_temp.shape
				#pos = self.pairParamsICS.values()[cudaorder.index(cudafile)].index([x[1] for x in self.x0prior[model]])
				pos = self.pairParamsICS[cudafile].index([x[1] for x in self.x0prior[model]])
				#print cudaout_temp[pos*sum(cuda_NAs[cudafile][pos]):(pos+1)*sum(cuda_NAs[cudafile][pos]),:,:]
				N1_temp = self.cudaout_structure[cudafile][pos][0]
				noise = normal(loc=0.0, scale=self.sigma, size=(N1_temp,len(self.times),len(self.fitSpecies[model])))
				self.trajectories[model] = self.cudaout[model][:N1_temp,:,:] + noise
			#cudaout_red[i][j*size_cudaout:j*size_cudaout+self.cudaout_structure[cudafile][j][0],:,:]+noise
			#print cudaout
		else:
			for model, cudafile in enumerate(self.cuda):
				N1_temp = self.cudaout_structure[cudafile][0][0]
				noise = normal(loc=0.0, scale=self.sigma, size=(N1_temp,len(self.times),len(self.fitSpecies[model])))
				self.trajectories[model] = self.cudaout[model][:N1_temp,:,:] + noise

	def fitSort(self):

		for i, exp_n in enumerate(self.cudaout):
			cudaout_temp = numpy.zeros((exp_n.shape[0],exp_n.shape[1],len(self.fitSpecies[i])))
			for j, fit in enumerate(self.fitSpecies[i]):
				if len(fit) == 1:
					cudaout_temp[:,:,j] = exp_n[:,:,fit[0]]
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
			self.cudaout[i] = cudaout_temp

	def scaling(self):
		self.scale = [""]*self.nmodels
		for model in range(self.nmodels):
			maxDistTraj = max([math.fabs(numpy.amax(self.trajectories[model]) - numpy.amin(self.cudaout[model])),math.fabs(numpy.amax(self.cudaout[model]) - numpy.amin(self.trajectories[model]))])
			
			print maxDistTraj
			preci = pow(10,-34)
			FmaxDistTraj = 1.0*math.exp(-(maxDistTraj*maxDistTraj)/(2.0*self.sigma*self.sigma))
			print FmaxDistTraj
			#print len(self.fitSpecies[model])

			if FmaxDistTraj<preci:
				self.scale[model] = pow(1.79*pow(10,300),1.0/(len(self.fitSpecies[model])*len(self.times)))
			else:
				self.scale[model] = pow(preci,1.0/(len(self.fitSpecies[model])*len(self.times)))*1.0/FmaxDistTraj
		#sys.exit()
		
	def scaling_ge3(self):
		
		maxDistList =[]
		for model in range(self.nmodels):
			distance = []
			# Only dealing with constant number of timepoints over all models here, need to change!
			for tp in range(len(self.times)):
				distance.append(max([math.fabs(numpy.amax(self.trajectories[model][:,tp,:]) - numpy.amin(self.cudaout[model][:,tp,:])),math.fabs(numpy.amax(self.cudaout[model][:,tp,:]) - numpy.amin(self.trajectories[model][:,tp,:]))]))
			print distance
			maxDistList.append(numpy.amax(numpy.array(distance)))
		maxDistTraj = max(maxDistList)

		self.scale = [""]*self.nmodels
		preci = pow(10,-34)
		# Only dealing with constant number of timepoints over all models here, need to change!
		M_Ref = len(self.times)
		P_Ref = len(self.fitSpecies[0])

		for model in range(self.nmodels):

			# Only dealing with constant number of timepoints over all models here, need to change!
			M_Alt = len(self.times)
			P_Alt = len(self.fitSpecies[model])
			# Only dealing with constant number of timepoints over all models here, need to change!
			M_Max = float(max(M_Ref,M_Alt))
			P_Max = float(max(P_Ref,P_Alt))

			scale1 = math.log(preci)/(2.0*M_Max*P_Max) + (maxDistTraj*maxDistTraj)/(2.0*self.sigma*self.sigma)
			scale2 = math.log(pow(10,300))/(2.0*M_Max*P_Max)
			print scale1
			print scale2
			if(scale1<scale2): self.scale[model] = scale1
			else: self.scale[model] = 0.0

		print self.scale
		#sys.exit()

	def copyTHETAS(self,refmod):
		self.particles -= self.N3sample
		self.N3sample = 0

		self.parameterSample = numpy.concatenate((refmod.parameterSample[:self.N1sample+self.N2sample,:],refmod.N4parameterSample),axis = 0)

		if self.initialprior == True and refmod.initialprior == True:
			self.speciesSample = numpy.concatenate((refmod.speciesSample[:self.N1sample+self.N2sample,:],refmod.N4speciesSample),axis = 0)
		elif self.initialprior == False and refmod.initialprior == False:
			x0prior_uniq = [self.x0prior[0]]
			for ic in self.x0prior[1:]:
				if ic not in x0prior_uniq:
					x0prior_uniq.append(ic)


			species = [numpy.zeros([self.particles,self.nspecies_all]) for x in range(len(x0prior_uniq))]  # number of repeats x species in system

			for ic in range(len(x0prior_uniq)):
				for j in range(len(x0prior_uniq[ic])): # loop through number of parameter
					#####Constant prior#####
					if(x0prior_uniq[ic][j][0]==0):  # j paramater self.index
						species[ic][:,j] = x0prior_uniq[ic][j][1]
					else:
						print " Prior distribution not defined on initial conditions"
						sys.exit()

			self.speciesSample = species
		else:
			print "Not sampling from the same prior between reference and experimental models"
			sys.exit()

		if refmod.ncompparams_all > 0:
			self.compsSample = numpy.concatenate((refmod.compsSample[:self.N1sample+self.N2sample,:],refmod.N4compsSample),axis = 0)
