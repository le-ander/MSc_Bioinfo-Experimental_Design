# Algorithm information

import re, sys, numpy
from numpy.random import *
import itertools


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


def parse_fitting_information( mod_str, node, species_number ):
	fitref = node.getElementsByTagName(mod_str)[0]
	tmp = str( fitref.firstChild.data ).split()
	ret1 = []

	if (tmp[0]=="All") and (len(tmp)==1):
		ret_temp = []

		for i in range(1,species_number+1):
			ret_temp.append(i)

		ret1.append(ret_temp)

	else:
		for index, i in enumerate(tmp):
			ttmp = re.sub('species','', i )
			ttmp = re.sub(r'\+', ' + ', ttmp)
			ttmp = re.sub(r'\-', ' - ', ttmp)
			ttmp = ttmp.split(" ")
			ttmp_int = [ parseint(y) for y in ttmp]
			ret1.append(ttmp_int)

	return( ret1 )

class algorithm_info:
	"""
	A class to parse the user-provided input file and return all information required to run the abc-SMC algorithm.

	"""

	def __init__(self, filename, mode):
		xmldoc = minidom.parse(filename)
		self.mode = mode
		### mode is 0  inference, 1 simulate, 2 design

		self.modelnumber = 0
		self.particles = 0
		self.beta = 0
		self.dt = 0
		self.times = []
		self.ntimes = 0

		self.nspecies_all=0
		self.ncompparams_all=0
		self.nparameters_all = 0
		self.sampleFromPost = False
		self.samplepostfile=""
		self.sampleweightfile=""
		self.fitParams = []
		self.fitICs=[]
		self.fitComps=[]


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

		### get parameter fit information
		fitParams_temp = parse_required_vector_value(dataref, "paramfit", "Please provide whitespace seperated list of subset of parameter <data><paramfit>", str)
		fitParams_regex= re.compile(r'param(\d+)')
		if fitParams_temp[0]!="All":
			self.fitParams = [int(fitParams_regex.match(tppp).group(1)) - 1 for tppp in fitParams_temp]
		else:
			self.fitParams = [i for i in range(self.nparameters_all)]
		'''
		fitIC_temp = parse_required_vector_value(dataref, "ICfit", "Please provide whitespace seperated list of subset of parameter <data><ICfit>", str)
		fitIC_regex= re.compile(r'IC(\d+)')
		if fitIC_temp[0]!="All":
			self.fitICs = [int(fitIC_regex.match(tppp).group(1)) - 1 for tppp in fitIC_temp]
		else:
			self.fitICs = [i for i in range(self.nspecies_all)]

		fitComp_temp = parse_required_vector_value(dataref, "compfit", "Please provide whitespace seperated list of subset of parameter <data><compfit>", str)
		fitComp_regex= re.compile(r'comp(\d+)')
		if fitComp_temp[0]!="All":
			self.fitComps = [int(fitComp_regex.match(tppp).group(1)) - 1 for tppp in fitComp_temp]
		else:
			self.fitComps = [i for i in range(self.ncompparams_all)]
		'''
		### get information about sample from posterior
		self.sampleFromPost = bool(parse_required_single_value( xmldoc, "samplefrompost", "Please provide a boolean value for <samplefrompost>", int ))




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

				compref = m.getElementsByTagName('compartments')[0]
				for p in compref.childNodes:
					if p.nodeType == p.ELEMENT_NODE:
						ncompparam += 1
						prior_tmp = [0,0,0]
						tmp = str( p.firstChild.data ).split()
						self.compprior[self.nmodels-1].append( process_prior( tmp ) )
				
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

		if len(set(self.ncompparams))==1:
			self.ncompparams_all = list(set(self.ncompparams))[0]
		else:
			print "Models don't have the same number of compartments"
			sys.exit()



		if (len(set(self.nparameters))!=1) or (self.nparameters_all != list(set(self.nparameters))[0]):
			print "Models don't have the same number of parameters"
			sys.exit()


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
		print "fitParams:", [i+1 for i in self.fitParams] 
		print "nspecies:", self.nspecies_all
		print "ncompparams:", self.ncompparams_all
		print "sample from posterior:", self.sampleFromPost


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

	def THETAS(self, sampleFromPost="", weight="", inputpath="", usesbml = False, init_condit = False):
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

			if init_condit == True:

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
				for i in self.fitParams:
					params_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = parameters[j,i]

			parameters = numpy.concatenate((parameters[range(self.particles-self.N3sample),:],params_final),axis=0)
			
			if init_condit == True:
				speciesN3 = species[(self.particles-self.N3sample):,:]
				species_final = numpy.concatenate((speciesN3,)*self.N1sample,axis=0)

				for j in range(0,self.N1sample):
					for i in self.fitICs:
						species_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = species[j,i]

				species = numpy.concatenate((species[range(self.particles-self.N3sample),:],species_final),axis=0)
			else:
				for ic in range(len(species)):
					species[ic] = numpy.tile(species[ic][0,:],(self.N1sample*self.N3sample+self.N1sample+self.N2sample,1))
					
			if usesbml == True:
				compsN3 = compartments[(self.particles-self.N3sample):,:]
				comp_final = numpy.concatenate((compsN3,)*self.N1sample,axis=0)
				for j in range(0,self.N1sample):
					for i in self.fitComps:
						comp_final[range((j*self.N3sample),((j+1)*self.N3sample)),i] = compartments[j,i]

				compartments = numpy.concatenate((compartments[range(self.particles-self.N3sample),:],comp_final),axis=0)

		if usesbml == True:
			self.compsSample = compartments

		self.parameterSample = parameters
		self.speciesSample = species
	
	def getAnalysisType(self,analysisType):
		self.analysisType = analysisType

	def getSampleSizes(self,N1=0,N2=0,N3=0,N4=0):
		if N1+N2+N3+N4==self.particles:
			self.N1sample = N1
			self.N2sample = N2
			self.N3sample = N3
			self.N4sample = N4
		else:
			print "Sum of N1, N2, N3, and N4 is not the number of particles given in the input XML file"
			sys.exit()
		   