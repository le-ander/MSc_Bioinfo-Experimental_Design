##### PARSER WAS ADAPTED FROM ABC_SYSBIO PACKAGE ####


import libsbml
import re
import sys
import numpy



def getSpeciesValue(species):
	"""
	Return the initial amount of a species.
	If species.isSetInitialAmount() == True, return the initial amount.
	Otherwise, return the initial concentration.

	***** args *****

	species:    a libsbml.Species object

	"""

	if species.isSetInitialAmount():
		return species.getInitialAmount()
	else:
		return species.getInitialConcentration()



def generateTemplate(source, analysis_type, filename="input_file", dataname=None, inpath="", outpath="",iname="", template_creator=False):

	"""

	Generate a model summary file (model_summary.xml) and a input xml file (filename.xml) from one or more SBML source files.


	***** args *****

	source:    a list of strings.
			   Each entry describes a SBML file.


	***** kwargs *****

	filename:  a string.
			   The name of the template to be generated.

	dataname:  a string.
			   The name of a datafile.

	inpath:  a string.
			   Path where the source SBML files are located.

	outpath:  a string.
			   Path where the template file is written to.

	iname:  a string.
			   Path where the the data file is located.

	"""

	
	####Determine if source SBML files defines models with same number of species and parameters#####
	reader=libsbml.SBMLReader()

	models_nparameters=[]
	models_nspecies=[]

	for i in range(0, len(source)):
		document_check=reader.readSBML(inpath+"/"+source[i])
		model_check=document_check.getModel()
		numSpecies_check=model_check.getNumSpecies()
		numGlobalParameters_check=model_check.getNumParameters()
		models_nparameters.append(numGlobalParameters_check)
		models_nspecies.append(numSpecies_check)

	### MIght be redundant ###

	if (len(set(models_nparameters)) == 1) and (len(set(models_nspecies)) ==1):
		nparameters_all=models_nparameters[0]
		nspecies_all = models_nspecies[0]

	#################################################################################################
	


	####Open input xml file###################################################
	out_file=open(outpath+"/"+filename+".xml","w")
	#################################################################################################

	###regex to identify corresponding cuda file##########################################
	cudaid = re.compile(r'Exp_\d+')
	#################################################################################################

	###regex to read input file###########################################################
	prior_regex = re.compile(r'>prior\s*\n(.+)\n<prior', re.DOTALL)
	fit_regex = re.compile(r'>measuredspecies\s*\n(.+)\n<measuredspecies', re.DOTALL)
	comp_regex = re.compile(r'>compartment\s*\n(.+)\n<compartment', re.DOTALL)
	fitparam_regex = re.compile(r'>paramfit\s*\n(.+)\n<paramfit')
	times_regex = re.compile(r'>timepoint\n(.+)\n<timepoint', re.DOTALL)
	particles_regex = re.compile(r'>particles\n(.+)\n<particles', re.DOTALL)
	dt_regex = re.compile(r'>dt\n(.+)\n<dt', re.DOTALL)
	init_regex = re.compile(r'>Initial\sConditions\s\d+\s*\n(.+?)\n<Initial\sConditions\s\d', re.DOTALL)
	init_prior_regex = re.compile(r'>initials\s*\n(.+)\n<initials', re.DOTALL)
	samplefrompost_regex = re.compile(r'>samplefromposterior\s*\n(True|False)\s*\n((\w+\.\w+)\n(\w+\.\w+)\n)?<samplefromposterior')
	init_prior_bool = re.compile(r'>initprior\s*\n(True|False)\s*\n<initprior')
	comb_regex = re.compile(r'>combination\s*\n(.*)\n<combination', re.DOTALL)
	init_fit_regex = re.compile(r'>initialfit\s*\n(.+)\n<initialfit')
	comp_fit_regex = re.compile(r'>compfit\s*\n(.+)\n<compfit')
	nsample_regex = re.compile(r'>nsample\s*\n(.+)\n<nsample')
	sigma_regex = re.compile(r'>sigma\n(.+)\n<sigma', re.DOTALL)
	type_regex = re.compile(r'>type\s*\n(ODE|SDE)\s*\n<type')
	#################################################################################################
	
	
	##### regex for prior distribution error checks##################################################
	prior_check_constant=re.compile(r'constant ((\d+\.\d+)|(\d+))')
	prior_check_uniform=re.compile(r'uniform ((\d+\.\d+)|(\d+)) ((\d+\.\d+)|(\d+))')
	prior_check_normal=re.compile(r'normal ((\d+\.\d+)|(\d+)) ((\d+\.\d+)|(\d+))')
	prior_check_lognormal=re.compile(r'lognormal ((\d+\.\d+)|(\d+)) ((\d+\.\d+)|(\d+))')
	#################################################################################################



	###Initialise variables to capture information from data file####################################
	have_data = False
	times = []
	fit_species = []
	prior =[]
	init_con = []
	fit_param = []
	fit_init = []
	fit_comp = []
	comps=[]
	#################################################################################################


	###Reads data file and assigns values to variables####################################
	if template_creator == False and dataname!=None:
		
		data_file = open(iname+"/"+dataname, 'r')
		info = data_file.read()
		data_file.close()


		####obtain dt value####
		dt = dt_regex.search(info).group(1)
		try:
			dt = float(dt)
		except:
			print "\n\nERROR: Please provide an integer/float value for dt >dt ... <dt in data file " + dataname +"!\n\n"
			sys.exit()
		#######################


		####obtain number of particles####
		particles = particles_regex.search(info).group(1)
		try:
			particles = int(particles)
		except:
			print "\n\nERROR: Please provide an integer value as number of particles: >particles ... <particles in input data file " + dataname +"!\n\n"
			sys.exit()
		##################################
		
		####obtain type of model (ODE/SDE)#####
		try:
			model_type=type_regex.search(info).group(1)
		except:
			print "\n\nERROR: The model type is not properly defined: >type ... <type in data file " + dataname +"!\n\n"
			sys.exit()
		#### SDE model message#################	
		if model_type == "SDE":
			print "\n\nWARNING: SDE models will be supported in future version!\n\n"
			sys.exit()
		########################################

		####obtain timepoint for cudasim####
		times = times_regex.search(info).group(1).rstrip().split(" ")
		try:
			times = [float(i) for i in times]
		except:
			print "\n\nERROR: Please provide a whitespace seperated list of float value: >timepoints ... <timepoints in input data file " + dataname +"!\n\n"
			sys.exit()

		### error check if a timepoint was defined more than once
		if len(set(times)) != len(times):
			print "\n\nERROR: One or more timepoints are defined twice or more: >timepoints ... <timepoints in input data file " + dataname +"!\n\n"
			sys.exit()
		
		### error check if timepoint are in ascending order ####
		if sorted(times) != times:
			print "\n\nERROR: Please provide a whitespace seperated list of float value in ascending order: >timepoints ... <timepoints in input data file " + dataname +"!\n\n"
			sys.exit()
		
		####################################

		#####obtain the numbers of N1, N2, N3 and N4####
		nsample = nsample_regex.search(info).group(1).rstrip().split(" ")
		try:
			nsample = [int(i) for i in nsample]
		except:
			print "\n\nERROR: Please provide a whitespace seperated list of integer value as samples sizes: >nsample ... <nsample in input data file " + dataname +"!\n\n"
			sys.exit()

		### error check if sample sizes are correctly defined for the slected approach####
		if analysis_type == 0:
			if nsample[0]==0 or nsample[1]==0:
				print "\n\nERROR: N1 and N2 can not be of size 0: >nsample ... <nsample in input data file " + dataname +"!\n\n"
				sys.exit()				
			elif nsample[2]!=0 or nsample[3]!=0:
				print "\n\nERROR: N3 and N4 must be of size 0: >nsample ... <nsample in input data file " + dataname +"!\n\n"
				sys.exit()
			elif nsample[0]+nsample[1]!=particles:
				print "\n\nERROR: Sum of samples N1 and N2 is not equal to number of particle: >nsample ... <nsample in input data file " + dataname +"!\n\n"
				sys.exit()

		elif analysis_type == 1:
			if nsample[0]==0 or nsample[1]==0 or nsample[2]==0:
				print "\n\nERROR: N1, N2 and N3 can not be of size 0: >nsample ... <nsample in input data file " + dataname +"!\n\n"
				sys.exit()
			elif nsample[3]!=0:
				print "\n\nERROR: N4 must of be size 0: >nsample ... <nsample in input data file " + dataname +"!\n\n"
				sys.exit()
			elif nsample[0]+nsample[1]+nsample[2]!=particles:
				print "\n\nERROR: Sum of samples N1, N2 and N3 is not equal to number of particle: >nsample ... <nsample in input data file " + dataname +"!\n\n"
				sys.exit()		
		
		elif analysis_type == 2:
			if nsample[0]==0 or nsample[1]==0 or nsample[2]==0 or nsample[3]==0:
				print "\n\nERROR: N1, N2, N3 and N4 can not be of size 0: >nsample ... <nsample in input data file " + dataname +"!\n\n"
				sys.exit()
			elif nsample[0]+nsample[1]+nsample[2]+nsample[3]!=particles:
				print "\n\nERROR: Sum of samples N1, N2, N3 and N4 is not equal to number of particle: >nsample ... <nsample in input data file " + dataname +"!\n\n"
				sys.exit()
		################################################

		####obtain sigma####
		sigma = sigma_regex.search(info).group(1)
		try:
			sigma = float(sigma)
		except:
			print "\n\nERROR: Please provide an float value for sigma: >sigma ... <sigma in input data file " + dataname +"!\n\n"
			sys.exit()
		####################

		####Sample from posterior#######################################
		'''
		Reads if sample from posterior is provided and obtains the 
		location of the files provided by the user containing the 
		posterior sample and the associated weights.

		'''
		try:
			samplefrompost_match = samplefrompost_regex.search(info)
		except:
			print "\n\nERROR: Sample from posterior in in the wrong format: >samplefromposterior ... <samplefromposterior in input data file " + dataname +"!\n\n"
			sys.exit()
		if samplefrompost_match.group(1) == "True":
			samplefrompost = True
			samplefrompost_samplefile=samplefrompost_match.group(3)
			samplefrompost_weights=samplefrompost_match.group(4)
			
			#### Token values when sample from posterior is provided####
			init_con= [["constant 1" for k in range(nspecies_all)]]
			############################################################

			####obtain fit information for initials####
			fit_init_list = init_fit_regex.search(info).group(1)
			fit_init = fit_init_list
			###########################################

		elif samplefrompost_match.group(1) == "False":
			samplefrompost = False
		#################################################################


		####Determine if initial condition are treated as priors####
		try:
			init_prior_bool = init_prior_bool.search(info).group(1)
		except:
			print "\n\nERROR: Input if initial conditions are defined by prior distributions is not the right format (True or False): >initprior ... <initprior in input data file " + dataname +"!\n\n"
			sys.exit()
		if (init_prior_bool == "True") or (samplefrompost==True):
			init_prior = True
		elif init_prior_bool == "False":
			init_prior = False
		############################################################


		####Obtains formal prior distributions for parameter/(initial condtions)#############
		##when no sample from posterior is provided##
		if samplefrompost == False:

			####obtain prior distribution of model parameter	
			try:
				prior = prior_regex.search(info).group(1).split("\n")
			except:
				print "\n\nERROR: Prior distributions of model parameter are not the right format: >prior ... <prior in input data file " + dataname +"!\n\n"
				sys.exit()
			### error check for prior distribution format ###
			for i in prior:
					if prior_check_constant.match(i) == None and prior_check_uniform.match(i)==None and prior_check_normal.match(i)==None and prior_check_lognormal.match(i)==None:
						print "\n\nERROR: Prior distributions of model parameter are not in the right format: >prior ... <prior in input data file " + dataname +"!"
						sys.exit()
			####obtain constants of initial condition
			if init_prior == False:
				try:
					init_list = init_regex.findall(info)
					init_con = [k.split("\n") for k in init_list]
					fit_init = "None"
				except:
					print "\n\nERROR: Constant values for initial conditions are not in the right format: >Initial Conditions ... <Initial Conditions in input data file " + dataname +"!\n\n"
					sys.exit()
				### error check for initial condition format ###
				for i in init_con:
					for j in i:
						if prior_check_constant.match(j) == None and (j!="Unchanged" and len(i)!=1):
							print "Initial conditions can only be defined as constants: >Initial Conditions ... <Initial Conditions in input data file " + dataname +"!\n\n"
							sys.exit()


			####obtain prior distributions of initial condition
			elif init_prior == True:
				try:
					init_con.append(init_prior_regex.search(info).group(1).split("\n"))
				except:
					print "\n\nERROR: Prior distributions of initial conditions are not in the right format: >initials ... <initials in input data file " + dataname +"!\n\n"
					sys.exit()
				### error check for prior distribution format ###
				for i in init_con:
					for j in i:
						if prior_check_constant.match(i) == None and prior_check_uniform.match(i)==None and prior_check_normal.match(i)==None and prior_check_lognormal.match(i)==None:
							print "\n\nERROR: Prior distributions of initial conditions are not in the right format: >initials ... <initials in input data file " + dataname +"!\n\n"
							sys.exit()

				####obtain fit information for initials
				if analysis_type==1:
					try:
						fit_init_list = init_fit_regex.search(info).group(1)
						fit_init = fit_init_list
					except:
						print "\n\nERROR: Fitting information for initial condition is not in the right format: >initialfit ... <initialfit in input data file " + dataname +"!\n\n"
						sys.exit()
				else:
					fit_init = "All"
				####
		#######################################################################################
		

		####obtain fit information for species####
		try:
			fit_list = fit_regex.search(info).group(1).split("\n")
			fit_species = fit_list
		except:
			print "\n\nERROR: No measurable species are defined: >measuredspecies .... <measurespecies in input data file " + dataname +"!\n\n"
			sys.exit()
		##########################################


		####obtain fit information for parameters####
		if analysis_type==1:		
			try:
				fitparam_list = fitparam_regex.search(info).group(1)
				fit_param = fitparam_list
			except:
				print "\n\nERROR: Fitting information for model parameter is not in the right format: >paramfit .... <paramfit in input data file " + dataname +"!\n\n"
				sys.exit()
		else:
			fit_param = "All"
		#############################################


		####obtain fit information for compartments####
		if analysis_type==1:
			try:
				fit_comp_list = comp_fit_regex.search(info).group(1)
				fit_comp = fit_comp_list
			except:
				print "\n\nERROR: Fitting information for compartments is not in the right format: >compfit .... <compfit in input data file " + dataname +"!\n\n"
				sys.exit()
		else:
			fit_comp = "All"
		###############################################


		####obtain prior distributions for compartment parameters####
		try:
			comps_list = comp_regex.search(info).group(1).split("\n")
			comps = comps_list
		except:
			print "\n\nERROR: Prior distributions/constant of compartments is not in the right format: >compartment .... <compartment in input data file " + dataname +"!\n\n"
			sys.exit()
		#############################################################


		####obtain the user defined combinations of initial conditions, parameter changes and species which define an experiment/model####
		try:
			comb_list = comb_regex.search(info).group(1).split("\n")
		except:
			print "\n\nERROR: Combinations of initial conditions, parameter changes and species fit defining an experiment are not in the right format: >combination .... <combination in input data file " + dataname +"!\n\n"
			sys.exit()
		if comb_list[0]=="All":
			all_combination = True
			comb=[]
			for j in range (0, len(fit_species)):
				for h  in range(0, len(init_con)):
					for i in range(0,len(source)):
						comb.append([h,i,j])
		else:
			try:
				comb = [[int(j)-1 for j in re.search(r'initset(\d+) paramexp(\d+) fit(\d+)', i).group(1,2,3)] for i in comb_list]
			except:
				print "\n\nERROR: Combinations of initial conditions, parameter changes and species fit defining an experiment are not in the right format: >combination .... <combination in input data file " + dataname +"!\n\n"
				sys.exit()
			all_combination = False
		###################################################################################################################################
	elif template_creator==True:
		comb=[[0,0,0]]

	#################################################################################################


	#####Writing input.xml file and summary file####################################################################
	out_file.write("<input>\n\n")

	####Write number of models/experiments defined to input.xml file#####################################################################
	out_file.write("######################## number of experiments\n\n")
	out_file.write("# Number of experiments for which details are described in this input file\n")
	if template_creator==False:
		out_file.write("<experimentnumber> "+repr(len(comb))+ " </experimentnumber>\n\n")
	elif template_creator == True:
		out_file.write("<experimentnumber> 1 </experimentnumber>\n\n")
	###################################################################################################################################

	#### Model type #######################################################################################################################
	out_file.write("######################## type of models\n\n")
	out_file.write("# type: the method used to simulate your model. ODE, SDE or Gillespie.\n")
	if (template_creator==False and model_type):
		out_file.write("<type> "+model_type+" </type>\n\n")
	elif template_creator == True:
		out_file.write("<type> ODE </type>\n\n")

	####Writes number of particles to be simulated to input.xml file#######################################################################
	out_file.write("######################## particles\n\n")
	if (template_creator==False and particles):
		out_file.write("<particles> "+ repr(particles) +" </particles>\n\n")
	elif template_creator == True:
		out_file.write("<particles> 10000 </particles>\n\n")
	###################################################################################################################################

	####Writes dt value for ODE/SDE solver to input.xml file###############################################################################
	out_file.write("######################## dt\n\n")
	out_file.write("# Internal timestep for solver.\n# Make this small for a stiff model.\n")
	if (template_creator==False and dt):
		out_file.write("<dt> "+ repr(dt) +" </dt>\n\n")
	elif template_creator == True:
		out_file.write("<dt> 0.01 </dt>\n\n")

	###################################################################################################################################
	
	out_file.write("######################## User-supplied data\n\n")
	out_file.write("<data>\n")

	####Writes timepoint for simulation output to input.xml file###########################################################################	
	out_file.write("# times: For ABC SMC, times must be a whitespace delimited list\n")
	out_file.write("# In simulation mode these are the timepoints for which the simulations will be output\n")
	if (template_creator==False and times):
		out_file.write("<times>");
		for i in times:
			out_file.write(" "+repr(i) )
		out_file.write(" </times>\n\n");

	elif template_creator == True:
		out_file.write("<times> 0 1 2 3 4 5 6 7 8 9 10 </times>\n\n")
	###################################################################################################################################

	####Writes sample sizes for N1, N2, N3 and N4######################################################################################
	out_file.write("# Sizes of N1, N2, N3 and N4 samples for entropy calculation\n")
	if (template_creator==False and nsample):
		out_file.write("<nsamples>\n");
		for index, i in enumerate(nsample):
			out_file.write("<N"+repr(index+1)+">")
			out_file.write(" "+repr(i) )
			out_file.write(" </N"+repr(index+1)+">\n")
		out_file.write(" </nsamples>\n\n");

	elif template_creator == True:
		out_file.write("<nsamples>\n<N1>9000</N1>\n<N2>1000</N2>\n<N3>0</N3>\n<N4>0</N4>\n</nsamples>\n\n")
	###################################################################################################################################

	####Writes sigma###################################################################################################################
	out_file.write("# Sigma\n")
	if (template_creator==False and sigma):
		out_file.write("<sigma> "+ repr(sigma) +" </sigma>\n\n")
	elif template_creator == True:
		out_file.write("<sigma> 5.0 </sigma>\n\n")
	###################################################################################################################################

	####Writes numbers of parameters found in all experimental models##################################################################
	out_file.write("# Numbers of parameters defined in experimental models below \n")
	out_file.write("<nparameters_all> ")
	if template_creator==False:
		out_file.write(repr(nparameters_all))
	elif template_creator == True:
		out_file.write(repr(nparameters_all))
	out_file.write(" </nparameters_all> \n\n")
	###################################################################################################################################

	####Writes if initial conditions are defined by prior distributions################################################################
	out_file.write("# Indicates if a initial conditions are provided as prior distributions \n")
	out_file.write("<initialprior> ")
	if template_creator==False:
		out_file.write(repr(init_prior))
	elif template_creator == True:
		out_file.write("False")
	out_file.write(" </initialprior>\n\n")
	###################################################################################################################################

	####Writes the fit for parameters, initial condtions and compartments##############################################################
	out_file.write("# Single or subset of parameters to be considered for calculation of mututal inforamtion:\n")
		
	out_file.write("<paramfit> ")
	if template_creator==False:
		out_file.write(fit_param)
	elif template_creator == True:
		out_file.write("All")
	out_file.write(" </paramfit>\n\n")


	out_file.write("<initfit> ")
	if template_creator==False:
		out_file.write(fit_init)
	elif template_creator == True:
		out_file.write("All")
	out_file.write(" </initfit>\n\n")

	out_file.write("<compfit> ")
	if template_creator==False:
		out_file.write(fit_comp)
	elif template_creator == True:
		out_file.write("All")
	out_file.write(" </compfit>\n\n")
	#######################################################################

	####Writes if posterior sample is provided and where the samples and associated weights file are located ###########################
	out_file.write("# Indicates if a sample from a posterior + associated weights are provided(1=True / 0=False) and the names of sample and weight file \n")
	if template_creator==False:
		out_file.write("<samplefrompost> ")
		out_file.write(repr(samplefrompost))
		out_file.write(" </samplefrompost>\n")
		
		if samplefrompost == True:
			out_file.write("<samplefrompost_file> " + samplefrompost_samplefile + " </samplefrompost_file>\n")
			out_file.write("<samplefrompost_weights> " + samplefrompost_weights + " </samplefrompost_weights>\n\n")
		else:
			out_file.write("<samplefrompost_file>  </samplefrompost_file>\n")
			out_file.write("<samplefrompost_weights>  </samplefrompost_weights>\n\n")
	elif template_creator==True:
		out_file.write("<samplefrompost> False </samplefrompost>\n")
		out_file.write("<samplefrompost_file>  </samplefrompost_file>\n")
		out_file.write("<samplefrompost_weights>  </samplefrompost_weights>\n\n")
	#######################################################################

	out_file.write("</data>\n\n")
	


	####Experiments################################################################
	out_file.write("######################## experiments\n\n")
	out_file.write("<experiments>\n")


	for j in range(len(comb)):
		####Writes general information about experiment##############################################################################
		#####SBML source file, associated cuda file and type of model#####################################################################
		if template_creator==False:
			out_file.write("<experiment"+repr(j+1)+">\n")
			out_file.write("<name> experiment"+repr(j+1)+" </name>\n<source> "+source[comb[j][1]]+" </source>\n")
			out_file.write("<cuda> " + cudaid.match(source[comb[j][1]]).group() + ".cu </cuda>\n\n")
		elif template_creator==True:
			out_file.write("<experiment"+repr(j+1)+">\n")
			out_file.write("<name> experiment"+repr(j+1)+" </name>\n<source> "+source[0]+" </source>\n")
			out_file.write("<cuda> " + source[0] + ".cu </cuda>\n\n")
		###################################################################################################################################

		####Writes which species are being measured############################################################################################
		out_file.write("# Fitting information. If measured species is ALL, all species in the experiment are fitted to the data in the order they are listed in the experiment.\n")
		out_file.write("# Otherwise, give a whitespace delimited list of fitting instrictions the same length as the dimensions of your data.\n")
		out_file.write("# Use speciesN to denote the Nth species in your model. Simple arithmetic operations can be performed on the species from your model.\n")
		out_file.write("# For example, to fit the sum of the first two species in your model to your first variable, write fit: species1+species2\n")
		if template_creator==False:
			out_file.write("<measuredspecies> " + fit_species[comb[j][2]] + " </measuredspecies>\n\n");
		elif template_creator==True:
			out_file.write("<measuredspecies> All </measuredspecies>\n\n")
		
		####Open source file and obtain basic information about the model##################################################################
		document=reader.readSBML(inpath+"/"+source[comb[j][1]])
		model=document.getModel()

		numSpecies=model.getNumSpecies()
		numGlobalParameters=model.getNumParameters()

		parameter=[]
		parameterId=[]
		parameterId2=[]
		listOfParameter=[]

		r1=0
		r2=0
		r3=0
		listOfRules=model.getListOfRules()
		for k in range(0, len(listOfRules)):
			if model.getRule(k).isAlgebraic(): r1=r1+1
			if model.getRule(k).isAssignment(): r2=r2+1
			if model.getRule(k).isRate(): r3=r3+1

		comp=0
		NumCompartments=model.getNumCompartments()
		for k in range(0,NumCompartments):
			if model.getCompartment(k).isSetVolume():
				comp=comp+1
				numGlobalParameters=numGlobalParameters+1
				parameter.append(model.getListOfCompartments()[k].getVolume())
				parameterId.append(model.getListOfCompartments()[k].getId())
				parameterId2.append('compartment'+repr(k+1))
				listOfParameter.append(model.getListOfCompartments()[k])

		for k in range(0,numGlobalParameters-comp):
			param=model.getParameter(k)
			parameter.append(param.getValue())
			parameterId.append(param.getId())
			parameterId2.append('parameter'+repr(k+1))
			listOfParameter.append(param)

		numLocalParameters=0
		NumReactions=model.getNumReactions()
		for k in range(0,NumReactions):
			local=model.getReaction(k).getKineticLaw().getNumParameters()
			numLocalParameters=numLocalParameters+local

			for j in range(0,local):
				parameter.append(model.getListOfReactions()[k].getKineticLaw().getParameter(j).getValue())
				parameterId.append(model.getListOfReactions()[k].getKineticLaw().getParameter(j).getId())
				x=len(parameterId)-comp
				parameterId2.append('parameter'+repr(x))
				listOfParameter.append(model.getListOfReactions()[k].getKineticLaw().getParameter(j))

		numParameters=numLocalParameters+numGlobalParameters

		species = model.getListOfSpecies()


		paramAsSpecies=0
		


		###################################################################################################################################
		out_file.write("# Priors on initial conditions, compartments and parameters:\n")
		out_file.write("# one of \n")
		out_file.write("#       constant, value \n")
		out_file.write("#       normal, mean, variance \n")
		out_file.write("#       uniform, lower, upper \n")
		out_file.write("#       lognormal, mean, variance \n")
		out_file.write("#       posterior \n\n")
		####Writes values/prior for initial conditions#####################################################################################
		out_file.write("<initial>\n")

		counter=0
		if template_creator==False:
			if samplefrompost == False:

				if init_con[comb[j][0]][0]=="Unchanged":
					x=0
					for k in range(0,len(species)):
						##if (species[k].getConstant() == False):
						x=x+1
						#out_file.write(repr(getSpeciesValue(species[k]))+", ")
						out_file.write(" <ic"+repr(x)+"> constant "+repr(getSpeciesValue(species[k]))+" </ic"+repr(x)+">\n")


				else:
					for k in range(len(init_con[comb[j][0]])):
						counter=counter+1

						out_file.write("<ic"+repr(counter)+"> ")
						out_file.write(init_con[comb[j][0]][k])
						out_file.write(" </ic" + repr(counter)+">\n")

			elif samplefrompost == True:
				for k in range(0, len(species)):
						counter=counter+1

						out_file.write("<ic"+repr(counter)+"> ")
						out_file.write("posterior")
						out_file.write(" </ic" + repr(counter)+">\n")


		elif template_creator==True:
			x=0
			for k in range(0,len(species)):
				x=x+1
				out_file.write(" <ic"+repr(x)+"> constant "+repr(getSpeciesValue(species[k]))+" </ic"+repr(x)+">\n")


		out_file.write("</initial>\n\n")
		###################################################################################################################################


		####Writes compartment defined for model/experiment and their associated sizes expressed as constants or priors####################
		out_file.write("<compartments>\n")
		counter=0
		if template_creator==False:
			for k in range(len(comps)):
				counter=counter+1
				out_file.write("<compartment"+repr(counter)+"> ")
				out_file.write(comps[k])
				out_file.write(" </compartment" + repr(counter)+">\n")


		elif template_creator==True:
			for k in range(comp):
				counter=counter+1
				out_file.write("<compartment"+repr(counter)+"> ")
				out_file.write("constant "+repr(parameter[k]))
				out_file.write(" </compartment" + repr(counter)+">\n")

		out_file.write("</compartments>\n\n")
		###################################################################################################################################



		####Writes parameters of model defined for experiment and their associated sizes expressed as priors##############
		out_file.write("<parameters>\n")
		counter=0

		if template_creator==False:
			if samplefrompost==False:
				for k in range(models_nparameters[comb[j][1]]):
					counter=counter+1
					out_file.write("<parameter"+repr(counter)+"> ")
					out_file.write(prior[k])
					out_file.write(" </parameter" + repr(counter)+">\n")
#					
			elif samplefrompost==True:
				for k in range(models_nparameters[comb[j][1]]):
					counter=counter+1
					out_file.write("<parameter"+repr(counter)+"> ")
					out_file.write("posterior")
					out_file.write(" </parameter" + repr(counter)+">\n")



		elif template_creator==True:
			for k in range(comp,numParameters-paramAsSpecies):
				counter=counter+1
				out_file.write("<parameter"+repr(counter)+">")
				out_file.write(" constant ")
				out_file.write(repr(parameter[k])+" </parameter"+repr(counter)+">\n")


		out_file.write("</parameters>\n")
		###################################################################################################################################

		out_file.write("</experiment"+repr(j+1)+">\n\n") 
		

	out_file.write("</experiments>\n\n")
	out_file.write("</input>\n\n")
	

	#####Closes input xml file################################################################################################
	out_file.close()
	######################################################################################

	#####Return list of the combination of the experimental condition used to define aboves experiment#####################################
	if template_creator==False:
		return comb
	elif template_creator==True:
		print "Templates created"
