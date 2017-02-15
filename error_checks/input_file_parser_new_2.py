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



def generateTemplate(source, filename="input_file", sumname="summary_file", dataname=None, inpath="", outpath="",iname=""):

	"""

	Generate a model summary file (model_summary.txt) and a template file (filename) from one or more SBML source files.


	***** args *****

	source:    a list of strings.
			   Each entry describes a SBML file.


	***** kwargs *****

	filename:  a string.
			   The name of the template to be generated.

	sumnname:  a string.
			   The name of the summary to be generated.

	dataname:  a string.
			   The name of a datafile.

	"""

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


	if (len(set(models_nparameters)) == 1) and (len(set(models_nspecies)) ==1):
		nparameters_all=models_nparameters[0]
		nspecies_all = models_nspecies[0]
	else:
		print "Input models do not have identical numbers of parameters"
		sys.exit()

	out_file=open(outpath+"/"+filename+".xml","w")
	sum_file=open(outpath+"/"+sumname+".xml","w")

	###regex for corresponding cuda file
	cudaid = re.compile(r'Exp_\d+')

	###regex for input file
	prior_regex = re.compile(r'>prior\s*\n(.+)\n<prior', re.DOTALL)

	fit_regex = re.compile(r'>fit\s*\n(.+)\n<fit', re.DOTALL)

	comp_regex = re.compile(r'>compartment\s*\n(.+)\n<compartment', re.DOTALL)

	fitparam_regex = re.compile(r'>paramfit\s*\n(.+)\n<paramfit')

	times_regex = re.compile(r'>timepoint\n(.+)\n<timepoint', re.DOTALL)

	particles_regex = re.compile(r'>particles\n(.+)\n<particles', re.DOTALL)

 	dt_regex = re.compile(r'>dt\n(.+)\n<dt', re.DOTALL)

	init_regex = re.compile(r'>Initial\sConditions\s\d+\s*\n(.+?)\n<Initial\sConditions\s\d', re.DOTALL)

	init_prior_regex = re.compile(r'>initials\s*\n(.+)\n<initials', re.DOTALL)

	samplefrompost_regex = re.compile(r'>samplefromposterior\s*\n(True|False)\s*\n((\w+\.\w+)\n(\w+\.\w+)\n)?<samplefromposterior')
 
	init_prior_bool = re.compile(r'>initprior\s*\n(True|False)\s*\n<initprior', re.DOTALL)

	comb_regex = re.compile(r'>combination\s*\n(.*)\n<combination', re.DOTALL)

	init_fit_regex = re.compile(r'>initialfit\s*\n(.+)\n<initialfit')

	comp_fit_regex = re.compile(r'>compfit\s*\n(.+)\n<compfit')

	nsample_regex = re.compile(r'>nsample\s*\n(.+)\n<nsample')

	sigma_regex = re.compile(r'>sigma\n(.+)\n<sigma', re.DOTALL)




	have_data = False
	times = []
	fit_species = []
	prior =[]
	init_con = []
	fit_param = []
	fit_init = []
	fit_comp = []
	comps=[]
#	nvar = 0    ###used for error checking
	first = True
	second = True
	count = 0
	if dataname != None:
		have_data = True

		data_file = open(iname+"/"+dataname, 'r')
		info = data_file.read()
		data_file.close()


		####obtain dt value
		dt = dt_regex.search(info).group(1)
		dt = float(dt)
		####


		####obtain number of particles
		particles = particles_regex.search(info).group(1)
		particles = int(particles)
		####


		#####obtain timepoint for cudasim
		times = times_regex.search(info).group(1).rstrip().split(" ")
		times = [float(i) for i in times]
		####

		##### nsample
		nsample = nsample_regex.search(info).group(1).rstrip().split(" ")
		nsample = [int(i) for i in nsample]
		#####

		#### sigma
		sigma = sigma_regex.search(info).group(1)
		sigma = float(sigma)
		####

		####Sample form posterior
		samplefrompost_match = samplefrompost_regex.search(info)
		if samplefrompost_match.group(1) == "True":
			samplefrompost = True
			samplefrompost_samplefile=samplefrompost_match.group(3)
			samplefrompost_weights=samplefrompost_match.group(4)
		else:
			samplefrompost = False
		####


		####Init prior
		init_prior_bool = init_prior_bool.search(info).group(1)
		if (init_prior_bool == "True") or (samplefrompost==True):
			init_prior = True
		else:
			init_prior = False
		#####


		#### obtain information when no sample from posterior is provided
		if samplefrompost == False:

			####obtain prior distribution of model parameter	
			prior = prior_regex.search(info).group(1).split("\n")

			####obtain constants of initial condition
			if init_prior == False:
				init_list = init_regex.findall(info)
				init_con = [k.split("\n") for k in init_list]
				fit_init = "None"

			####obtain prior distributions of initial condition
			elif init_prior == True:
				
				init_con.append(init_prior_regex.search(info).group(1).split("\n"))
				
				####obtain fit information for initials
				fit_init_list = init_fit_regex.search(info).group(1)
				fit_init = fit_init_list  ## should solve NONE issue
				####
		####

		
		else:
			#### Token values when sample from posterior is provided
			init_con= [["constant 1" for k in range(nspecies_all)]]
			####

			####obtain fit information for initials
			fit_init_list = init_fit_regex.search(info).group(1)
			fit_init = fit_init_list  ## should solve NONE issue
			####
		####
		

		####obtain fit information for species
		fit_list = fit_regex.search(info).group(1).split("\n")
		fit_species = fit_list  ## should solve NONE issue
		####


		####obtain fit information for parameters
		fitparam_list = fitparam_regex.search(info).group(1)
		fit_param = fitparam_list  ## should solve NONE issue
		####





		####obtain fit information for compartments
		fit_comp_list = comp_fit_regex.search(info).group(1)
		fit_comp = fit_comp_list  ## should solve NONE issue
		####


		####obtain prior distributions for compartment parameters
		comps_list = comp_regex.search(info).group(1).split("\n")
		comps = comps_list
		####


		####combinations
		comb_list = comb_regex.search(info).group(1).split("\n")
		if comb_list[0]=="All":
			all_combination = True
			comb=[]
		else:
			comb = [[int(j)-1 for j in re.search(r'initset(\d+) paramexp(\d+) fit(\d+)', i).group(1,2,3)] for i in comb_list]
			all_combination = False
		#####



#### error checking if times are greater than previous###
#            if first==True:
#                for j in range(1,len(vals)):
#                    vars.append([])
#                first=False
#                nvar = len(vals)-1

#           if first==True:
#                for j in range(0,len(vals)):
#                    times.append([])
#                first=False
#                nvar = len(vals)-1

#            times.append(vals[0])

#            for j in range(1,len(vals)):
#                vars[j-1].append(vals[j])
#


	
	out_file.write("<input>\n\n")
	out_file.write("######################## number of models\n\n")
	out_file.write("# Number of models for which details are described in this input file\n")
	if all_combination==True:
		out_file.write("<modelnumber> "+repr(len(source)*len(fit_species)*len(init_con))+ " </modelnumber>\n\n")
	
	elif all_combination==False:
		out_file.write("<modelnumber> "+repr(len(comb))+ " </modelnumber>\n\n")

	out_file.write("######################## particles\n\n")
	if (have_data== True and particles):
		out_file.write("<particles> "+ repr(particles) +" </particles>\n\n")
	else:
		out_file.write("<particles> 100 </particles>\n\n")


	out_file.write("######################## dt\n\n")
	out_file.write("# Internal timestep for solver.\n# Make this small for a stiff model.\n")
	if (have_data==True and dt):
		out_file.write("<dt> "+ repr(dt) +" </dt>\n\n")
	else:
		out_file.write("<dt> 0.01 </dt>\n\n")


	out_file.write("######################## User-supplied data\n\n")
	out_file.write("<data>\n")
	out_file.write("# times: For ABC SMC, times must be a whitespace delimited list\n")
	out_file.write("# In simulation mode these are the timepoints for which the simulations will be output\n")
	if (have_data==True and times):
		out_file.write("<times>");
		for i in times:
			out_file.write(" "+repr(i) )
		out_file.write(" </times>\n\n");

	else:
		out_file.write("<times> 0 1 2 3 4 5 6 7 8 9 10 </times>\n\n")


	out_file.write("# Sizes of N1, N2, N3 and N4 samples for enthropy calculation\n")
	if (have_data==True and nsample):
		out_file.write("<nsamples>\n");
		for index, i in enumerate(nsample):
			out_file.write("<N"+repr(index+1)+">")
			out_file.write(" "+repr(i) )
			out_file.write(" </N"+repr(index+1)+">\n")
		out_file.write(" </nsamples>\n\n");

	else:
		out_file.write("<nsamples>\n<N1>9000</N1>\n<N2>1000</N2>\n<N3>0</N3>\n<N4>0</N4>\n</nsamples>\n\n")

	out_file.write("# Sigma\n")
	if (have_data==True and sigma):
		out_file.write("<sigma> "+ repr(sigma) +" </sigma>\n\n")
	else:
		out_file.write("<sigma> 5.0 </sigma>\n\n")



	out_file.write("# Numbers of parameters defined in models below \n")
	out_file.write("<nparameters_all> ")
	out_file.write(repr(nparameters_all))
	out_file.write(" </nparameters_all> \n\n")

	out_file.write("# Indicates if a initial conditions are provided as prior distributions \n")
	out_file.write("<initialprior> ")
	out_file.write(repr(init_prior))
	out_file.write(" </initialprior>\n\n")

	out_file.write("# Single or subset of parameters to be considered for calculation of mututal inforamtion:\n")
		
	out_file.write("<paramfit> ")
	out_file.write(fit_param)
	out_file.write(" </paramfit>\n\n")

	out_file.write("<initfit> ")

	out_file.write(fit_init)
	out_file.write(" </initfit>\n\n")

	out_file.write("<compfit> ")
	out_file.write(fit_comp)
	out_file.write(" </compfit>\n\n")



	out_file.write("# Indicates if a sample from a posterior + associated weights are provided(1=True / 0=False) and the names of sample and weight file \n")
	out_file.write("<samplefrompost> ")
	out_file.write(repr(samplefrompost))
	out_file.write(" </samplefrompost>\n")
	if samplefrompost == True:
		out_file.write("<samplefrompost_file> " + samplefrompost_samplefile + " </samplefrompost_file>\n")
		out_file.write("<samplefrompost_weights> " + samplefrompost_weights + " </samplefrompost_weights>\n\n")
	else:
		out_file.write("<samplefrompost_file>  </samplefrompost_file>\n")
		out_file.write("<samplefrompost_weights>  </samplefrompost_weights>\n\n")		


	out_file.write("</data>\n\n")

	out_file.write("######################## Models\n\n")
	out_file.write("<models>\n")



	if all_combination==False:
		for j in range(len(comb)):

			sum_file.write("Model "+repr(j+1)+"\n")
			sum_file.write("name: model"+repr(j+1)+"\nsource: "+source[comb[j][1]]+"\n\n")

			out_file.write("<model"+repr(j+1)+">\n")
			out_file.write("<name> model"+repr(j+1)+" </name>\n<source> "+source[comb[j][1]]+" </source>\n")
			out_file.write("<cuda> " + cudaid.match(source[comb[j][1]]).group() + ".cu </cuda>\n\n")
			out_file.write("# type: the method used to simulate your model. ODE, SDE or Gillespie.\n")
			out_file.write("<type> ODE </type>\n\n")  ################# Needs to be adapted

			out_file.write("# Fitting information. If fit is ALL, all species in the model are fitted to the data in the order they are listed in the model.\n")
			out_file.write("# Otherwise, give a whitespace delimited list of fitting instrictions the same length as the dimensions of your data.\n")
			out_file.write("# Use speciesN to denote the Nth species in your model. Simple arithmetic operations can be performed on the species from your model.\n")
			out_file.write("# For example, to fit the sum of the first two species in your model to your first variable, write fit: species1+species2\n")
			if (have_data==True):
				out_file.write("<fit> " + fit_species[comb[j][2]] + " </fit>\n\n");
			else:
				out_file.write("<fit> All </fit>\n\n")

			document=reader.readSBML(inpath+"/"+source[comb[j][1]])  ###put this outside the for loop
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
			##for k in range(0, len(species)):
				##if (species[k].getConstant() == True):
					##numParameters=numParameters+1
					##parameter.append(getSpeciesValue(species[k]))
					##parameterId.append(species[k].getId())
					##parameterId2.append('species'+repr(k+1))
					##numSpecies=numSpecies-1

			sum_file.write("number of compartments: "+repr(NumCompartments)+"\n")
			sum_file.write("number of reactions: "+repr(NumReactions)+"\n")
			sum_file.write("number of rules: "+repr(model.getNumRules())+"\n")
			if model.getNumRules()>0:
				sum_file.write("\t Algebraic rules: "+repr(r1)+"\n")
				sum_file.write("\t Assignment rules: "+repr(r2)+"\n")
				sum_file.write("\t Rate rules: "+repr(r3)+"\n\n")
			sum_file.write("number of functions: "+repr(model.getNumFunctionDefinitions())+"\n")
			sum_file.write("number of events: "+repr(model.getNumEvents())+"\n\n")


			paramAsSpecies=0
			sum_file.write("Species with initial values: "+repr(numSpecies)+"\n")



			out_file.write("# Priors on initial conditions, compartments and parameters:\n")
			out_file.write("# one of \n")
			out_file.write("#       constant, value \n")
			out_file.write("#       normal, mean, variance \n")
			out_file.write("#       uniform, lower, upper \n")
			out_file.write("#       lognormal, mean, variance \n\n")

			out_file.write("<initial>\n")

			counter=0
			if have_data==True:
				if samplefrompost == False:

					if init_con[comb[j][0]][0]=="Unchanged":
						x=0
						for k in range(0,len(species)):
							##if (species[k].getConstant() == False):
							x=x+1
							#out_file.write(repr(getSpeciesValue(species[k]))+", ")
							out_file.write(" <ic"+repr(x)+"> constant "+repr(getSpeciesValue(species[k]))+" </ic"+repr(x)+">\n")
							sum_file.write("S"+repr(x)+":\t"+species[k].getId()+"\tspecies"+repr(k+1)+"\t("+repr(getSpeciesValue(species[k]))+")\n")


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
							out_file.write("constant 1")
							out_file.write(" </ic" + repr(counter)+">\n")


			else:
				out_file.write("<ic1> ")
				out_file.write("constant 1")
				out_file.write(" </ic1>\n")






			# x=0
			# for k in range(0,len(species)):
			# 	##if (species[k].getConstant() == False):
			# 	x=x+1
			# 	#out_file.write(repr(getSpeciesValue(species[k]))+", ")
			# 	out_file.write(" <ic"+repr(x)+"> constant "+repr(getSpeciesValue(species[k]))+" </ic"+repr(x)+">\n")
			# 	sum_file.write("S"+repr(x)+":\t"+species[k].getId()+"\tspecies"+repr(k+1)+"\t("+repr(getSpeciesValue(species[k]))+")\n")



			# for k in range(0,len(listOfParameter)):
			# 	if listOfParameter[k].getConstant()==False:
			# 		for j in range(0, len(listOfRules)):
			# 			if listOfRules[j].isRate():
			# 				if parameterId[k]==listOfRules[j].getVariable():
			# 					x=x+1
			# 					paramAsSpecies=paramAsSpecies+1
			# 					#out_file.write(repr(listOfParameter[k].getValue())+", ")
			# 					out_file.write(" <ic"+repr(x)+"> constant "+repr(listOfParameter[k].getValue())+" </ic"+repr(x)+">\n")
			# 					sum_file.write("S"+repr(x)+":\t"+listOfParameter[k].getId()+"\tparameter"+repr(k+1-comp)+"\t("+repr(listOfParameter[k].getValue())+") (parameter included in a rate rule and therefore treated as species)\n")

			out_file.write("</initial>\n\n")

			sum_file.write("\n")



			out_file.write("<compartments>\n")
			counter=0
			if have_data==True:
				for k in range(len(comps)):
					counter=counter+1
#					sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
#					out_file.write("<compartment"+repr(counter)+"> ")
#					out_file.write(comps[k][0] + " ")
#					out_file.write(repr(comps[k][1]) + " " + repr(comps[k][2]) + " </compartment" + repr(counter)+">\n")
					out_file.write("<compartment"+repr(counter)+"> ")
					out_file.write(comps[k])
					out_file.write(" </compartment" + repr(counter)+">\n")


			else:
				out_file.write("<compartment1> ")
				out_file.write("constant 1")
				out_file.write(" </compartment1>\n")

			out_file.write("</compartments>\n\n")





			if(numGlobalParameters==0): string=" (all of them are local parameters)\n"
			elif(numGlobalParameters==1): string=" (the first parameter is a global parameter)\n"
			elif(numLocalParameters==0): string=" (all of them are global parameters)\n"
			else: string=" (the first "+repr(numGlobalParameters)+" are global parameter)\n"

			sum_file.write("Parameter: "+repr(numParameters)+string)
			sum_file.write("("+repr(paramAsSpecies)+" parameter is treated as species)\n")

			out_file.write("<parameters>\n")
#				print numParameters
#				print paramAsSpecies

			counter=0

			if have_data==True:
				if samplefrompost==False:
					for k in range(models_nparameters[comb[j][1]]):
						counter=counter+1
						sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
						out_file.write("<parameter"+repr(counter)+"> ")
						out_file.write(prior[k])
						out_file.write(" </parameter" + repr(counter)+">\n")
	#					out_file.write(prior[k][0] + " ")
	#					out_file.write(repr(prior[k][1]) + " " + repr(prior[k][2]) + " </parameter" + repr(counter)+">\n")

		#            Print = True
		#            if k<len(listOfParameter):
		#                if listOfParameter[k].getConstant()==False:
		#                    for j in range(0, len(listOfRules)):
		#                        if listOfRules[j].isRate():
		#                            if parameterId[k]==listOfRules[j].getVariable(): Print = False
		#            else: Print == True

				elif samplefrompost==True:
					for k in range(models_nparameters[comb[j][1]]):
						counter=counter+1
						sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
						out_file.write("<parameter"+repr(counter)+"> ")
						out_file.write("constant 1")
						out_file.write(" </parameter" + repr(counter)+">\n")








			else:
				for k in range(numParameters-paramAsSpecies):
					counter=counter+1
					sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
					out_file.write("<parameter"+repr(counter)+">")
					out_file.write(" constant ")
					out_file.write(repr(parameter[k])+" </parameter"+repr(counter)+">\n")

			sum_file.write("\n############################################################\n\n")

			out_file.write("</parameters>\n")
			out_file.write("</model"+repr(j+1)+">\n\n") 



	else:
		comb=[]
		for j in range (0, len(fit_species)):
			for h  in range(0, len(init_con)):

				for i in range(0,len(source)):
					comb.append([h,i,j])	
					sum_file.write("Model "+repr(i+1+h*len(source)+j*len(source)*len(init_con))+"\n")
					sum_file.write("name: model"+repr(i+1+h*len(source)+j*len(source)*len(init_con))+"\nsource: "+source[i]+"\n\n")

					out_file.write("<model"+repr(i+1+h*len(source)+j*len(source)*len(init_con))+">\n")
					out_file.write("<name> model"+repr(i+1+h*len(source)+j*len(source)*len(init_con))+" </name>\n<source> "+source[i]+" </source>\n")
					out_file.write("<cuda> " + cudaid.match(source[i]).group() + ".cu </cuda>\n\n")
					out_file.write("# type: the method used to simulate your model. ODE, SDE or Gillespie.\n")
					out_file.write("<type> ODE </type>\n\n")  ################# Needs to be adapted

					out_file.write("# Fitting information. If fit is ALL, all species in the model are fitted to the data in the order they are listed in the model.\n")
					out_file.write("# Otherwise, give a whitespace delimited list of fitting instrictions the same length as the dimensions of your data.\n")
					out_file.write("# Use speciesN to denote the Nth species in your model. Simple arithmetic operations can be performed on the species from your model.\n")
					out_file.write("# For example, to fit the sum of the first two species in your model to your first variable, write fit: species1+species2\n")
					if (have_data==True):
						out_file.write("<fit> " + fit_species[j] + " </fit>\n\n");
					else:
						out_file.write("<fit> All </fit>\n\n")

					document=reader.readSBML(inpath+"/"+source[i])  ###put this outside the for loop
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
					##for k in range(0, len(species)):
						##if (species[k].getConstant() == True):
							##numParameters=numParameters+1
							##parameter.append(getSpeciesValue(species[k]))
							##parameterId.append(species[k].getId())
							##parameterId2.append('species'+repr(k+1))
							##numSpecies=numSpecies-1

					sum_file.write("number of compartments: "+repr(NumCompartments)+"\n")
					sum_file.write("number of reactions: "+repr(NumReactions)+"\n")
					sum_file.write("number of rules: "+repr(model.getNumRules())+"\n")
					if model.getNumRules()>0:
						sum_file.write("\t Algebraic rules: "+repr(r1)+"\n")
						sum_file.write("\t Assignment rules: "+repr(r2)+"\n")
						sum_file.write("\t Rate rules: "+repr(r3)+"\n\n")
					sum_file.write("number of functions: "+repr(model.getNumFunctionDefinitions())+"\n")
					sum_file.write("number of events: "+repr(model.getNumEvents())+"\n\n")


					paramAsSpecies=0
					sum_file.write("Species with initial values: "+repr(numSpecies)+"\n")



					out_file.write("# Priors on initial conditions, compartments and parameters:\n")
					out_file.write("# one of \n")
					out_file.write("#       constant, value \n")
					out_file.write("#       normal, mean, variance \n")
					out_file.write("#       uniform, lower, upper \n")
					out_file.write("#       lognormal, mean, variance \n\n")

					out_file.write("<initial>\n")

					counter=0
					if have_data==True:
						if samplefrompost == False:

							if init_con[h][0]=="Unchanged":
								x=0
								for k in range(0,len(species)):
									##if (species[k].getConstant() == False):
									x=x+1
									#out_file.write(repr(getSpeciesValue(species[k]))+", ")
									out_file.write(" <ic"+repr(x)+"> constant "+repr(getSpeciesValue(species[k]))+" </ic"+repr(x)+">\n")
									sum_file.write("S"+repr(x)+":\t"+species[k].getId()+"\tspecies"+repr(k+1)+"\t("+repr(getSpeciesValue(species[k]))+")\n")


							else:
								for k in range(len(init_con[h])):
									counter=counter+1

									out_file.write("<ic"+repr(counter)+"> ")
									out_file.write(init_con[h][k])
									out_file.write(" </ic" + repr(counter)+">\n")

						elif samplefrompost == True:
							for k in range(0, len(species)):
									counter=counter+1

									out_file.write("<ic"+repr(counter)+"> ")
									out_file.write("constant 1")
									out_file.write(" </ic" + repr(counter)+">\n")


					else:
						out_file.write("<ic1> ")
						out_file.write("constant 1")
						out_file.write(" </ic1>\n")






					# x=0
					# for k in range(0,len(species)):
					# 	##if (species[k].getConstant() == False):
					# 	x=x+1
					# 	#out_file.write(repr(getSpeciesValue(species[k]))+", ")
					# 	out_file.write(" <ic"+repr(x)+"> constant "+repr(getSpeciesValue(species[k]))+" </ic"+repr(x)+">\n")
					# 	sum_file.write("S"+repr(x)+":\t"+species[k].getId()+"\tspecies"+repr(k+1)+"\t("+repr(getSpeciesValue(species[k]))+")\n")



					# for k in range(0,len(listOfParameter)):
					# 	if listOfParameter[k].getConstant()==False:
					# 		for j in range(0, len(listOfRules)):
					# 			if listOfRules[j].isRate():
					# 				if parameterId[k]==listOfRules[j].getVariable():
					# 					x=x+1
					# 					paramAsSpecies=paramAsSpecies+1
					# 					#out_file.write(repr(listOfParameter[k].getValue())+", ")
					# 					out_file.write(" <ic"+repr(x)+"> constant "+repr(listOfParameter[k].getValue())+" </ic"+repr(x)+">\n")
					# 					sum_file.write("S"+repr(x)+":\t"+listOfParameter[k].getId()+"\tparameter"+repr(k+1-comp)+"\t("+repr(listOfParameter[k].getValue())+") (parameter included in a rate rule and therefore treated as species)\n")

					out_file.write("</initial>\n\n")

					sum_file.write("\n")



					out_file.write("<compartments>\n")
					counter=0
					if have_data==True:
						for k in range(len(comps)):
							counter=counter+1
		#					sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
		#					out_file.write("<compartment"+repr(counter)+"> ")
		#					out_file.write(comps[k][0] + " ")
		#					out_file.write(repr(comps[k][1]) + " " + repr(comps[k][2]) + " </compartment" + repr(counter)+">\n")
							out_file.write("<compartment"+repr(counter)+"> ")
							out_file.write(comps[k])
							out_file.write(" </compartment" + repr(counter)+">\n")


					else:
						out_file.write("<compartment1> ")
						out_file.write("constant 1")
						out_file.write(" </compartment1>\n")

					out_file.write("</compartments>\n\n")





					if(numGlobalParameters==0): string=" (all of them are local parameters)\n"
					elif(numGlobalParameters==1): string=" (the first parameter is a global parameter)\n"
					elif(numLocalParameters==0): string=" (all of them are global parameters)\n"
					else: string=" (the first "+repr(numGlobalParameters)+" are global parameter)\n"

					sum_file.write("Parameter: "+repr(numParameters)+string)
					sum_file.write("("+repr(paramAsSpecies)+" parameter is treated as species)\n")

					out_file.write("<parameters>\n")
	#				print numParameters
	#				print paramAsSpecies

					counter=0

					if have_data==True:
						if samplefrompost==False:
							for k in range(models_nparameters[i]):
								counter=counter+1
								sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
								out_file.write("<parameter"+repr(counter)+"> ")
								out_file.write(prior[k])
								out_file.write(" </parameter" + repr(counter)+">\n")
			#					out_file.write(prior[k][0] + " ")
			#					out_file.write(repr(prior[k][1]) + " " + repr(prior[k][2]) + " </parameter" + repr(counter)+">\n")

				#            Print = True
				#            if k<len(listOfParameter):
				#                if listOfParameter[k].getConstant()==False:
				#                    for j in range(0, len(listOfRules)):
				#                        if listOfRules[j].isRate():
				#                            if parameterId[k]==listOfRules[j].getVariable(): Print = False
				#            else: Print == True

						elif samplefrompost==True:
							for k in range(models_nparameters[i]):
								counter=counter+1
								sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
								out_file.write("<parameter"+repr(counter)+"> ")
								out_file.write("constant 1")
								out_file.write(" </parameter" + repr(counter)+">\n")








					else:
						for k in range(numParameters-paramAsSpecies):
							counter=counter+1
							sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
							out_file.write("<parameter"+repr(counter)+">")
							out_file.write(" constant ")
							out_file.write(repr(parameter[k])+" </parameter"+repr(counter)+">\n")

					sum_file.write("\n############################################################\n\n")

					out_file.write("</parameters>\n")
					out_file.write("</model"+repr(i+1+h*len(source)+j*len(source)*len(init_con))+">\n\n")

	out_file.write("</models>\n\n")
	out_file.write("</input>\n\n")

	out_file.close()
	sum_file.close()
	return comb



