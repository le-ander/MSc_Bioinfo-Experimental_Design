

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



def generateTemplate(source, filename="input_file", sumname="summary_file", dataname=None):

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
	
	import libsbml
	import re

	reader=libsbml.SBMLReader()
	
	models_nparameters=[]
	models_nspecies=[]

	for i in range(0, len(source)):
		document_check=reader.readSBML(source[i])
		model_check=document_check.getModel()
		numSpecies_check=model_check.getNumSpecies()
		numGlobalParameters_check=model_check.getNumParameters()
		models_nparameters.append(numGlobalParameters_check)
		models_nspecies.append(numSpecies_check)

	print models_nspecies
	print models_nparameters

	out_file=open(filename+".xml","w")
	sum_file=open(sumname+".xml","w")
	

	have_data = False
	times = []
	fit_species = []
#    vars = []
	prior =[]
	nvar = 0
	first = True
	second = True
	count = 0
	if dataname != None:
		have_data = True
		df = open(dataname,'r')
		for line in df:
			strs = str(line).rstrip().split(' ')
#            vals = [float(i) for i in strs[1:]]
			print strs
			if (strs[0]=="dt") :
				dt=float(strs[1])
			elif (strs[0]=="particles"):
				particles=float(strs[1])
			elif (strs[0] == "timepoint"):
				for i in range(1,len(strs)):
					times.append(float(strs[i]))
			elif (strs[0]=="prior"):
				prior.append([strs[1],float(strs[2]), float(strs[3])])
				count += 1
			elif (strs[0]=="fit"):
				if (len(strs)==1):
					fit_species.append("None")
				else:
					for i in range(1,len(strs)):
						fit_species.append(strs[i])

	print count
	print models_nparameters[0]
					
	if (count<models_nparameters[0]):
		print "Not enough prior defined in input file"
	elif (count>models_nparameters[0]):
		print "More prior than parameters defined in input file"
	elif (count==models_nparameters[0]):
		print "Prior distributions are properly defined"


	print particles
	print times
	print dt
	print prior
	print fit_species



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
#    print times
#    print vars
		
	out_file.write("<input>\n\n")
	out_file.write("######################## number of models\n\n")
	out_file.write("# Number of models for which details are described in this input file\n")
	out_file.write("<modelnumber> "+repr(len(source)*len(fit_species))+ " </modelnumber>\n\n")
	   
	out_file.write("######################## particles\n\n")
	if (have_data== True and particles):
		out_file.write("<particles> "+ repr(particles) +" </particles>\n\n")
	else:
		out_file.write("<particles> 100 </particles>\n\n")

	out_file.write("######################## beta\n\n")
	out_file.write("# Beta is the number of times to simulate each sampled parameter set.\n# This is only applicable for models simulated using Gillespie and SDE\n")
	out_file.write("<beta> 1 </beta>\n\n")

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

	
	out_file.write("</data>\n\n")

	out_file.write("######################## Models\n\n")
	out_file.write("<models>\n")

		
	for j in range (0, len(fit_species)):

		for i in range(0,len(source)):
			sum_file.write("Model "+repr(i+1+j*len(source))+"\n")
			sum_file.write("name: model"+repr(i+1+j*len(source))+"\nsource: "+source[i]+"\n\n")

			out_file.write("<model"+repr(i+1+j*len(source))+">\n")
			out_file.write("<name> model"+repr(i+1+j*len(source))+" </name>\n<source> "+source[i]+" </source>\n\n")
			out_file.write("# type: the method used to simulate your model. ODE, SDE or Gillespie.\n")
			out_file.write("<type> SDE </type>\n\n")  ################# Needs to be adapted

			out_file.write("# Fitting information. If fit is None, all species in the model are fitted to the data in the order they are listed in the model.\n")
			out_file.write("# Otherwise, give a whitespace delimited list of fitting instrictions the same length as the dimensions of your data.\n")
			out_file.write("# Use speciesN to denote the Nth species in your model. Simple arithmetic operations can be performed on the species from your model.\n")
			out_file.write("# For example, to fit the sum of the first two species in your model to your first variable, write fit: species1+species2\n")
			if (have_data==True):
				out_file.write("<fit> " + fit_species[j] + " </fit>\n\n");
			else:
				out_file.write("<fit> None </fit>\n\n")

			document=reader.readSBML(source[i])
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

			out_file.write("# Priors on initial conditions and parameters:\n")
			out_file.write("# one of \n")
			out_file.write("#       constant, value \n")
			out_file.write("#       normal, mean, variance \n")
			out_file.write("#       uniform, lower, upper \n")
			out_file.write("#       lognormal, mean, variance \n\n")

			out_file.write("<initial>\n")

			x=0
			for k in range(0,len(species)):
				##if (species[k].getConstant() == False):
				x=x+1
				#out_file.write(repr(getSpeciesValue(species[k]))+", ")
				out_file.write(" <ic"+repr(x)+"> constant "+repr(getSpeciesValue(species[k]))+" </ic"+repr(x)+">\n")
				sum_file.write("S"+repr(x)+":\t"+species[k].getId()+"\tspecies"+repr(k+1)+"\t("+repr(getSpeciesValue(species[k]))+")\n")
			for k in range(0,len(listOfParameter)):
				if listOfParameter[k].getConstant()==False:
					for j in range(0, len(listOfRules)):
						if listOfRules[j].isRate():
							if parameterId[k]==listOfRules[j].getVariable():
								x=x+1
								paramAsSpecies=paramAsSpecies+1
								#out_file.write(repr(listOfParameter[k].getValue())+", ")
								out_file.write(" <ic"+repr(x)+"> constant "+repr(listOfParameter[k].getValue())+" </ic"+repr(x)+">\n")
								sum_file.write("S"+repr(x)+":\t"+listOfParameter[k].getId()+"\tparameter"+repr(k+1-comp)+"\t("+repr(listOfParameter[k].getValue())+") (parameter included in a rate rule and therefore treated as species)\n")

			out_file.write("</initial>\n\n")

			sum_file.write("\n")
			
			if(numGlobalParameters==0): string=" (all of them are local parameters)\n"
			elif(numGlobalParameters==1): string=" (the first parameter is a global parameter)\n"
			elif(numLocalParameters==0): string=" (all of them are global parameters)\n"
			else: string=" (the first "+repr(numGlobalParameters)+" are global parameter)\n"

			sum_file.write("Parameter: "+repr(numParameters)+string)
			sum_file.write("("+repr(paramAsSpecies)+" parameter is treated as species)\n")

			out_file.write("<parameters>\n")
			print numParameters
			print paramAsSpecies

			counter=0

			if have_data==True:
				for k in range(models_nparameters[i]):
					print k
					counter=counter+1
					sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
					out_file.write("<parameter"+repr(counter)+"> ")
					out_file.write(prior[k][0] + " ")
					out_file.write(repr(prior[k][1]) + " " + repr(prior[k][2]) + " </parameter" + repr(counter)+">\n")

	#            Print = True
	#            if k<len(listOfParameter):
	#                if listOfParameter[k].getConstant()==False:
	#                    for j in range(0, len(listOfRules)):
	#                        if listOfRules[j].isRate():
	#                            if parameterId[k]==listOfRules[j].getVariable(): Print = False
	#            else: Print == True

			else:
				for k in range(numParameters-paramAsSpecies):
					counter=counter+1
					sum_file.write("P"+repr(counter)+":\t"+parameterId[k]+"\t"+parameterId2[k]+"\t("+repr(parameter[k])+")\n")
					out_file.write("<parameter"+repr(counter)+">")
					out_file.write(" constant ")
					out_file.write(repr(parameter[k])+" </parameter"+repr(counter)+">\n")
		
			sum_file.write("\n############################################################\n\n")

			out_file.write("</parameters>\n")
			out_file.write("</model"+repr(i+1+j*len(source))+">\n\n")

	out_file.write("</models>\n\n")
	out_file.write("</input>\n\n")

	out_file.close()
	sum_file.close()

   
generateTemplate(["rep_test1.xml","rep_test2.xml" ], "test1", "test2", "new_file2")