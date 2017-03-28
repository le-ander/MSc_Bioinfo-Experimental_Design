import sys

def printOptions():

	print "\nList of possible options:"

	print "\n Input options:"
	print "-1\t creates a data file template"
	print "-2\t creates an input_xml.xml template"
	print "-3\t creates templates for both input files"
	
	print "\n-h\t--help\t\t print this list of options."

	print "\n"


def input_checker(sys_arg):
	#Defines the default for variables
	template=3

	#For loop cycles over the command line arguments
	for i in range(1,len(sys_arg)):
		if sys_arg[i].startswith('--'):
			#If help flag is used calls printOptions()
			option = sys_arg[i][2:]
			if option == 'help':
				printOptions()
				sys.exit()

		elif sys_arg[i].startswith('-'):
			option = sys_arg[i][1:]
			#If help flag is used calls printOptions()
			if option == 'h':
				printOptions()
				sys.exit()
			#Sets type of approach
			elif option == '1':
				template=1
			elif option=='2':
				template=2
			elif option=='3':
				template=3
	return template

def main():
	template=input_checker(sys.argv)
	

	if template==1 or template==3:
		out_file = open("data_file_template", 'w')
		
		out_file.write(">type\nODE\n<type")
		out_file.write("\n\n")		
		out_file.write(">dt\n1\n<dt")
		out_file.write("\n\n")
		out_file.write(">particles\n1000\n<particles")
		out_file.write("\n\n")
		out_file.write(">nsample\n100 900 0 0\n<nsample")
		out_file.write("\n\n")
		out_file.write(">sigma\n5.0\n<sigma")
		out_file.write("\n\n")
		out_file.write(">timepoint\n0.0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30\n<timepoint")
		out_file.write("\n\n")
		out_file.write(">samplefromposterior\nFalse\ndata_2.txt\nw_1.txt\n<samplefromposterior")
		out_file.write("\n\n")
		out_file.write(">initprior\nFalse\n<initprior")
		out_file.write("\n\n")
		out_file.write(">prior\nuniform 1 10\nuniform 0 20\nuniform 500 2000\nuniform 0 10\n<prior")
		out_file.write("\n\n")
		out_file.write(">initials\nconstant 0\nconstant 1\nconstant 0\nconstant 0\nconstant 0\nconstant 0\n<initials")
		out_file.write("\n\n")
		out_file.write(">fit\nAll\nspecies1 species2\nspecies2+species3\n<fit")
		out_file.write("\n\n")
		out_file.write(">paramfit\nAll\n<paramfit")
		out_file.write("\n\n")
		out_file.write(">initialfit\nNone\n<initialfit")
		out_file.write("\n\n")
		out_file.write(">compfit\nNone\n<compfit")
		out_file.write("\n\n")
		out_file.write(">compartment\nconstant 1\n<compartment")
		out_file.write("\n\n")
		out_file.write(">Initial Conditions 1\nUnchanged\n<Initial Conditions 1")
		out_file.write("\n\n")
		out_file.write(">Initial Conditions 2\nconstant 2.0\nconstant 3.0\nconstant 4.0\nconstant 5.0\nconstant 6.0\nconstant 7.0\n<Initial Conditions 2")
		out_file.write("\n\n")
		out_file.write(">Parameter - Experiment 1\nUnchanged\n<Parameter - Experiment 1")
		out_file.write("\n\n")
		out_file.write(">Parameter - Experiment 2\nalpha0 0.1 1\nalpha0 0.1 3\nalpha0 0.1 5\n<Parameter - Experiment 2")
		out_file.write("\n\n")
		out_file.write(">combination\ninitset1 paramexp1 fit1\ninitset2 paramexp1 fit2\ninitset1 paramexp2 fit3\n<combination")
		
		out_file.close()
		
		
		
	if template==2 or template==3:
		out_file = open("input_xml_template.xml", 'w')
		#####Writing input.xml file and summary file####################################################################
		out_file.write("<input>\n\n")
	
		####Write number of models/experiments defined to input.xml file#####################################################################
		out_file.write("######################## number of models\n\n")
		out_file.write("# Number of models for which details are described in this input file\n")
		out_file.write("<modelnumber> 1 </modelnumber>\n\n")
		###################################################################################################################################
	
		####Writes number of particles to be simulated to input.xml file#######################################################################
		out_file.write("######################## particles\n\n")
		out_file.write("<particles> 10000 </particles>\n\n")
		###################################################################################################################################
	
		####Writes dt value for ODE/SDE solver to input.xml file###############################################################################
		out_file.write("######################## dt\n\n")
		out_file.write("# Internal timestep for solver.\n# Make this small for a stiff model.\n")
		out_file.write("<dt> 0.01 </dt>\n\n")
	
		###################################################################################################################################
		
		out_file.write("######################## User-supplied data\n\n")
		out_file.write("<data>\n")
	
		####Writes timepoint for simulation output to input.xml file###########################################################################	
		out_file.write("# times: For ABC SMC, times must be a whitespace delimited list\n")
		out_file.write("# In simulation mode these are the timepoints for which the simulations will be output\n")
		out_file.write("<times> 0 1 2 3 4 5 6 7 8 9 10 </times>\n\n")
		###################################################################################################################################
	
		####Writes sample sizes for N1, N2, N3 and N4######################################################################################
		out_file.write("# Sizes of N1, N2, N3 and N4 samples for enthropy calculation\n")
		out_file.write("<nsamples>\n<N1>9000</N1>\n<N2>1000</N2>\n<N3>0</N3>\n<N4>0</N4>\n</nsamples>\n\n")
		###################################################################################################################################
	
		####Writes sigma###################################################################################################################
		out_file.write("# Sigma\n")
		out_file.write("<sigma> 5.0 </sigma>\n\n")
		###################################################################################################################################
	
		####Writes numbers of parameters found in all models###############################################################################
		out_file.write("# Numbers of parameters defined in models below \n")
		out_file.write("<nparameters_all> 5 </nparameters_all> \n\n")
		###################################################################################################################################
	
		####Writes if initial conditions are defined by prior distributions################################################################
		out_file.write("# Indicates if a initial conditions are provided as prior distributions \n")
		out_file.write("Option: True or False")
		out_file.write("<initialprior> False </initialprior>\n\n")
		###################################################################################################################################
	
		####Writes the fit for parameters, initial condtions and compartments##############################################################
		out_file.write("# Single or subset of parameters, initial conditions(if defined as priors) and compartment\n")
		out_file.write("to be considered for calculation of mututal inforamtion:\n")
		out_file.write("Options:\n")
		out_file.write("1) All\n")
		out_file.write("2) None\n")
		out_file.write("3) parameter1 parameter2  /  initial2 initial4  /  compartment1\n\n")
	
		out_file.write("<paramfit> All </paramfit>\n\n")
	
		out_file.write("<initfit> None </initfit>\n\n")
	
		out_file.write("<compfit> None </compfit>\n\n")
		#######################################################################
	
		####Writes if posterior sample is provided and where the samples and associated weights file are located ###########################
		out_file.write("# Indicates if a sample from a posterior + associated weights are provided(True / False) and the names of sample and weight file \n")
		out_file.write("<samplefrompost> False </samplefrompost>\n")
		out_file.write("<samplefrompost_file>  </samplefrompost_file>\n")
		out_file.write("<samplefrompost_weights>  </samplefrompost_weights>\n\n")		
		#######################################################################
	
		out_file.write("</data>\n\n")
		
	
	
		####Models/Experiments################################################################
		out_file.write("######################## Models\n\n")
		out_file.write("<models>\n")
	
	
		####Writes general information about model/experiment##############################################################################
		#####SBML source file, associated cuda file and type of model#####################################################################
		out_file.write("<model1>\n")
		out_file.write("<name> model 1 </name>\n")
		out_file.write("<source> SBML_file_1.xml </source>\n")
		out_file.write("<cuda> Exp1.cu </cuda>\n\n")
		out_file.write("# type: the method used to simulate your model. ODE, SDE or Gillespie.\n")
		out_file.write("<type> ODE </type>\n\n")
		###################################################################################################################################
		####Writes which species will be fitted############################################################################################
		out_file.write("# Fitting information. If fit is ALL, all species in the model are fitted to the data in the order they are listed in the model.\n")
		out_file.write("# Otherwise, give a whitespace delimited list of fitting instrictions the same length as the dimensions of your data.\n")
		out_file.write("# Use speciesN to denote the Nth species in your model. Simple arithmetic operations can be performed on the species from your model.\n")
		out_file.write("# For example, to fit the sum of the first two species in your model to your first variable, write fit: species1+species2\n")
		out_file.write("<fit> All </fit>\n\n")
	
		###################################################################################################################################
		out_file.write("# Priors on initial conditions, compartments and parameters:\n")
		out_file.write("# one of \n")
		out_file.write("#       constant, value \n")
		out_file.write("#       normal, mean, variance \n")
		out_file.write("#       uniform, lower, upper \n")
		out_file.write("#       lognormal, mean, variance \n\n")
		out_file.write("#       posterior \n\n")
		####Writes values/prior for initial conditions#####################################################################################
		out_file.write("<initial>\n")
		out_file.write("<ic1> ")
		out_file.write("constant 1")
		out_file.write(" </ic1>\n")
		
		out_file.write("<ic2> ")
		out_file.write("constant 2")
		out_file.write(" </ic2>\n")
		
		out_file.write("<ic3> ")
		out_file.write("constant 3")
		out_file.write(" </ic3>\n")
		
		out_file.write("<ic4> ")
		out_file.write("constant 4")
		out_file.write(" </ic4>\n")
	
		out_file.write("</initial>\n\n")
		###################################################################################################################################
	
		####Writes compartment defined for model/experiment and their associated sizes expressed as constants or priors####################
		out_file.write("<compartments>\n")
	
		out_file.write("<compartment1> ")
		out_file.write("constant 1")
		out_file.write(" </compartment1>\n")
	
		out_file.write("</compartments>\n\n")
		###################################################################################################################################
	
	
		####Writes parameters of model/experiment defined for model/experiment and their associated sizes expressed as priors##############
		out_file.write("<parameters>\n")
		##Constant
		out_file.write("<parameter1> ")
		out_file.write("constant 1")
		out_file.write(" </parameter1>\n")
		##Uniform distribution
		out_file.write("<parameter2> ")
		out_file.write("uniform 1 10")
		out_file.write(" </parameter2>\n")
		##Normal distribution
		out_file.write("<parameter3> ")
		out_file.write("normal 2 20")
		out_file.write(" </parameter3>\n")
		##Lognormal distribution
		out_file.write("<parameter4> ")
		out_file.write("lognormal 3 30")
		out_file.write(" </parameter4>\n")
		##Placeholder if sample from posterior is provided
		out_file.write("<parameter5> ")
		out_file.write("posterior")
		out_file.write(" </parameter5>\n")
	
		out_file.write("</parameters>\n")
		###################################################################################################################################
		out_file.write("</model1>\n\n") 
	
		out_file.write("</models>\n\n")
		out_file.write("</input>\n\n")
		
	
		#####Closes input xml file################################################################################################
		out_file.close()
	if template==1:
		print "\nData file template has been created.\n"
	elif template==2:
		print "\nInput_xml template has been created.\n"
	elif template==3:
		print "\nData file and input_xml template have been created.\n"


main()
	
	