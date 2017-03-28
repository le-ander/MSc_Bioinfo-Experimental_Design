import sys
import os
from shutil import copyfile
import re
sys.path.insert(1, '../Errors_and_Parsers/Error_Checks')
sys.path.insert(1, '../Errors_and_Parsers/ODE_Parsers')
sys.path.insert(1, '../Errors_and_Parsers/SDE_Parsers')
sys.path.insert(1, '../Errors_and_Parsers/abc-sysbio')
sys.path.insert(1, '../Simulations/cudasim')
sys.path.insert(1, '../Simulations/Simulate')
sys.path.insert(1, '../Mut_Info/getEntropies')
sys.path.insert(1, '../Mut_Info/Outputs')


import simulation_functions
import error_check
import SBML_check
import input_file_parser_new_2
import parse_infoEnt_new_2
import getEntropy1
import getEntropy2
import getEntropy3
import cudacodecreater
import plotbar
import SBML_reactions
from numpy import *
import time

#Initiates the program
##Requires no arguments to run
def main():
	## Start timing for total runtime
	time3=time.time()

	# Reads in command line arguments
	input_file_SBMLs, input_file_datas, analysis, fname, usesbml, iname = error_check.input_checker(sys.argv)
	# Calls SBML_checker - checks all the SBML files that have been inputted - basic check
	SBML_check.SBML_checker([iname+"/"+input_file_SBMLs[i] for i, value in enumerate(usesbml) if value=="0"])

	#list of 1s and 0s with 1s indicating that an SBML file is used and 0s indicating local code is used
	usesbml=[not(bool(int(i))) for i in list(usesbml)]

	#If statement deals with whether we are doing 1st, 2nd or 3rd approach
	if analysis != 2:
		count = 0
		#Cycles through the list of SBML and local code files
		for i in range(0,len(input_file_SBMLs)):
			#NEED TO REMOVE SEED
			#random.seed(123)
			#If statment between whether SBML or local code used as requires two different workflows
			if usesbml[i]==True:
				sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, input_file_data = input_file_datas[count])
				count += 1
			else:
				sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname)
	else:
		#In the case of the last approach the first file is always the reference model so is treated differently
		count = 0
		#If statment between whether SBML or local code used as requires two different workflows
		#Reference model
		if usesbml[0] == True:
			random.seed(123) #NEED TO REMOVE SEED
			ref_model = sorting_files(input_file_SBMLs[0],analysis,fname,usesbml[0], iname, input_file_data = input_file_datas[count])
			count += 1
		else:
			ref_model = sorting_files(input_file_SBMLs[0],analysis,fname,usesbml[0], iname)

		#Not reference models
		for i in range(1,len(input_file_SBMLs)):
			#random.seed(123) #NEED TO REMOVE SEED
			#If statment between whether SBML or local code used as requires two different workflows
			if usesbml[i] == True:
				sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, refmod = ref_model,input_file_data = input_file_datas[count])
				count += 1
			else:
				sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, refmod = ref_model)
	## Stop timing for total runtime
	time4=time.time()

	## Print total runtime
	print "Total Runtime", time4-time3


# A function that works on one SBML or local code at a time
##(gets called by main)
##Arguments:
##input_file_SBML - either an SBML file or the input.xml file name (input.xml file used when doing local code)
##analysis - the approach we want to carry out 1, 2, or 3
##fname - string for the output file name
##usesbml - indicates whether an SBML file is used or local code
##refmod - used for approach 2 when the first SBML/local code is the reference model
##input_file_data - this holds the additional data alongside an SBML file that is required such as total number of particles etc
def sorting_files(input_file_SBML, analysis, fname, usesbml, iname, refmod="", input_file_data = ""):
	#Used to remove the .xml at the end of the file if present to name directories
	input_file_SBML_name = input_file_SBML
	if input_file_SBML_name[-4:]==".xml":
		input_file_SBML_name = input_file_SBML_name[:-4]

	#Makes directory to hold the cudacode files
	if not(os.path.isdir("./"+fname+"/cudacodes")):
			os.mkdir(fname+"/cudacodes")

	#Workflow used is SBML file is used
	if usesbml == True:
		#Sets the outpath for where CUDA code is stored
		if not(os.path.isdir("./"+fname+"/cudacodes/cudacodes_"+input_file_SBML_name)):
			os.mkdir(fname+"/cudacodes/cudacodes_"+input_file_SBML_name)
		#outPath is a string to where the cudacode is stored
		outPath=fname+"/cudacodes/cudacodes_"+input_file_SBML_name

		#Depending on the way changes have been made to the SBML files only require certain versions
		input_files_SBML=[]

		#Start of making new SBML files depending on experiments
		print "-----Creating SBML files for experiments-----"

		#Sets directory to hold new SBML files
		if not(os.path.isdir("./"+fname+"/exp_xml")):
			os.mkdir(fname+"/exp_xml")
		if not(os.path.isdir("./"+fname+"/exp_xml/exp_xml_"+input_file_SBML_name)):
			os.mkdir(fname+"/exp_xml/exp_xml_"+input_file_SBML_name)
		#inPath is a string for where the SBML files are stored
		inPath = fname + "/exp_xml/exp_xml_" + input_file_SBML_name

		#Carries out the changes to the original SBML file and then creates a new SBML file in the directory made
		try:
			no_exp = SBML_reactions.SBML_reactionchanges(input_file_SBML, iname, inPath,input_file_data)
		except:
			print ""
			print "Parameters not defined properly in input file"
			print "Need to be defined sequentially e.g."
			print ">Parameter - Experiment 1"
			print "..."
			print "<Parameter - Experiment 1"
			print ""
			print ">Parameter - Experiment 2"
			print "..."  
			print "<Parameter - Experiment 2"
			print ""
			print ">Parameter - Experiment 3"
			print "..."
			print "<Parameter - Experiment 3"
			print ""
			print "Also if you plan to run an unchanged version of SBML file this must be Experiment 1 as:"
			print ">Parameter - Experiment 1"
			print "Unchanged"
			print "<Parameter - Experiment 1\n"
			sys.exit()

		#Start of creating cudacode from SBML files just made
		print "-----Creating CUDA code-----"

		#cudacode files are saved with a specific name and so make a list of these names
		for i in range(0,no_exp):
			input_files_SBML.append("Exp_" + repr(i+1) + ".xml")

		#Creates cudacode and saves to the directory made
		cudacodecreater.cudacodecreater(input_files_SBML,inPath=inPath+"/",outPath=outPath)
		
		#Creates directory to store the input.xml file along with a summary file
		if not(os.path.isdir("./"+fname+"/input_xml")):
			os.mkdir(fname+"/input_xml")
		if not(os.path.isdir("./"+fname+"/input_xml/input_xml_"+input_file_SBML_name)):
			os.mkdir(fname+"/input_xml/input_xml_"+input_file_SBML_name)
		#xml_out is a string for where the input.xml file is stored
		xml_out=fname+"/input_xml/input_xml_"+input_file_SBML_name

		#Obtains a list of the new SBML files
		exp_xml_files = os.listdir(inPath)

		#Start of creating the input.xml file
		print "-----Input XML file-----"
		comb_list = input_file_parser_new_2.generateTemplate(exp_xml_files, analysis, "input_xml", "summmary", input_file_data, inpath = inPath, outpath= xml_out, iname=iname)

		#input_xml holds the file name of the input.xml file
		input_xml="/input_xml"

	#Workflow used if local code is being used
	elif usesbml == False:
		#Labels the variables to where arguments are
		#These are given by the user
		outPath=iname #Where the cudacode is stored
		xml_out=iname #Where the input.xml file is
		#Holds name of the input_.xml file
		input_xml="/"+input_file_SBML_name
		comb_list = []

	#Starts making the object from the input.xml file
	print "-----Creating object from input XML file-----"

	#Calls function to make the object
	sbml_obj = parse_infoEnt_new_2.algorithm_info(xml_out+input_xml+".xml", comb_list)

	#Calls a function to make an attribute which is a dictionary that relates cudacode files to the initial conditions it needs
	sbml_obj.getpairingCudaICs()

	#Startes sampling from prior
	print "-----Sampling from prior-----"
	#Assigns attribute with which approach is being conducted 1, 2, or 3
	sbml_obj.getAnalysisType(analysis)

	#Samples from prior for approach 1, 2 and 3 only for the reference model
	if sbml_obj.analysisType != 2 or refmod == "":
		sbml_obj.THETAS(inputpath=iname, usesbml=usesbml)
	else:
		#For approach 3 copies over the samples from the reference model
		sbml_obj.copyTHETAS(refmod)

	sbml_obj.print_info()
	#Starts CUDA sim
	print "-----Running CUDA-Sim-----"

	#Calls function to run cudasim and sort output
	cudasim_run = simulation_functions.run_cudasim(sbml_obj,inpath=outPath)

	#Calculates the scaling factor
	print "-----Calculating scaling factor-----"
	#Calculating scaling is different when doing approach 3 or not
	if sbml_obj.analysisType != 2:
		#Scaling for when doing approach 1 or 2
		sbml_obj.scaling()
	else:
		#Scaling for when doing approach 3
		if refmod == "":
			sbml_obj.scaling_ge3()
		else:
			sbml_obj.scaling_ge3(len(refmod.times),len(refmod.fitSpecies[0]))

	#Depending upon the approach different functions are run to calculate the mutual information
	
	## Start timing for mutual information calculation
	time1=time.time()
	#####

	if sbml_obj.analysisType == 0:
		MutInfo1=getEntropy1.run_getEntropy1(sbml_obj)
		plotbar.plotbar(MutInfo1, sbml_obj.name ,sbml_obj.nmodels ,0)
	elif sbml_obj.analysisType == 1:
		MutInfo2=getEntropy2.run_getEntropy2(sbml_obj)
		plotbar.plotbar(MutInfo2, sbml_obj.name ,sbml_obj.nmodels ,1)
	elif sbml_obj.analysisType == 2 and refmod == "":
		return sbml_obj
	elif sbml_obj.analysisType == 2 and refmod != "":
		getEntropy3.run_getEntropy3(sbml_obj, refmod)

	## End timing for mutual information calculation
	time2=time.time()
	#####

	## print runtime of mutual information calculation
	print "MutualInfo Runtime", time2-time1

#Starts the program
main()
