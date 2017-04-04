import sys, numpy
import os
from shutil import copyfile
import re
import csv

import peitho.simulations.simulate.simulation_functions as simulation_functions
import peitho.errors_and_parsers.error_checks.error_check as error_check
import peitho.errors_and_parsers.error_checks.SBML_check as SBML_check
import peitho.errors_and_parsers.ode_parsers.input_data_file_parser as input_data_file_parser
import peitho.errors_and_parsers.ode_parsers.parse_object_info as parse_object_info
import peitho.mut_Info.mutInfos.mutInfo1 as mutInfo1
import peitho.mut_Info.mutInfos.mutInfo2 as mutInfo2
import peitho.mut_Info.mutInfos.mutInfo3 as mutInfo3
import peitho.errors_and_parsers.ode_parsers.cudacodecreater as cudacodecreater
import peitho.mut_Info.outputs.plotbar as plotbar
import peitho.errors_and_parsers.ode_parsers.SBML_reactions as SBML_reactions
import peitho.mut_Info.outputs.csv_output as csv_output
from numpy import *
import time

#Initiates the program
##Requires no arguments to run
def main():
	## Start timing for total runtime
	time3=time.time()

	# Reads in command line arguments
	input_file_SBMLs, input_file_datas, analysis, fname, usesbml, iname, template_creator, memory_check = error_check.input_checker(sys.argv)

	# Run program when in non template mode
	if template_creator==False:
		# Calls SBML_checker - checks all the SBML files that have been inputted - basic check
		SBML_check.SBML_checker([iname+"/"+input_file_SBMLs[i] for i, value in enumerate(usesbml) if value=="0"])

		#list of 1s and 0s with 1s indicating that an SBML file is used and 0s indicating local code is used
		usesbml=[not(bool(int(i))) for i in list(usesbml)]

		if memory_check == True and len(input_file_SBMLs)>1 and analysis != 2:
			print "ERROR: Can only give an estimation of memory for one SBML or input XML file at a time for a = 0 or 1 and two files (one for the reference the other for experiments) when a = 2\n"
			sys.exit()
		elif memory_check == True and len(input_file_SBMLs)>2 and analysis == 2:
			print "ERROR: Can only give an estimation of memory for one SBML or input XML file at a time for a = 0 or 1 and two files (one for the reference the other for experiments) when a = 2\n"
			sys.exit()

		#If statement deals with whether we are doing 1st, 2nd or 3rd approach
		if analysis != 2:
			count = 0
			#Cycles through the list of SBML and local code files
			for i in range(0,len(input_file_SBMLs)):
				#NEED TO REMOVE SEED
				#random.seed(123)
				#If statment between whether SBML or local code used as requires two different workflows
				if usesbml[i]==True:
					sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, input_file_data = input_file_datas[count], memory_check = memory_check)
					count += 1
				else:
					sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, memory_check = memory_check)
		else:
			#In the case of the last approach the first file is always the reference model so is treated differently
			count = 0
			#If statment between whether SBML or local code used as requires two different workflows
			#Reference model
			if usesbml[0] == True:
				#random.seed(123) #NEED TO REMOVE SEED
				ref_model = sorting_files(input_file_SBMLs[0],analysis,fname,usesbml[0], iname, input_file_data = input_file_datas[count], memory_check = memory_check)
				count += 1
			else:
				ref_model = sorting_files(input_file_SBMLs[0],analysis,fname,usesbml[0], iname, memory_check = memory_check)

			#Not reference models
			for i in range(1,len(input_file_SBMLs)):
				#random.seed(123) #NEED TO REMOVE SEED
				#If statment between whether SBML or local code used as requires two different workflows
				if usesbml[i] == True:
					sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, refmod = ref_model,input_file_data = input_file_datas[count], memory_check = memory_check)
					count += 1
				else:
					sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, refmod = ref_model)
		## Stop timing for total runtime
		time4=time.time()

		## Print total runtime
		print "Total Runtime", time4-time3

	# create template from SBML file if in template mode
	elif template_creator==True:
		write_template(input_file_SBMLs,iname,fname)



# A function that works on one SBML or local code at a time
##(gets called by main)
##Arguments:
##input_file_SBML - either an SBML file or the input.xml file name (input.xml file used when doing local code)
##analysis - the approach we want to carry out 1, 2, or 3
##fname - string for the output file name
##usesbml - indicates whether an SBML file is used or local code
##refmod - used for approach 2 when the first SBML/local code is the reference model
##input_file_data - this holds the additional data alongside an SBML file that is required such as total number of particles etc
def sorting_files(input_file_SBML, analysis, fname, usesbml, iname,memory_check, refmod="", input_file_data = ""):
	#Used to remove the .xml at the end of the file if present to name directories
	if analysis == 2 and refmod == "":
		print "-----PRE-PROCESSING FOR REFERENCE MODEL-----\n"
	elif analysis == 2 and refmod != "":
		print "-----PRE-PROCESSING FOR EXPERIMENTS MODEL-----\n"
	else:
		print "-----PRE-PROCESSING-----\n"

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
		if memory_check == False:
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
			print "ERROR: Parameters not defined properly in input file"
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
		if memory_check == False:
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
		if memory_check == False:
			print "-----Generating input XML file-----"
		comb_list = input_data_file_parser.generateTemplate(exp_xml_files, analysis, "input_xml", input_file_data, inpath = inPath, outpath= xml_out, iname=iname)

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
	if memory_check == False:
		print "-----Creating object from input XML file-----"

	#Calls function to make the object
	sbml_obj = parse_object_info.algorithm_info(xml_out+input_xml+".xml", comb_list)

	#Calls a function to make an attribute which is a dictionary that relates cudacode files to the initial conditions it needs
	sbml_obj.getpairingCudaICs()

	#Startes sampling from prior
	if memory_check == False:
		print "-----Sampling from prior-----"
	#Assigns attribute with which approach is being conducted 1, 2, or 3
	sbml_obj.getAnalysisType(analysis)

	if memory_check == False:
		#Samples from prior for approach 1, 2 and 3 only for the reference model
		if sbml_obj.analysisType != 2 or refmod == "":
			sbml_obj.THETAS(inputpath=iname, usesbml=usesbml)
		else:
			#For approach 3 copies over the samples from the reference model
			sbml_obj.copyTHETAS(refmod)

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

		#Makes directory to hold the results files and sets results path variable
		if sbml_obj.analysisType == 0 or sbml_obj.analysisType==1 or (sbml_obj.analysisType == 2 and refmod != ""):
			if not(os.path.isdir("./"+fname+"/results")):
					os.mkdir(fname+"/results")
			if not(os.path.isdir("./"+fname+"/results/results_"+input_file_SBML_name)):
					os.mkdir("./"+fname+"/results/results_"+input_file_SBML_name)

		resultspath="./"+fname+"/results/results_"+input_file_SBML_name
		####

		### Initialise list to hold infinites information
		MutInfo_inf=[[],[]]
		######
		print "\n"

		if sbml_obj.analysisType == 0:
			MutInfo, MutInfo_inf[0], MutInfo_inf[1] =mutInfo1.run_mutInfo1(sbml_obj,input_file_SBML)
		elif sbml_obj.analysisType == 1:
			MutInfo, MutInfo_inf[0], MutInfo_inf[1]=mutInfo2.run_mutInfo2(sbml_obj,input_file_SBML)
		elif sbml_obj.analysisType == 2 and refmod == "":
			return sbml_obj
		elif sbml_obj.analysisType == 2 and refmod != "":
			MutInfo, MutInfo_inf[0], MutInfo_inf[1]=mutInfo3.run_mutInfo3(sbml_obj, refmod, input_file_SBML)

		## Plot bar graphs
		plotbar.plotbarh(MutInfo, sbml_obj.name ,sbml_obj.nmodels ,sbml_obj.analysisType,resultspath,input_file_SBML_name)

		## Writes CSV file containing results
		csv_output.csv_output_writer(MutInfo, MutInfo_inf[0], MutInfo_inf[1] , sbml_obj,resultspath, input_file_data, input_file_SBML_name)

		## End timing for mutual information calculation
		time2=time.time()
		#####

		## print runtime of mutual information calculation
		print "MutualInfo Runtime", time2-time1

	else:
		if analysis != 2:
			memory_checker(sbml_obj,fname)
		elif analysis == 2 and refmod == "":
			r_m = memory_checker(sbml_obj,fname,ref_mod=True)
			return r_m
		elif analysis == 2 and refmod != "":
			memory_checker(sbml_obj,fname,required_mem = refmod)



def write_template(sbml_file,inPath,fname):
	if len(sbml_file) > 1:
		print "ERROR: Can only take one SBML file at a time when generating templates\n"
		sys.exit()
	cudacodecreater.cudacodecreater(sbml_file,inPath=inPath+"/",outPath=fname, template=True)
	input_file_parser_new_2.generateTemplate(sbml_file, 0, "template_input_xml", inpath = inPath, outpath= fname, template_creator=True)



def memory_checker(sbml_obj, fname="_results_",required_mem = 0, ref_mod = False):

	required_mem += sys.getsizeof(sbml_obj)
	num_init = len(set(tuple([tuple([tuple(y) for y in x]) for x in sbml_obj.x0prior])))

	if sbml_obj.analysisType == 1:
		Nparticles = sbml_obj.N1sample+sbml_obj.N2sample+sbml_obj.N1sample*sbml_obj.N3sample
	else:
		Nparticles = sbml_obj.particles

	num_param = sbml_obj.nparameters_all + sbml_obj.ncompparams_all
	num_species = sbml_obj.nspecies_all

	theta_mem = 8*num_param*Nparticles
	ic_mem = 8*num_species*Nparticles*num_init

	required_mem += theta_mem
	required_mem += ic_mem

	num_time = len(sbml_obj.times)
	num_models = sbml_obj.nmodels

	cuda_sim_mem = 8*Nparticles*num_time*num_species*num_models

	required_mem += cuda_sim_mem

	if sbml_obj.analysisType != 2:
		required_mem += 5*1024*1024*1024
		required_mem += 0.95*driver.mem_get_info()[1]
		required_mem /= (1024*1024*1024)
		print "\nEstimated RAM requirements:", required_mem, " GB"
		if required_mem < 32:
			print "WARNING: The estimated RAM requirements are less than the recommended size of 32 GB\n"
		shutil.rmtree(fname)
		sys.exit()
	elif sbml_obj.analysisType == 2 and ref_mod == True:
		return required_mem
	elif sbml_obj.analysisType == 2 and ref_mod == False:
		required_mem += 5*1024*1024*1024
		required_mem += 0.95*driver.mem_get_info()[1]
		required_mem /= (1024*1024*1024)
		print "\nEstimated RAM requirements:", required_mem, " GB"
		if required_mem < 32:
			print "WARNING: The estimated RAM requirements are less than the recommended size of 32 GB\n"
		shutil.rmtree(fname)
		sys.exit()
#Starts the program
#main()
