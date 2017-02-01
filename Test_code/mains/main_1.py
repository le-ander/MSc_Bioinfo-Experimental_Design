import sys
import os
from shutil import copyfile

sys.path.insert(0, '/cluster/home/saw112/work/Test_code/error_checks')
import error_check
import cudacodecreater
import SBML_check
import input_file_parser

def main():
	# Calls error_checker - reads in command line arguments and does some basic error checks
	input_file_SBMLs, input_file_datas, analysis, fname, usesbml, which_species, parameter_change, init_condit = error_check.input_checker(sys.argv,0)
	# Calls SBML_checker - checks all the SBML files that have been inputted
	SBML_check.SBML_checker([input_file_SBMLs[i] for i in usesbml if usesbml=="0"])
	# Unpacks the following four command line arguments - each element corresponds to each SBML and data file pair 
	usesbml=[not(bool(int(i))) for i in list(usesbml)]
	which_species=[int(i) for i in list(which_species)]
	parameter_change=[int(i) for i in list(parameter_change)]
	init_condit=[int(i) for i in list(init_condit)]
	# Calls sorting_files which creates new SBML files for new experiments and creates CUDA code from SBML files if necessary
	for i in range(0,len(input_file_SBMLs)):
		sorting_files(input_file_SBMLs[i],input_file_datas[i],analysis,fname,usesbml[i], which_species[i], parameter_change[i], init_condit[i])

def sorting_files(input_file_SBML, input_file_data, analysis, fname, usesbml, which_species, parameter_change, init_condit):
	# Used to remove the .xml at the end of the file if present to name directories
	input_file_SBML_name = input_file_SBML
	if input_file_SBML_name[-4:]==".xml":
		input_file_SBML_name = input_file_SBML_name[:-4]
	# Following set of if statements takes SBML files and depending on the way it needs to be changed carries out parsers to create new SBML files for each experiment
	
	if usesbml == True:
		# Creates CUDA code if local code not used
		# Sets the outpath for where CUDA code is stored
		if not(os.path.isdir("./"+fname+"/cudacodes_"+input_file_SBML_name)):
			os.mkdir(fname+"/cudacodes_"+input_file_SBML_name)
		outPath=fname+"/cudacodes_"+input_file_SBML_name
		# Depending on the way changes have been made to the SBML files only require certain versions
		input_files_SBML=[]

		if parameter_change == True and init_condit == True:
			# Creates SBML files corresponding to changes in parameters and initial conditions
			print "-----Creating SBML files for experiments-----"
			if not(os.path.isdir("./"+fname+"/exp_xml_"+input_file_SBML_name)):
				os.mkdir(fname+"/exp_xml_"+input_file_SBML_name)
			inPath = fname + "/exp_xml_" + input_file_SBML_name
			SBML_check.SBML_initialcond(nu=1,input_file=input_file_SBML,init_cond=input_file_data,outputpath=inPath)
			no_exp = 1 + SBML_check.SBML_reactionchanges(input_file_SBML, inPath,input_file_data,init_cond=True)
			print "-----Creating CUDA code-----"
			for i in range(0,no_exp):
				input_files_SBML.append("Exp_" + repr(i+1) + "_1.xml")
		elif parameter_change == False and init_condit == True:
			# Creates SBML files corresponding to changes in initial conditions but not parameters
			print "-----Creating SBML files for experiments-----"
			if not(os.path.isdir("./"+fname+"/exp_xml_"+input_file_SBML_name)):
				os.mkdir(fname+"/exp_xml_"+input_file_SBML_name)
			inPath = fname + "/exp_xml_" + input_file_SBML_name
			SBML_check.SBML_initialcond(nu=1,input_file=input_file_SBML,init_cond=input_file_data,outputpath=inPath)
			print "-----Creating CUDA code-----"
			input_files_SBML.append("Exp_" + repr(1) + "_1.xml")
		elif parameter_change == True and init_condit == False:
			# Creates SBML files corresponding to changes in parameters but not initial conditions
			print "-----Creating SBML files for experiments-----"
			if not(os.path.isdir("./"+fname+"/exp_xml_"+input_file_SBML_name)):
				os.mkdir(fname+"/exp_xml_"+input_file_SBML_name)
			inPath = fname + "/exp_xml_" + input_file_SBML_name
			copyfile(input_file_SBML,inPath + "/Exp_1.xml")
			no_exp = 1 + SBML_check.SBML_reactionchanges(input_file_SBML, inPath,input_file_data,init_cond=False)
			print "-----Creating CUDA code-----"
			for i in range(0,no_exp):
				input_files_SBML.append("Exp_" + repr(i+1) + "_1.xml")
		elif parameter_change == False and init_condit == False:
			# Creates one SBML file if only the species measured is changed
			print "-----Creating SBML files for experiments-----"
			if not(os.path.isdir("./"+fname+"/exp_xml_"+input_file_SBML_name)):
				os.mkdir(fname+"/exp_xml_"+input_file_SBML_name)
			inPath = fname + "/exp_xml_" + input_file_SBML_name	
			copyfile(input_file_SBML,inPath + "/Exp_1_1.xml")
			print "-----Creating CUDA code-----"
			input_files_SBML.append("Exp_" + repr(1) + "_1.xml")

		# Creates the required CUDA code if an SBML is used and CUDA code not provided
		cudacodecreater.cudacodecreater(input_files_SBML,inPath=inPath+"/",outPath=outPath)

		if not(os.path.isdir("./"+fname+"/input_xml_"+input_file_SBML_name)):
			os.mkdir(fname+"/input_xml_"+input_file_SBML_name)
		xml_out=fname+"/input_xml_"+input_file_SBML_name

		input_xml_files = os.listdir(inPath)
		print "-----Input XML file-----"
		input_file_parser.generateTemplate(input_xml_files, "input_xml", "summmary", input_file_data, inputpath = inPath, outputpath= xml_out)

	if usesbml == False:
		print "----Using local code-----"


main()

