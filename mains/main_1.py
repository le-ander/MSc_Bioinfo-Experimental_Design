import sys
import os
from shutil import copyfile
import re
sys.path.insert(1, '../error_checks')
sys.path.insert(1, '../simulations')
sys.path.insert(1, '../mutualInfo')
sys.path.insert(1, '../abc-sysbio')
sys.path.insert(1, '../cudasim')


import simulation_functions
import organiser
import error_check
import SBML_check
import input_file_parser_new_2
import parse_infoEnt_new_2
import getEntropy1
import getEntropy2
import getEntropy3
import cudacodecreater
import plotbar
from numpy import *
import time


def main():
	# Calls error_checker - reads in command line arguments and does some basic error checks
	input_file_SBMLs, input_file_datas, analysis, fname, usesbml, iname = error_check.input_checker(sys.argv,0)
	# Calls SBML_checker - checks all the SBML files that have been inputted
	SBML_check.SBML_checker([iname+"/"+input_file_SBMLs[i] for i, value in enumerate(usesbml) if value=="0"])
	# Unpacks the following four command line arguments - each element corresponds to each SBML and data file pair
	
	usesbml=[not(bool(int(i))) for i in list(usesbml)]
	#parameter_change=[int(i) for i in list(parameter_change)]
	#init_condit=[int(i) for i in list(init_condit)]
	# Calls sorting_files which creates new SBML files for new experiments and creates CUDA code from SBML files if necessary


	if analysis != 2:
		count = 0
		for i in range(0,len(input_file_SBMLs)):
			random.seed(123)
			if usesbml[i]==True:
				sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, input_file_data = input_file_datas[count])
				count += 1
			else:
				sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname)
	else:
		####Reference model
		count = 0
		if usesbml[0] == True:
			random.seed(123)
			ref_model = sorting_files(input_file_SBMLs[0],analysis,fname,usesbml[0], iname, input_file_data = input_file_datas[count])
			count += 1 
		else:
			ref_model = sorting_files(input_file_SBMLs[0],analysis,fname,usesbml[0], iname)
		####Not reference models
		for i in range(1,len(input_file_SBMLs)):
			random.seed(123)
			if usesbml[i] == True:
				sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, refmod = ref_model,input_file_data = input_file_datas[count])
				count += 1 
			else:
				sorting_files(input_file_SBMLs[i],analysis,fname,usesbml[i], iname, refmod = ref_model)

def sorting_files(input_file_SBML, analysis, fname, usesbml, iname, refmod="", input_file_data = ""):
	# Used to remove the .xml at the end of the file if present to name directories
	input_file_SBML_name = input_file_SBML
	if input_file_SBML_name[-4:]==".xml":
		input_file_SBML_name = input_file_SBML_name[:-4]
	# Following set of if statements takes SBML files and depending on the way it needs to be changed carries out parsers to create new SBML files for each experiment

	if not(os.path.isdir("./"+fname+"/cudacodes")):
			os.mkdir(fname+"/cudacodes")

	if usesbml == True:
		# Creates CUDA code if local code not used
		# Sets the outpath for where CUDA code is stored
		if not(os.path.isdir("./"+fname+"/cudacodes/cudacodes_"+input_file_SBML_name)):
			os.mkdir(fname+"/cudacodes/cudacodes_"+input_file_SBML_name)
		outPath=fname+"/cudacodes/cudacodes_"+input_file_SBML_name
		# Depending on the way changes have been made to the SBML files only require certain versions
		input_files_SBML=[]

		print "-----Creating SBML files for experiments-----"
		if not(os.path.isdir("./"+fname+"/exp_xml")):
			os.mkdir(fname+"/exp_xml")
		if not(os.path.isdir("./"+fname+"/exp_xml/exp_xml_"+input_file_SBML_name)):
			os.mkdir(fname+"/exp_xml/exp_xml_"+input_file_SBML_name)
		inPath = fname + "/exp_xml/exp_xml_" + input_file_SBML_name

		#if parameter_change == True:
			# Creates SBML files corresponding to changes in parameters and initial conditions
		no_exp = SBML_check.SBML_reactionchanges(input_file_SBML, iname, inPath,input_file_data)
		print "-----Creating CUDA code-----"
		for i in range(0,no_exp):
			input_files_SBML.append("Exp_" + repr(i+1) + ".xml")
		#else:
			# Creates SBML files corresponding to changes in initial conditions but not parameters
			#SBML_check.SBML_initialcond(nu=1,input_file=input_file_SBML,init_cond=input_file_data,outputpath=inPath,indicate=True)
			#copyfile(input_file_SBML,inPath + "/Exp_1.xml")
			#print "-----Creating CUDA code-----"
			#input_files_SBML.append("Exp_" + repr(1) + ".xml")
		'''
		elif parameter_change == True and init_condit == False:
			# Creates SBML files corresponding to changes in parameters but not initial conditions
			#copyfile(input_file_SBML,inPath + "/Exp_1_1.xml")
			no_exp = SBML_check.SBML_reactionchanges(input_file_SBML, inPath,input_file_data,init_cond=False)
			print "-----Creating CUDA code-----"
			for i in range(0,no_exp):
				input_files_SBML.append("Exp_" + repr(i+1) + "_1.xml")
		elif parameter_change == False and init_condit == False:
			# Creates one SBML file if only the species measured is changed
			copyfile(input_file_SBML,inPath + "/Exp_1_1.xml")
			print "-----Creating CUDA code-----"
			input_files_SBML.append("Exp_" + repr(1) + "_1.xml")
		'''
		# Creates the required CUDA code if an SBML is used and CUDA code not provided

		cudacodecreater.cudacodecreater(input_files_SBML,inPath=inPath+"/",outPath=outPath)

		if not(os.path.isdir("./"+fname+"/input_xml")):
			os.mkdir(fname+"/input_xml")
		if not(os.path.isdir("./"+fname+"/input_xml/input_xml_"+input_file_SBML_name)):
			os.mkdir(fname+"/input_xml/input_xml_"+input_file_SBML_name)
		xml_out=fname+"/input_xml/input_xml_"+input_file_SBML_name

		exp_xml_files = os.listdir(inPath)
		
		print "-----Input XML file-----"
		comb_list = input_file_parser_new_2.generateTemplate(exp_xml_files, "input_xml", "summmary", input_file_data, inpath = inPath, outpath= xml_out, iname=iname)

		input_xml="/input_xml"

	elif usesbml == False:
		outPath=iname
		inPath=""
		xml_out=iname
		input_xml="/"+input_file_SBML_name
		comb_list = []

	print "-----Creating object from input XML file-----"
	sbml_obj = parse_infoEnt_new_2.algorithm_info(xml_out+input_xml+".xml", 0, comb_list)
	sbml_obj.getpairingCudaICs()
	print "-----Sampling from prior-----"
	sbml_obj.getAnalysisType(analysis)
	if sbml_obj.analysisType != 2 or refmod == "":
		sbml_obj.THETAS(inputpath=iname, usesbml=usesbml)
	else:
		sbml_obj.copyTHETAS(refmod)

	print "-----Running CUDA-Sim-----"
	#cudasim_run = simulation_functions.run_cudasim(sbml_obj,inpath=outPath)
	#if sbml_obj.analysisType != 2:
	cudasim_run = simulation_functions.run_cudasim(sbml_obj,inpath=outPath)
	#elif sbml_obj.analysisType == 2 and refmod == "":
	#	cudasim_run = simulation_functions.run_cudasim(sbml_obj,inpath=outPath, analysis = 1)
	#elif sbml_obj.analysisType == 2 and refmod != "":
	#	cudasim_run = simulation_functions.run_cudasim(sbml_obj,inpath=outPath, analysis = 2)

	print "-----Calculating scaling factor-----"
	if sbml_obj.analysisType != 2:
		sbml_obj.scaling()
	else:
		sbml_obj.scaling_ge3()
	#print sbml_obj.scale
	if sbml_obj.analysisType == 0:
		time1=time.time()
		MutInfo1=getEntropy1.run_getEntropy1(sbml_obj)
		time2=time.time()
		print "MutInfo1 runtime", time2-time1
		print sbml_obj.prior
		plotbar.plotbar(MutInfo1, sbml_obj.name ,sbml_obj.nmodels ,0)
	elif sbml_obj.analysisType == 1:
		MutInfo2=getEntropy2.run_getEntropy2(sbml_obj)
		plotbar.plotbar(MutInfo2, sbml_obj.name ,sbml_obj.nmodels ,1)
	elif sbml_obj.analysisType == 2 and refmod == "":
		return sbml_obj
	elif sbml_obj.analysisType == 2 and refmod != "":
		getEntropy3.run_getEntropy3(sbml_obj, refmod)
		


	#sbml_obj.print_info()
	#pairings = organiser.organisePairings(sbml_obj, comb_list)
	#print [x.shape for x in sbml_obj.cudaout]
	#print sum(sbml_obj.cudaout[0][:,0:sbml_obj.cudaout[0].shape[1],:],axis=2)

	#print sbml_obj.cudaout
	#print sbml_obj.speciesSample
	#print set(sbml_obj.cuda)
	#print comb_list



main()

#info_new = error_check.parse_infoEnt.algorithm_info("input_file_repressilator.xml", 0)
#print info_new.globalnparameters
#p = obtain_thetas.THETAS(info_new, sampleFromPost = False)
#s = obtain_thetas.SPECIES(info_new)
#p,s=obtain_thetas.THETAS(info_new, sampleGiven = True, sampleFromPost= "data_1.txt", weight= "w_1.txt", analysisType = 1, N1 = 2, N3 = 3, parameter_i = [0,3])

#obtain_thetas.ThetasGivenI(info_new,p,s,[0,3],[1],2,3)

#print p[1,:]
#print s[1,:]
#print concatenate((p,s),axis=1).shape[1]
