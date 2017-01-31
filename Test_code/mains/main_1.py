#!/usr/bin/python2.5

#from numpy import *
#from numpy.random import *
#import abcsysbio
#import sys
#import re
#import time, os

#import cudasim
#import cudasim.SBMLParser as Parser
#import cudasim.Lsoda as Lsoda


#from pycuda import compiler, driver
#from pycuda import autoinit


#from abcsysbio import model_py
#from abcsysbio import model_cu
#from abcsysbio import model_c
#from abcsysbio import data
#from abcsysbio import input_output

import sys
import os
sys.path.insert(0, '/cluster/home/saw112/work/Test_code/error_checks')
#sys.path.insert(0, '/cluster/home/saw112/work/Test_code/abcsysbio_parser')


#import abcsysbio_parser
#from abcsysbio_parser import ParseAndWrite
#import generateTemplate
import error_check
import cudacodecreater
import SBML_check
#xmlModel="rep_test.xml"
#paramChange="test_data.txt"
#generateTemplate.generateTemplate([xmlModel],"input_file.xml","summary_file.txt")



#integrationType = "CUDA ODE"
#name = "test_model"
#ParseAndWrite.ParseAndWrite([xmlModel],[integrationType],[name],inputPath="",outputPath="",method=None)


#SBML_check.SBML_initialcond(xmlModel,[[1.0,2.0,3.0,4.0,5.0,6.0],[2.0,3.0,4.0,5.0,6.0,7.0]],names=["test_1.xml","test_2.xml"])
#SBML_check.SBML_initialcond(input_file=xmlModel,init_cond=paramChange,outputpath="_results_")
#SBML_check.SBML_reactionchanges(xmlModel,"_results_",paramChange)


def main():
	input_file_SBML, input_file_data, analysis, fname, usesbml, which_species, parameter_change, init_condit = error_check.input_checker(sys.argv,0)
	SBML_check.SBML_checker(input_file_SBML)
	if len(input_file_SBML) == 1:
		input_file_SBML = input_file_SBML[0]
	if usesbml == True and parameter_change == True and init_condit == True and type(input_file_SBML) == str:
		if not(os.path.isdir("./"+fname+"/exp_xml")):
			os.mkdir(fname+"/exp_xml")
		SBML_check.SBML_initialcond(nu=1,input_file=input_file_SBML,init_cond=input_file_data,outputpath=fname)
		SBML_check.SBML_reactionchanges(input_file_SBML,fname,input_file_data)
	elif usesbml == True and init_condit == True and type(input_file_SBML) == str:
		if not(os.path.isdir("./"+fname+"/exp_xml")):
			os.mkdir(fname+"/exp_xml")
		SBML_check.SBML_initialcond(input_file=input_file_SBML,init_cond=input_file_data,outputpath=fname)
	
	#if usesbml == True:
	#	if not(os.path.isdir("./"+fname+"/cudacodes")):
	#		os.mkdir(fname+"/cudacodes")
	#	outPath=fname+"/cudacodes"
	#	print "-----Creating CUDA code-----"
	#	cudacodecreater.cudacodecreater(input_file_SBML,inPath="",outPath=outPath)

#cudaCode="test_model.cu"
#timepoints=array(range(100+1),dtype=float32)*50.0/100.0
#params=[[2,10,1000,5]]
#species=[[0,1,0,0,0,0]]
#modeInstance=Lsoda.Lsoda(timepoints,cudaCode,1)
#result=modeInstance.run(params,species)
#print result

#xmlModel="rep_test.
main()

