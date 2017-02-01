import sys
from libsbml import *
import re

def SBML_checker(input_files):
	tot_errors=0
	for i in input_files:
		reader = SBMLReader()
		document = reader.readSBML(i)
		no_errors = document.getNumErrors()
		if no_errors != 0:
			document.printErrors()
		tot_errors += no_errors
	if tot_errors > 0:
		sys.exit()

def SBML_initialcond(nu="", input_file="", SBML_obj="", init_cond="", outputpath="",indicate=False):
	if indicate==True:
		data = open(init_cond,'r')
		init_cond = data.read()
		data.close()

	start_index = re.compile(r'>Initial Conditions\s*\n(.+?)\n<Initial Conditions+?', re.DOTALL)
	init_lines = start_index.findall(init_cond)
	init_lines = init_lines[0].split("\n")
	for i, ic in enumerate(init_lines):
		init_lines[i] = [float(j) for j in ic.split(" ")]

	if SBML_obj!="":
		if outputpath == "":
			writeSBML(SBML_obj,"test" + "_" + repr(nu) + "_" + repr(1) + ".xml")
		else:
			writeSBML(SBML_obj,"./" + outputpath + "/Exp" + "_" + repr(nu) + "_" + repr(1) + ".xml")
		for i in range(0,len(init_lines)):
			if sum([SBML_obj.getModel().getListOfSpecies()[j].isSetInitialAmount() for j in range(0,len(init_lines[i]))]) == len(init_lines[i]):
				for j, k in enumerate(init_lines[i]):
					SBML_obj.getModel().getListOfSpecies()[j].setInitialAmount(k) 
			else:
				for j, k in enumerate(init_lines[i]):
					SBML_obj.getModel().getListOfSpecies()[j].setInitialConcentration(k)
			if outputpath == "":
				writeSBML(SBML_obj,"test" + "_" + repr(nu) + "_" + repr(i+2) + ".xml")
			else:
				writeSBML(SBML_obj,"./" + outputpath + "/Exp" + "_" + repr(nu) + "_" + repr(i+2) + ".xml")
	else:
		reader = SBMLReader()
		SBML_master = reader.readSBML(input_file)
		if outputpath == "":
			writeSBML(SBML_master,"test" + "_" + repr(nu) + "_" + repr(1) + ".xml")
		else:
			writeSBML(SBML_master,"./" + outputpath + "/Exp" + "_" + repr(1) + "_" + repr(1) + ".xml")
		for i in range(0,len(init_lines)):
			if sum([SBML_master.getModel().getListOfSpecies()[j].isSetInitialAmount() for j in range(0,len(init_lines[i]))]) == len(init_lines[i]):
				for j, k in enumerate(init_lines[i]):
					SBML_master.getModel().getListOfSpecies()[j].setInitialAmount(k) 
			else:
				for j, k in enumerate(init_lines[i]):
					SBML_master.getModel().getListOfSpecies()[j].setInitialConcentration(k)
			if outputpath == "":
				writeSBML(SBML_master,"test" + "_" + repr(nu) + "_" + repr(i+2) + ".xml")
			else:
				writeSBML(SBML_master,"./" + outputpath + "/Exp" + "_" + repr(nu) + "_" + repr(i+2) + ".xml")

def SBML_reactionchange(input_file, lines, param_to_change, mult_fact, param_reaction, nu, init_condit=False, dest=""):
	reader = SBMLReader()
	SBML_master = reader.readSBML(input_file)
	for i, reaction_no in enumerate(param_reaction):
		reaction = formulaToString(SBML_master.getModel().getListOfReactions()[reaction_no-1].getKineticLaw().getMath())
		temp = re.sub(param_to_change[i] + " ",mult_fact[i]+" * "+param_to_change[i] + " ",reaction + " ")
		SBML_master.getModel().getListOfReactions()[reaction_no-1].getKineticLaw().setMath(parseFormula(temp[:-1]))
	if init_condit == False:
		if dest=="":
			writeSBML(SBML_master, "Exp_" + repr(nu) + "_1" + ".xml")
		else:
			writeSBML(SBML_master, "./" + dest + "/Exp_" + repr(nu) + "_1.xml")
	else:
		SBML_initialcond(nu, input_file,SBML_master,lines,dest)


def SBML_reactionchanges(input_file, fname="", param_changes="",init_cond=False):
	change_params=open(param_changes,'r')
	data = change_params.read()
	#change_params.seek(0)
	#all_lines = [i.rstrip() for i in change_params.readlines()]
	change_params.close()

	start_index = re.compile(r'>Parameter - Experiment (\d+)\s*\n(.+?)\n<Parameter - Experiment (\d+)+?', re.DOTALL)
	lines = start_index.findall(data)
	exp_list = [int(lines[i][0]) for i in range(0,len(lines))]
	lines = [lines[i][1] for i in range(0,len(lines))]

	for i, start in enumerate(lines):
		line = start.split("\n")
		param_to_change=[]
		mult_fact=[]
		param_reaction=[]  
		for end in line:
			el_temp=end.split(' ')
			param_to_change.append(el_temp[0])
			mult_fact.append(el_temp[1])
			param_reaction.append(int(el_temp[2]))
		SBML_reactionchange(input_file, data, param_to_change,mult_fact,param_reaction,exp_list[i],init_condit=init_cond, dest=fname)

	return len(lines)