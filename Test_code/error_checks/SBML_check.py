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

def SBML_initialcond(nu="", input_file="", SBML_obj="", init_cond="", outputpath=""):
	if type(init_cond)!=list:
		init_cond = open(init_cond,'r')
		init_cond = [i.rstrip() for i in init_cond.readlines()]

	start_index = re.compile(r'>Initial Conditions')
	end_index = re.compile(r'<Initial Conditions')
	for i, line in enumerate(init_cond):
		if start_index.search(line):
			index_start = i
		if end_index.search(line):
			index_end = i
	init_lines = init_cond[index_start+1:index_end]
	for i, values in enumerate(init_lines):
		init_lines[i]=[float(j) for j in init_lines[i].split(" ")] 

	if SBML_obj!="":
		if outputpath == "":
			writeSBML(SBML_obj,"test" + "_" + repr(nu) + "_" + repr(1) + ".xml")
		else:
			writeSBML(SBML_obj,"./" + outputpath + "/exp_xml/Exp" + "_" + repr(nu) + "_" + repr(1) + ".xml")
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
				writeSBML(SBML_obj,"./" + outputpath + "/exp_xml/Exp" + "_" + repr(nu) + "_" + repr(i+2) + ".xml")
	else:
		reader = SBMLReader()
		SBML_master = reader.readSBML(input_file)
		if outputpath == "":
			writeSBML(SBML_master,"test" + "_" + repr(nu) + "_" + repr(1) + ".xml")
		else:
			writeSBML(SBML_master,"./" + outputpath + "/exp_xml/Exp" + "_" + repr(1) + "_" + repr(1) + ".xml")
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
				writeSBML(SBML_master,"./" + outputpath + "/exp_xml/Exp" + "_" + repr(nu) + "_" + repr(i+2) + ".xml")

def SBML_reactionchange(input_file, lines, param_to_change, mult_fact, param_reaction, nu, init_condit=False, dest=""):
	reader = SBMLReader()
	SBML_master = reader.readSBML(input_file)
	for i, reaction_no in enumerate(param_reaction):
		reaction = formulaToString(SBML_master.getModel().getListOfReactions()[reaction_no-1].getKineticLaw().getMath())
		temp = re.sub(param_to_change[i],mult_fact[i]+" * "+param_to_change[i],reaction)
		SBML_master.getModel().getListOfReactions()[reaction_no-1].getKineticLaw().setMath(parseFormula(temp))
	if init_condit == False:
		if dest=="":
			writeSBML(SBML_master, "Exp" + repr(nu) + ".xml")
		else:
			writeSBML(SBML_master, "./" + dest + "/exp_xml/Exp" + repr(nu) + ".xml")
	else:
		SBML_initialcond(nu, input_file,SBML_master,lines,dest)


def SBML_reactionchanges(input_file, fname="", param_changes=""):
	change_params=open(param_changes,'r')
	start_index = re.compile(r'>Parameter - Experiment (\d+)')
	end_index = re.compile(r'<Parameter - Experiment (\d+)')
	lines = [i.rstrip() for i in change_params.readlines()]
	start_list = []
	end_list = []
	exp_list = []
	for i, line in enumerate(lines):
		if start_index.search(line):
			start_list.append(i)
			exp_list.append(int(start_index.search(line).group(1)))
		if end_index.search(line):
			end_list.append(i)

	for i, start in enumerate(start_list):
		start_temp = start
		stop_temp = end_list[i]
		sub_lines = [j for j in lines[start:stop_temp]]
		sub_lines = [j for j in sub_lines if j!='']
		sub_lines = sub_lines[1:]
		param_to_change=[]
		mult_fact=[]
		param_reaction=[]                                                                                     
		for element in sub_lines:
			el_temp=element.split(' ')
			param_to_change.append(el_temp[0])
			mult_fact.append(el_temp[1])

			param_reaction.append(int(el_temp[2]))
		SBML_reactionchange(input_file, lines, param_to_change,mult_fact,param_reaction,exp_list[i],init_condit=True, dest=fname)