import sys
from libsbml import *
import re
from shutil import copyfile

def stringSearch(orig_str,orig_param,replacement):
	print "here"
	not_an = re.compile(r"[^A-Za-z0-9\s]+")
	replaced_string = ""

	if orig_str[:len(orig_param)] == orig_param and not_an.search(orig_str[len(orig_param)+1]):
		replaced_string+=replacement
	else:
		replaced_string+=orig_str[:len(orig_param)]


	for i, param in enumerate(orig_str[len(orig_param):-1]):
		if orig_str[i:i+len(orig_param)] == orig_param and not_an.search(orig_str[i+len(orig_param)+1]) and not_an.search(orig_str[i-1]):
			replaced_string+=replacement
		else:
			replaced_string+=param

	print replaced_string
	
	sys.exit()

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

def SBML_reactionchange(input_file, lines, param_to_change, mult_fact, param_reaction, nu, dest="", inpath=""):
	reader = SBMLReader()
	SBML_master = reader.readSBML(inpath+"/"+input_file)
	for i, reaction_no in enumerate(param_reaction):
		reaction = formulaToString(SBML_master.getModel().getListOfReactions()[reaction_no-1].getKineticLaw().getMath())
		a = re.compile(r"[^A-Za-z0-9]+({string1})[^A-Za-z0-9]+".format(string1=param_to_change[i]))
		#temp = a.sub("[^A-Za-z0-9]+(" + param_to_change[i] + ")[^A-Za-z0-9]+",mult_fact[i]+" * "+param_to_change[i] + " ",reaction + " ")
		#temp = a.sub(mult_fact[i]+" * "+param_to_change[i] + " ",reaction + " ")
		temp = r"{string1} * {string2}".format(string1=mult_fact[i],string2=param_to_change[i])
		print reaction
		print ""
		print param_to_change[i]
		print ""
		print temp
		temp = stringSearch(reaction,param_to_change[i],temp)
		#myRe.sub(r'\1"noversion"\3', val)
		SBML_master.getModel().getListOfReactions()[reaction_no-1].getKineticLaw().setMath(parseFormula(temp[:-1]))
		
	#if init_condit == False:
	if dest=="":
		writeSBML(SBML_master, "Exp_" + repr(nu) + ".xml")
	else:
		writeSBML(SBML_master, "./" + dest + "/Exp_" + repr(nu) + ".xml")
	#else:
	#	SBML_initialcond(nu, input_file,SBML_master,lines,dest)


def SBML_reactionchanges(input_file, inpath="", fname="", param_changes=""):
	change_params=open(inpath+"/"+param_changes,'r')
	data = change_params.read()
	#change_params.seek(0)
	#all_lines = [i.rstrip() for i in change_params.readlines()]
	change_params.close()

	start_index = re.compile(r'>Parameter - Experiment (\d+)\s*\n(.+?)\n<Parameter - Experiment (\d+)', re.DOTALL)
	lines = start_index.findall(data)

	exp_list = [int(lines[i][0]) for i in range(0,len(lines))]
	lines = [lines[i][1] for i in range(0,len(lines))]

	start_point = 0
	if lines[0]=='Unchanged':
		copyfile(inpath+"/"+input_file,fname + "/Exp_1.xml")
		start_point = 1

	for i, start in enumerate(lines[start_point:]):
		line = start.split("\n")
		param_to_change=[]
		mult_fact=[]
		param_reaction=[]  
		for end in line:
			el_temp=end.split(' ')
			param_to_change.append(el_temp[0])
			mult_fact.append(el_temp[1])
			param_reaction.append(int(el_temp[2]))
		
		print param_to_change
		print mult_fact
		print param_reaction

		SBML_reactionchange(input_file, data, param_to_change,mult_fact,param_reaction,exp_list[i+start_point], dest=fname, inpath = inpath)

	return len(lines)

'''
def SBML_initialcond(nu="", input_file="", SBML_obj="", init_cond="", outputpath="",indicate=False):
	if indicate==True:
		data = open(init_cond,'r')
		init_cond = data.read()
		data.close()

	start_index = re.compile(r'>Initial Conditions (\d+)\s*\n(.+?)\n<Initial Conditions \d+.+?', re.DOTALL)
	init_lines = start_index.findall(init_cond)
	exp_list = [int(init_lines[i][0]) for i in range(0,len(init_lines))]
	init_lines = [init_lines[i][1] for i in range(0,len(init_lines))]

	#print init_lines

	#for i, ic in enumerate(init_lines):
	#	init_lines[i] = [float(j) for j in ic.split(" ")]

	for i in range(len(init_lines)):

		ic = init_lines[i].split("\n")
		ic = [i.split(" ") for i in ic]

		print ic

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
'''