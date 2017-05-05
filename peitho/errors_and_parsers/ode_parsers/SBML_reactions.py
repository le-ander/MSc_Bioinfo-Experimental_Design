import sys
from libsbml import *
import re
from shutil import copyfile

# A function that makes changes to the reactions
##(gets called by SBML_reactionchange)
##Arguments: 
##orig_str - is the original reaction
##orig_param - is the original parameter
##replacement - is the replacement for orig_param
def stringSearch(orig_str,orig_param,replacement):
	
	#Regular expressions for NOT alphanumerics and NOTspaces
	not_an = re.compile(r"[^A-Za-z0-9]")
	not_space = re.compile(r"[-\s]")

	#Builds up the replacement string as strings are immutable in Python
	replaced_string = ""

	#While loop travels along the length of orig_str
	i = 0
	while i < len(orig_str):
		#Whilst going along the length of the original string if a match is found for replacement then made here
		if orig_str[i:i+len(orig_param)] == orig_param and (not_an.search(orig_str[i+len(orig_param):i+len(orig_param)+1]) or not_space.search(orig_str[i+len(orig_param):i+len(orig_param)+1]) or len(orig_str[i+len(orig_param):i+len(orig_param)+1]) == 0 or len(orig_str[i+len(orig_param):i+len(orig_param)+1]) == 0) and (not_an.search(orig_str[i-1:i]) or not_space.search(orig_str[i-1:i]) or len(orig_str[i-1:i]) == 0 or len(orig_str[i-1:i]) == 0):
			#Adds replacement to the new reaction string
			replaced_string+=replacement
			#Shifts the pointer further up the original string
			i+=len(orig_param)
		#If no replacement made then go here 
		else:
			#Adds the current character to the new reaction string
			replaced_string+=orig_str[i]
			#Shifts the pointer to the adjacent character
			i+=1
	
	#Returns the new replaced string
	return replaced_string

# A function that creates the new SBML files based on the original
##(gets called by sorting_files)
##Arguments: 
##input_files - original SBML file
##inpath - where the SBML file is in the directory
##param_changes - contains the input file containing the experimental changes
def SBML_reactionchanges(input_file, inpath="", fname="", param_changes=""):
	
	#Reads in param_changes, loads in data
	change_params=open(inpath+"/"+param_changes,'r')
	data = change_params.read()
	change_params.close()

	#Finds where the changes have been made in data
	start_index = re.compile(r'>Parameter - Experiment (\d+)\s*\n(.+?)\n<Parameter - Experiment (\d+)', re.DOTALL)
	#lines contains a list of tuples containing the changes for each experiment
	lines = start_index.findall(data)
	#exp_list contains the number associated to each experiment at beginning
	exp_list = [int(lines[i][0]) for i in range(0,len(lines))]
	##exp_list_end contains the number associated to each experiment at end
	exp_list_end = [int(lines[i][2]) for i in range(0,len(lines))]
	#Unpacks the original lines to only hold the changes for each experiment
	lines = [lines[i][1] for i in range(0,len(lines))]

	#Checking if parameters in the input file are properly defined
	if exp_list_end != exp_list or sum(exp_list)!=len(exp_list)*(len(exp_list)+1)/2:
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

	start_point = 0
	#Unchanged SBML has to the be the first experiment if this is the case
	#Just copies the SBML file into the output folder where SBML files are stored
	if lines[0]=='Unchanged':
		copyfile(inpath+"/"+input_file,fname + "/Exp_1.xml")
		start_point = 1

	#For loop iterates over each experiments
	for i, start in enumerate(lines[start_point:]):
		#splits each line into its components and then puts them into lists
		line = start.split("\n")
		param_to_change=[] #holds which parameters to change
		mult_fact=[] #holds by what factor to change that parameter
		param_reaction=[] #holds which reaction the change needs to be made to
		#For loop fills the lists defined above
		for end in line:
			el_temp=end.split()
			param_to_change.append(el_temp[0])
			mult_fact.append(el_temp[1])
			param_reaction.append(int(el_temp[2]))
		#Function makes the appropriate changes to the reactions in the SBML files
		SBML_reactionchange(input_file, param_to_change,mult_fact,param_reaction,exp_list[i+start_point], dest=fname, inpath = inpath)

	#returns the number of experiments
	return len(lines)

# A function that changes the reactions to the original SBML file
##(gets called by SBML_reactionchanges)
##Arguments: 
##input_files - original SBML file
##param_to_change - list of parameters to change
##mult_fact - list containing the factor by which to change the parameter
##param_reaction - list of which reactions that need changing for the associated parameter
##nu - the number given to the experiment
##dest - output path
##inpath - input path
def SBML_reactionchange(input_file, param_to_change, mult_fact, param_reaction, nu, dest="", inpath=""):
	#Reads in original SBML file
	reader = SBMLReader()
	SBML_master = reader.readSBML(inpath+"/"+input_file)

	#For loop over the changes that need to be made
	for i, reaction_no in enumerate(param_reaction):
		#Reads in the reaction that needs to be changed
		reaction = formulaToString(SBML_master.getModel().getListOfReactions()[reaction_no-1].getKineticLaw().getMath())
		#temp holds the change that needs to be made
		temp = r"{string1} * {string2}".format(string1=mult_fact[i],string2=param_to_change[i])
		#Calls stringSearch to make the change to the reaction
		temp = stringSearch(reaction,param_to_change[i],temp)
		#Sets the new reaction to the SBML object
		SBML_master.getModel().getListOfReactions()[reaction_no-1].getKineticLaw().setMath(parseFormula(temp))

	#Writes the final new SBML file
	if dest=="":
		#Writes it to the current working directory if dest=""
		writeSBML(SBML_master, "Exp_" + repr(nu) + ".xml")
	else:
		#Writes it to the output destination if provided
		writeSBML(SBML_master, "./" + dest + "/Exp_" + repr(nu) + ".xml")