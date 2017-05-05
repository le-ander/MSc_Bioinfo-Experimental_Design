import sys
from libsbml import *
import re
from shutil import copyfile

# A function that does a simple check on any SBML files
##(gets called by main)
##Arguments: 
##input_files - list of the SBML files used
def SBML_checker(input_files):
	#Counter for number of errors
	tot_errors=0

	#Reads each SBML file in a for loop and detects whether there are any errors in them
	for i in input_files:
		reader = SBMLReader()
		document = reader.readSBML(i)
		no_errors = document.getNumErrors()
		#Prints any associated errors
		if no_errors != 0:
			print "ERROR: SBML file - " + i + " - has the following errors\n"
			document.printErrors()
		tot_errors += no_errors
	#If any errors will exit the program
	if tot_errors > 0:
		sys.exit()
