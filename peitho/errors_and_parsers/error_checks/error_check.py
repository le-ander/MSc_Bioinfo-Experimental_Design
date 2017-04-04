import sys
import os

# A function that prints the flag options when run in the command line
##(input_checker)
##Has no arguments
def printOptions():

	print "\nList of possible options:"

	print "\n Input options:"
	print "-i1\t--infile_SBML\t declaration of the input SBML file or input.xml file if using local code. This input file has to be provided to run the program!"
	print "-i2\t--infile_data\t declaration of the input data file for the associated SBML file. This input file has to be provided to run the program if using an SBML file!"
	print "-a\t--analysis\t declaration of the type of analysis to be carried out. This input file has to be provided to run the program!"
	print "\t0: prediction for all parameters"
	print "\t1: prediction for a subset of parameters"
	print "\t2: prediction for an experiment"
	print "-lc\t--localcode\t do not import model from sbml intead use a local .py, .hpp/.cpp or .cu file. This requires an input.xml file to be run."
	print "-if\t--infolder\t is the location as to where the input data is eg -if=./relative/path/to/folder. All input data must be in here. Cannot search this location recursively inside sub-directories."

	print "\n Output options:"
	print "-of\t--outfolder\t write results to folder eg -of=./relative/path/to/folder (default is _results_ in current directory)"

	print "\n Other options:"
	print "-tc\t--template_create\t\t used to create the input xml and cuda code file based on an SBML file"
	print "-mc\t--memory_check\t\t used to estimate the memory requirements for running peitho with certain inputs"
	print "-h\t--help\t\t print this list of options."

	print "\n Examples:"
	print "For running experimental selection for inference of all parameters with the high level pipeline:"
	print "\tpeitho -i1 SBML_file -i2 input_data_file -a 0 -lc 0 [-if=./relative/path/to/input/folder] [-of=./relative/path/to/output/folder]\n"
	print "For running experimental selection for inference on a subset of parameters with the low level pipeline:"
	print "\tpeitho -i1 input_xml_file -a 1 -lc 1 [-if=./relative/path/to/input/folder] [-of=./relative/path/to/output/folder]\n"
	print "For running experimental selection for prediction with the high level pipeline:"
	print "\tpeitho -i1 SBML_file_ref SBML_file_exp -i2 input_data_file_ref input_data_file_exp -a 2 -lc 00 [-if=./relative/path/to/input/folder] [-of=./relative/path/to/output/folder]"
	print "\n"


# A function that sorts out the command line inputs
##(gets called by main)
##Arguments:
##sys_arg - list of the command line arguments
##analysis - the approach we want to carry out 1, 2, or 3
def input_checker(sys_arg):
	#Defines the default for variables
	file_exist_SBML=False #for when the SBML or input.xml file exists
	file_exist_data=False #whether the assocaited input file for an SBML file is also provided
	usesbml=True #whether SBML or local code used
	fname = "_results_" #string for the output file
	iname = "" #string for where the input files are
	analysis = 3 #sets the type of approach
	input_file_data=[] #list containing associated input files for SBML files
	template_creator=False
	memory_check=False

	#For loop cycles over the command line arguments
	for i in range(1,len(sys_arg)):
		if sys_arg[i].startswith('--'):
			#If help flag is used calls printOptions()
			option = sys_arg[i][2:]
			if option == 'help':
				printOptions()
				sys.exit()
			#Sets analysis type
			elif option == 'analysis':
				analysis = int(sys_arg[i+1])
				if analysis == 0:
					print "Type of Analysis: Prediction for all parameters\n"
				elif analysis == 1:
					print "Type of Analysis: Prediction for a subset of parameters\n"
				elif analysis == 2:
					print "Type of Analysis: Prediction of experiment\n"
			#Sets whether local code or SBMLs used
			elif option == 'localcode' :
				usesbml = sys_arg[i+1]
			#Sets output folder
			elif option[0:10] == 'outfolder=' :
				fname = option[10:]
				print "Output file destination: " + fname + "\n"
			#Reads in the SBML/input.xml files
			elif option == 'infile_SBML':
				input_file_SBML=sys_arg[i+1:]
				file_exist_SBML=True
				print "Input SBML files: "
				keep = 0
				for j in input_file_SBML:
					if not(j.startswith('-')):
						keep += 1
						print "\t" + j
					else:
						break
				input_file_SBML=input_file_SBML[:keep]
				print ""
			#Reads in input files associated to SBML files
			elif option == 'memory_check':
				memory_check=True
			elif option == 'infile_data':
				input_file_data=sys_arg[i+1:]
				file_exist_data=True
				print "Input data files: "
				keep = 0
				for j in input_file_data:
					if not(j.startswith('-')):
						keep += 1
						print "\t" + j
					else:
						break
				input_file_data=input_file_data[:keep]
				print ""
			#Sets directory with input files
			elif option == 'temple_check':
				template_creator=True
				usesbml = "0"
				analysis = "0"
				input_file_data=None

			elif option[0:9] == "infolder=":
				iname = option[9:]
				print "Input file destination: " + iname + "\n"
			#If flag not recognised calls printOptions()
			elif not(sys_arg[i-1][2:] == 'infile_SBML'):
				print "\nERROR: unknown option "+sys_arg[i]
				printOptions()
				sys.exit()

		elif sys_arg[i].startswith('-'):
			option = sys_arg[i][1:]
			#If help flag is used calls printOptions()
			if option == 'h':
				printOptions()
				sys.exit()
			#Sets type of approach
			elif option == 'a':
				analysis = int(sys_arg[i+1])
				if analysis == 0:
					print "Type of Analysis: Prediction for all parameters\n"
				elif analysis == 1:
					print "Type of Analysis: Prediction for a subset of parameters\n"
				elif analysis == 2:
					print "Type of Analysis: Prediction of experiment\n"
			#Sets whether local code or SBMLs used
			elif option == 'lc' :
				usesbml = sys_arg[i+1]
			#Sets output folder
			elif option[0:3] == 'of=' :
				fname = option[3:]
				print "Output file destination: " + fname + "\n"
			#Reads in list of SBML/input.xml files
			elif option == 'i1':
				input_file_SBML=sys_arg[i+1:]
				file_exist_SBML=True
				print "Input SBML files: "
				keep = 0
				for j in input_file_SBML:
					if not(j.startswith('-')):
						keep += 1
						print "\t" + j
					else:
						break
				input_file_SBML=input_file_SBML[:keep]
				print ""
			#Reads in list of input files associated to SBML files
			elif option == 'i2':
				input_file_data=sys_arg[i+1:]
				file_exist_data=True
				print "Input data files: "
				keep = 0
				for j in input_file_data:
					if not(j.startswith('-')):
						keep += 1
						print "\t" + j
					else:
						break
				input_file_data=input_file_data[:keep]
				print ""
			#Sets the input folder
			elif option[0:3] == "if=":
				iname = option[3:]
				print "Input file destination: " + iname + "\n"
			elif option == 'tc':
				template_creator=True
				usesbml = "0"
				analysis = "0"
				input_file_data=None

				print "Create template from SBML file"
			elif option == 'mc':
				memory_check=True

			#If an unrecognised flag is called and calls printOptions
			elif not(sys_arg[i-1][2:] == 'i1'):
				print "\nERROR: unknown option "+sys_arg[i]
				printOptions()
				sys.exit()

	if template_creator==False or memory_check == False:
		#Checks whether an SBML or input.xml file was passed
		if file_exist_SBML == False:
			print "\nNo input_file_SBML is given!\nUse: \n\t-i1 'inputfile' \nor: \n\t--infile_SBML 'inputfile' \n"
			sys.exit()

		#Checks if an SBML file has been given and whether it has an associated input file
		if "0" in list(usesbml) and len([x for x in list(usesbml) if x=="0"]) != len(input_file_data):
			print "\nSBML file is being used but no associated input file given!\nUse: \n\t-i2 'inputfile' \nor: \n\t--infile_data 'inputfile' \n"
			sys.exit()

		#Check whether an approach has been given
		if analysis not in [0,1,2]:
			print "\nNo analysis type is given!\nUse: \n\t-a 'analysis type' \nor: \n\t --analysis 'analysis type' \n"
			sys.exit()

	#Sets up the output folder
	if not(os.path.isdir("./"+fname)):
		os.mkdir(fname)

	return input_file_SBML, input_file_data, analysis, fname, usesbml, iname, template_creator, memory_check
