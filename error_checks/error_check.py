import sys
import os
#import parse_infoEnt_new_2

def printOptions():
	
	print "\nList of possible options:"

	print "\n Input options:"
	print "-i1\t--infile_SBML\t declaration of the input SBML file. This input file has to be provided to run the program!"
	print "-i2\t--infile_data\t declaration of the input data file. This input file has to be provided to run the program!"
	print "-a\t--analysis\t declaration of the type of analysis to be carried out. This input file has to be provided to run the program!"
	print "\t0: prediction for all parameters"
	print "\t1: prediction for a subset of parameters"
	print "\t2: prediction for an experiment"
	print "-lc\t--localcode\t do not import model from sbml intead use a local .py, .hpp/.cpp or .cu file"
	print "-s\t--species\t to specify whether you are measuring different species in each experiment"
	print "-p\t--params\t to specify whether you are changing the parameters by some multiplicative factor in each experiment"
	print "-ic\t--init\t to specify whether you are changing the initial conditions of each experiment"

	print "\n Output options:"  
	print "-of\t--outfolder\t write results to folder eg -of=/full/path/to/folder (default is _results_ in current directory)"
	
	print "\n-h\t--help\t\t print this list of options."

	print "\n"

def input_checker(sys_arg,mode):
	file_exist_SBML=False
	file_exist_data=False
	usesbml=True
	rawout_p=False
	rawout_cu=False
	rawout_traj=False
	rawout_odesol=False
	fname = "_results_"
	iname = ""
	analysis = 3
	which_species = False
	parameter_change = False
	init_condit = False
	Nsamples = [0]*4

	for i in range(1,len(sys_arg)):
		if sys_arg[i].startswith('--'):
			option = sys_arg[i][2:]
			if option == 'help':
				printOptions()
				sys.exit()
			elif option == 'analysis':
				analysis = int(sys_arg[i+1])
				if analysis == 0:
					print "Type of Analysis: Prediction for all parameters\n"
				elif analysis == 1:
					print "Type of Analysis: Prediction for a subset of parameters\n"
				elif analysis == 2:
					print "Type of Analysis: Prediction of experiment\n"
			elif option == 'localcode' : 
				usesbml = sys_arg[i+1]
			elif option[0:10] == 'outfolder=' : 
				fname = option[10:]
				print "Output file destination: " + fname + "\n"
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
			elif option == "params":
				parameter_change = sys_arg[i+1]
			elif option == "init":
				init_condit = sys_arg[i+1]
			elif option[0:9] == "infolder=":
				iname = option[9:]
				print "Input file destination: " + iname + "\n"
			elif not(sys_arg[i-1][2:] == 'infile_SBML'): 
				print "\nunknown option "+sys_arg[i]
				printOptions()
				sys.exit()

		elif sys_arg[i].startswith('-'):
			option = sys_arg[i][1:]
			if option == 'h':
				printOptions()
				sys.exit()
			elif option == 'a':
				analysis = int(sys_arg[i+1])
				if analysis == 0:
					print "Type of Analysis: Prediction for all parameters\n"
				elif analysis == 1:
					print "Type of Analysis: Prediction for a subset of parameters\n"
				elif analysis == 2:
					print "Type of Analysis: Prediction of experiment\n"
			elif option == 'lc' : 
				usesbml = sys_arg[i+1]
			elif option[0:3] == 'of=' : 
				fname = option[3:]
				print "Output file destination: " + fname + "\n"
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
			elif option == "p":
				parameter_change = sys_arg[i+1]
			elif option == "ic":
				init_condit = sys_arg[i+1]
			elif option[0:3] == "if=":
				iname = option[3:]
				print "Input file destination: " + iname + "\n"
			elif not(sys_arg[i-1][2:] == 'i1'): 
				print "\nunknown option "+sys_arg[i]
				printOptions()
				sys.exit()
 
	if file_exist_SBML == False:
		print "\nNo input_file_SBML is given!\nUse: \n\t-i1 'inputfile' \nor: \n\t--infile_SBML 'inputfile' \n"
		sys.exit()
	if file_exist_data == False:
		print "\nNo input_file is given!\nUse: \n\t-i2 'inputfile' \nor: \n\t--infile_data 'inputfile' \n"
		sys.exit()
	if analysis not in [0,1,2]:
		print "\nNo analysis type is given!\nUse: \n\t-a 'analysis type' \nor: \n\t --analysis 'analysis type' \n"
		sys.exit()
	if which_species == False and parameter_change == False and init_condit == False:
		print "\nNeed to have different experiments! Specify whether you are: \n\tChanging species measured with -s or --species \n\tChanging parameters by a multiplicative factor with -p or --params \n\tChanging initial conditions with -ic or --init\n"
	
	if not(os.path.isdir("./"+fname)):
		os.mkdir(fname)

	return input_file_SBML, input_file_data, analysis, fname, usesbml, parameter_change, init_condit, iname