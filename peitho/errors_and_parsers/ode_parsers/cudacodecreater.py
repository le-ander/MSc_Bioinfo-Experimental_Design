import sys
import peitho.errors_and_parsers.abc_sysbio.abcsysbio_parser.ParseAndWrite as ParseAndWrite

# A function that writes the cuda code files (uses abcsys-bio parsers)
##(gets called by sorting_files)
##Arguments: 
##input_files - the names of the SBML files
##inPath - where the SBML files are
##outPath - where the cudacode files are written to
def cudacodecreater(input_files, inPath="",outPath="",template=False):
	#Since only doing ODEs right now this is given for the intType
	print input_files
	if template == False:
		intType = ["CUDA ODE"]*len(input_files)
		#Sets the names of the cuda code files
		names=[]
		for i in range(0,len(input_files)):
			names.extend(["Exp_"+repr(i+1)])
		#Calls ParseAndWrite which then creates the cuda code files (this is abc-sysbio)
		ParseAndWrite.ParseAndWrite(source=input_files,integrationType=intType,modelName=names,inputPath=inPath,outputPath=outPath,method=None)
	else:
		intType = ["CUDA ODE"]
		ParseAndWrite.ParseAndWrite(source=input_files,integrationType=intType,modelName=["template_cudacode"],inputPath=inPath,outputPath=outPath,method=None)