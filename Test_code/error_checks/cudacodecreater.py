import sys
sys.path.insert(0, '/cluster/home/saw112/work/Test_code/abcsysbio_parser')
import ParseAndWrite

def cudacodecreater(input_files, names="", inPath="",outPath=""):
	intType = ["CUDA ODE"]*len(input_files)
	if names=="":
		names=[]
		for i in range(0,len(input_files)):
			names.extend(["model_"+repr(i+1)])
	ParseAndWrite.ParseAndWrite(source=input_files,integrationType=intType,modelName=names,inputPath=inPath,outputPath=outPath,method=None)