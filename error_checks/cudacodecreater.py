import sys
import ParseAndWrite

def cudacodecreater(input_files, inPath="",outPath=""):
	intType = ["CUDA ODE"]*len(input_files)
	names=[]
	for i in range(0,len(input_files)):
		names.extend(["Exp_"+repr(i+1)])
	ParseAndWrite.ParseAndWrite(source=input_files,integrationType=intType,modelName=names,inputPath=inPath,outputPath=outPath,method=None)