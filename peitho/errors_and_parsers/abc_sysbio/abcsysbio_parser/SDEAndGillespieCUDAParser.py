from peitho.errors_and_parsers.abc_sysbio.abcsysbio_parser.CUDAParser import CUDAParser

class SdeAndGillespieCUDAParser(CUDAParser):
    def __init__(self, sbmlFileName, modelName, integrationType, method, inputPath="", outputPath=""):
        CUDAParser.__init__(self, sbmlFileName, modelName, integrationType, method, inputPath=inputPath, outputPath=outputPath) ###input output path needs to be double checked
        
    def renameMathFunctions(self):
        CUDAParser.renameMathFunctions(self)
        self.mathCuda.append('t')
