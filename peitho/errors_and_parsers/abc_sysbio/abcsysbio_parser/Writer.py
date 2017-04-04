#from CandPythonParser import CandPythonParser
#from CUDAParser import CUDAParser
from peitho.errors_and_parsers.abc_sysbio.abcsysbio_parser.ParsedModel import ParsedModel

class Writer:
    def __init__(self, sbmlFileName, modelName = "", inputPath = "", outputPath = ""):
        self.parsedModel = ParsedModel()
        
        if(modelName == ""):
            self.parsedModel.name = "unnamedModel"
        else:
            self.parsedModel.name = modelName
