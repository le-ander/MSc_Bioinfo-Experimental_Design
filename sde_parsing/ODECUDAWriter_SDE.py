#from libsbml import *
#from abcsysbio.relations import *
import os
import re
import sys
from Writer_SDE import Writer

class OdeCUDAWriter(Writer):
	def __init__(self, MEANS_obj, LNA_obj, modelName="",  outputPath="", num_comps = 0):
		Writer.__init__(self, modelName, outputPath = outputPath)
		self.out_file=open(os.path.join(outputPath,self.parsedModel.name+".cu"),"w")
		
		numSpecies_tot = len(MEANS_obj.species)
		a = LNA_obj.__str__()

		#print a
		a = [x.strip() for x in a.split("\n")[7::3]]
		self.parsedModel.numSpecies_exvar = numSpecies_tot
		self.parsedModel.numSpecies = numSpecies_tot*(numSpecies_tot+3)/2
		self.parsedModel.numReactions = MEANS_obj.stoichiometry_matrix.shape[1]
		self.parsedModel.numComps = num_comps
		variances = [""]*(numSpecies_tot*(numSpecies_tot+1)/2)
		pos = 0
		for i in range(numSpecies_tot):
			for j in range(i,numSpecies_tot):
				variances[pos] = "V_"+str(i)+"_"+str(j)
				pos+=1
	
		self.parsedModel.species = [str(x) for x in MEANS_obj.species]+variances
		self.parsedModel.parameter = [str(x) for x in MEANS_obj.parameters]
		self.parsedModel.numGlobalParameters = len(MEANS_obj.parameters)

		for i in range(self.parsedModel.numGlobalParameters):
			self.parsedModel.parameterId.append("tex2D(param_tex,"+str(i)+",tid)")

		for i in range(self.parsedModel.numSpecies):
			self.parsedModel.speciesId.append("y["+str(i)+"]")

		self.parsedModel.ODElist = a

	def write(self):     
		p=re.compile('\s')
	
		#Write number of parameters and species
		self.out_file.write("#define NSPECIES " + str(self.parsedModel.numSpecies) + "\n")
		self.out_file.write("#define NPARAM " + str(self.parsedModel.numGlobalParameters) + "\n")
		self.out_file.write("#define NREACT " + str(self.parsedModel.numReactions) + "\n")
		self.out_file.write("\n")

		self.out_file.write("struct myFex{\n    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)\n    {\n        int tid = blockDim.x * blockIdx.x + threadIdx.x;\n")
		self.out_file.write("\n\n\n")


		for i in range(self.parsedModel.numSpecies-1,-1,-1):
			self.out_file.write("        ydot["+repr(i)+"]=")

			if self.parsedModel.numComps == 1:
				self.out_file.write('(')
			self.out_file.write(self.reactionTransform(i))
			if self.parsedModel.numComps == 1 and i > (self.parsedModel.numSpecies_exvar - 1):
				self.out_file.write(')/__powf(tex2D(param_tex,0,tid),2)')
			elif self.parsedModel.numComps == 1 and i <= (self.parsedModel.numSpecies_exvar -1):
				self.out_file.write(')/tex2D(param_tex,0,tid)')
			self.out_file.write(";\n")

		self.out_file.write("\n    }")
		self.out_file.write("\n};\n\n\n struct myJex{\n    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){\n        return; \n    }\n};") 

	def reactionTransform(self,reac_no):
		reaction_new=""
		param_species = self.parsedModel.parameter + self.parsedModel.species
		param_species_ID = self.parsedModel.parameterId + self.parsedModel.speciesId
		reaction = self.parsedModel.ODElist[reac_no]
		param_species_sort = sorted(param_species, key=len)
		param_species_sort.reverse()
	
		pos = 0
		while pos < len(reaction):
			for i in param_species_sort:
				temp = reaction[pos:pos+len(i)]
				if temp == i:
					reaction_new += param_species_ID[param_species.index(i)]
					pos+=len(temp)
					break
				elif i == param_species_sort[-1]:
					reaction_new += reaction[pos:pos+1]
					pos+=1

		
		num_pows = len(re.findall(r'\*\*', reaction_new, re.M|re.I))
		for pow_replace in range(num_pows):
			matchObj = re.search( r'\*\*', reaction_new, re.M|re.I)
			if matchObj:
				LHS_end = matchObj.start()
				RHS_start = matchObj.end()
				close_bracket = 0
				open_bracket = 0
				#LHS side
				LHS = reaction_new[:LHS_end]
				RHS = reaction_new[RHS_start:]

				for pos, charac in enumerate(LHS[::-1]):
					arith_check = [" ", "+", "-", "/", "**", "*", "("]
					if pos == 1:
						if charac == ")":
							close_bracket += 1
						continue

					if charac in arith_check and open_bracket == close_bracket:
						LHS_start = LHS_end - pos
						break
					elif charac == "(":
						open_bracket += 1
					elif charac == ")":
						close_bracket += 1

				for pos, charac in enumerate(RHS):
					arith_check = [" ", "+", "-", "/", "**", "*", ")"]
					if pos == 0:
						if charac == "(":
							open_bracket += 1
						continue

					if charac in arith_check and open_bracket == close_bracket:
						RHS_end = RHS_start + pos
						break
					elif charac == "(":
						open_bracket += 1
					elif charac == ")":
						close_bracket += 1

				replace_str = "__powf("+reaction_new[LHS_start:LHS_end]+","+reaction_new[RHS_start:RHS_end]+")"
				
				reaction_new=reaction_new[:LHS_start]+replace_str+reaction_new[RHS_end:]
				
		return reaction_new
		
