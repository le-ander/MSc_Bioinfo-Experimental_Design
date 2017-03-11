def extractConstantsPairings(model_object, combinationList):
	#Cuda_files = [x[1] for x in combinationList]
	Init_sets = [x[0] for x in combinationList]
	pairings = {}
	for Cfile in set(model_object.cuda):
		temp = [Init_sets[j] for j in [i for i, x in enumerate(model_object.cuda) if x == Cfile]]
		temp.sort()
		pairings[Cfile] = temp
	return pairings