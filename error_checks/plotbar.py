import matplotlib.pyplot as plt

def plotbar(MutInfo, modelname, n_groups, approach):
	index = arange(n_groups)
	bar_width = 0.25

	opacity = 0.6

	plt.bar(index, MutInfo, bar_width, alpha=opacity, align='center', color='black')
	if approach==1:
		plt.ylabel('I(Theta,X)', fontweight="bold")
		plt.suptitle('Mutual Information All Parameter approach', fontweight="bold")
	elif approach==2:
		plt.ylabel('I(Theta,X)')
		plt.suptitle('Mutual Information Subset Parameter approach')
	elif approach==3:
		plt.ylabel('I(Y,X)')
		plt.suptitle('Mutual Information Model Outcome')

	plt.xlabel('Experiment')

	plt.xticks(index, modelname)
	if max(MutInfo)<100:
		ylim = max(MutInfo)+11
		ystep = 10
	elif max(MutInfo)<200:
		ylim = max(MutInfo)+21
		ystep = 20
	elif max(MutInfo)<400:
		ylim = max(MutInfo)+31
		ystep = 40
	else:
		ylim = max(MutInfo)+51
		ystep = 50
	plt.yticks(arange(0, ylim, ystep))


	plt.savefig("./results/out.svg")
