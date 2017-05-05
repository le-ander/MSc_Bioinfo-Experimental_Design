import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import *

def plotbar(MutInfo, modelname, n_groups, approach):
	index = arange(n_groups)
	bar_width = 0.25

	opacity = 0.6

	plt.bar(index, MutInfo, bar_width, alpha=opacity, align='center', color='black')
	if approach==0:
		plt.ylabel(r'I($\Theta$,X)', fontweight="bold")
		plt.suptitle('Mutual Information - Prediction for all parameters', fontweight="bold")
	elif approach==1:
		plt.ylabel(r'I($\Theta_c$,X)', fontweight="bold")
		plt.suptitle('Mutual Information - Prediction for subset of parameters')
	elif approach==2:
		plt.ylabel('I(Y,X)', fontweight="bold" )
		plt.suptitle('Mutual Information - Prediction of Model Outcome')

	plt.xlabel('Experiments', fontweight="bold")

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


	plt.savefig("./results/out.pdf")
