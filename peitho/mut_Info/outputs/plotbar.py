import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import *

def plotbar(MutInfo, modelname, n_groups, approach, out_path):

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


	plt.savefig(out_path+"/results_bar.pdf")

def plotbarh(MutInfo, modelname, n_groups, approach, out_path,sbml_name):
	plt.rcdefaults()
	fig, ax = plt.subplots()
	
	y_pos = arange(len(modelname))

	ax.barh(y_pos, MutInfo, align='center',
	        color='grey', ecolor='black')
	ax.set_yticks(y_pos)
	ax.set_yticklabels(modelname)
	ax.invert_yaxis()  # labels read top-to-bottom
	
	if max(MutInfo)<100:
		xlim = max(MutInfo)+31
		xstep = 10
	elif max(MutInfo)<200:
		xlim = max(MutInfo)+61
		xstep = 20
	elif max(MutInfo)<400:
		xlim = max(MutInfo)+91
		xstep = 40
	else:
		xlim = max(MutInfo)+151
		xstep = 50
	plt.xticks(arange(0, xlim, xstep))

	if approach==0:
		ax.set_xlabel(r'I($\Theta$,X)', fontweight="bold")
		ax.set_title('Mutual Information - Prediction for all parameters', fontweight="bold")
	elif approach==1:
		ax.set_xlabel(r'I($\Theta_c$,X)', fontweight="bold")
		ax.set_title('Mutual Information - Prediction for subset of parameters')
	elif approach==2:
		ax.set_xlabel('I(Y,X)', fontweight="bold" )
		ax.set_title('Mutual Information - Prediction of Model Outcome')
	
	for i,value in enumerate(MutInfo):
 		ax.text(value+xstep/4,y_pos[i],round(value,2), verticalalignment ='center', fontsize=8)

	plt.tight_layout()

	plt.savefig(out_path+"/"+sbml_name+"_results_graph.pdf")