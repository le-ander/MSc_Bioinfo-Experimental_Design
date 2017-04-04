import csv

def csv_output_writer(MutInfo, MutInfo_inf, MutInfo_inf_prop, sbml_obj, path, input_file,sbml_name):
	with open(path+"/"+sbml_name+"_results_table.csv", 'wb') as csvfile:
		fieldnames = ['Experiment name','Type' ,'Measured species', 'Approach', 'Fitted parameter', 'Fitted initials', 'Fitted compartments' ,'Total particle','N1','N2','N3','N4','Sigma', 'dt' , 'Cuda file', 'SBML file', 'Data input file', 'Posterior sample' , 'Posterior sample particles' ,'Posterior sample weights' , 'Total infinites/nan', 'Percentage infinites/nan', 'Mutual Information']

		resultswriter = csv.DictWriter(csvfile, fieldnames=fieldnames,  restval='NA', extrasaction='ignore', delimiter=';',quotechar='"', quoting=csv.QUOTE_MINIMAL)
		
		resultswriter.writeheader()
		for i in range(sbml_obj.nmodels):
			resultswriter.writerow({'Experiment name':sbml_obj.name[i],'Type':sbml_obj.type,'Measured species':sbml_obj.fitSpecies[i], 'Approach':sbml_obj.analysisType, 'Fitted parameter':sbml_obj.param_fit, 'Fitted initials':sbml_obj.init_fit, 'Fitted compartments':sbml_obj.comp_fit , 'Total particle':sbml_obj.particles, 'N1':sbml_obj.N1sample,'N2':sbml_obj.N2sample,'N3':sbml_obj.N3sample,'N4':sbml_obj.N4sample,'Sigma':sbml_obj.sigma, 'dt':sbml_obj.dt , 'Cuda file':sbml_obj.cuda[i], 'SBML file':sbml_obj.source[i], 'Data input file':input_file, 'Posterior sample':bool(sbml_obj.sampleFromPost) , 'Posterior sample particles':sbml_obj.post_sample_file ,'Posterior sample weights':sbml_obj.post_weight_file , 'Total infinites/nan': MutInfo_inf[i], 'Percentage infinites/nan':MutInfo_inf_prop[i], 'Mutual Information':MutInfo[i]})