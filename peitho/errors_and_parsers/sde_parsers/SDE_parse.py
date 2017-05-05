import sys, os, libsbml
sys.path.insert(1, '../abc-sysbio')
import means
import numpy as np
import ODECUDAWriter_SDE
import abcsysbio_parser.ParseAndWrite


def LNA_CUDAWriter(infiles,inpath="",outpath=""):

	for infile in infiles:
		if not(os.path.isdir(outpath+"/"+"LNA")):
			os.mkdir(outpath+"/"+"LNA")
		outPath = outpath+"/"+"LNA"+"/"

		reader = libsbml.SBMLReader()
		document = reader.readSBML(inpath+infile)
		sbml_model = document.getModel()

		ncomps = len(sbml_model.getListOfCompartments())

		model_test , rep_parameters, rep_IC = means.io.read_sbml(inpath+infile)

		ode_problem_lna = means.lna_approximation(model_test)

		txt_obj = ODECUDAWriter_SDE.OdeCUDAWriter(num_comps = ncomps, MEANS_obj = model_test, LNA_obj = ode_problem_lna ,modelName = infile[:-4],outputPath = outPath)
		
		txt_obj.write()

		#simulation = means.Simulation(ode_problem_lna,"cvode")
		
		#parameters = [1.0, 1544.70378, 5.61815339, 5.42635926, 0.327849618]
		#initial_conditions = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
		#time_points = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0])
		#trajectories = simulation.simulate_system(parameters,initial_conditions,time_points)
		#print trajectories
