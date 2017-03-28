#!/bin/sh

module load python
source activate /cluster/home/jm2716/work/conda_enviroments/Python27_GPU

if [ $THISHOST == cuda10 ]
then
	export CUDA_DEVICE=0
fi

cd $GPUDIR/mains

##gE1 (SBML input):
#python main_1.py -a 0 -of=results -i1 rep_test.xml -i2 new_file2 -lc 0 -if=Example_data

##gE2 (local code):
#python main_1.py -a 1 -of=results -i1 input_file_hess1.xml -lc 1 -if=Example_data

##gE3 (SBML input):
#python main_1.py -a 2 -i1 p53_model.xml p53_model_exp.xml -i2 data_p53_1 data_p53_exp_1 -of=results/results_p53_1 -lc 00 -if=Example_data &> results/log_files/log_file_p53_1.log 

#python main_1.py -a 2 -i1 p53_model.xml p53_model_exp.xml -i2 data_p53_2 data_p53_exp_2 -of=results/results_p53_2 -lc 00 -if=Example_data &> results/log_files/log_file_p53_2.log

#python main_1.py -a 2 -i1 p53_model.xml p53_model_exp.xml -i2 data_p53_3 data_p53_exp_3 -of=results/results_p53_3 -lc 00 -if=Example_data &> results/log_files/log_file_p53_3.log

#python main_1.py -a 2 -i1 p53_model.xml p53_model_exp.xml -i2 data_p53_4 data_p53_exp_4 -of=results/results_p53_4 -lc 00 -if=Example_data &> results/log_files/log_file_p53_4.log

python main_1.py -a 2 -i1 p53_model.xml p53_model_exp.xml -i2 data_p53_5 data_p53_exp_5 -of=results/results_p53_5 -lc 00 -if=Example_data &> results/log_files/log_file_p53_5.log

#&> ~/work/Testing/orig6_124e4_11_cuda10.txt

unset CUDA_DEVICE
