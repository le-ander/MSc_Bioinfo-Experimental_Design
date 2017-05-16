#!/bin/bash

module load python
source activate /cluster/home/jm2716/work/conda_enviroments/Python27_GPU

if [ $THISHOST == cuda10 ]
then
	export CUDA_DEVICE=10
fi

cd /cluster/home/saw112/work/git_group_Hess1/mains

for i in `seq 1 2000`; do
	python main_1.py -a 1 -of=results/Hess1/h1_h_pd -i1 hess1_h.xml -lc 1 -if=Example_data/Hess1 &>> results/Hess1/log_files_pd/h1_h_pd.log
done

#python main_1.py -a 1 -of=results/Hess1/h1_h_run1 -i1 hess1_h.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_h_run1.log
#python main_1.py -a 1 -of=results/Hess1/h1_nu_run1 -i1 hess1_v.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_nu_run1.log
#python main_1.py -a 1 -of=results/Hess1/h1_k1_run1 -i1 hess1_k1.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_k1_run1.log

#python main_1.py -a 1 -of=results/Hess1/h1_p0_run2 -i1 hess1_P0.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_p0_run2.log
#python main_1.py -a 1 -of=results/Hess1/h1_h_run2 -i1 hess1_h.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_h_run2.log
#python main_1.py -a 1 -of=results/Hess1/h1_nu_run2 -i1 hess1_v.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_nu_run2.log
#python main_1.py -a 1 -of=results/Hess1/h1_k1_run2 -i1 hess1_k1.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_k1_run2.log

#python main_1.py -a 1 -of=results/Hess1/h1_p0_run3 -i1 hess1_P0.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_p0_run3.log
#python main_1.py -a 1 -of=results/Hess1/h1_h_run3 -i1 hess1_h.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_h_run3.log
#python main_1.py -a 1 -of=results/Hess1/h1_nu_run3 -i1 hess1_v.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_nu_run3.log
#python main_1.py -a 1 -of=results/Hess1/h1_k1_run3 -i1 hess1_k1.xml -lc 1 -if=Example_data/Hess1 &> results/Hess1/log_files/h1_k1_run3.log

unset CUDA_DEVICE
