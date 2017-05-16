#!/bin/bash
    
module load python
source activate /cluster/home/jm2716/work/conda_enviroments/Python27_GPU

if [ $THISHOST == cuda10 ]
then
	export CUDA_DEVICE=6
fi

cd /cluster/home/saw112/work/git_group_Hess1/mains

#Wildtype
for i in `seq 1 1500`; do
	python main_1.py -a 2 -of=results/p53/p53_4xk1a_pd -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_4xk1a.data -lc 00 -if=Example_data/p53 &>> results/p53/log_files_pd/p53_4xk1a_pd.log
done

#python main_1.py -a 2 -of=results/p53/p53_WTb_run1 -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_WTb.data -lc 00 -if=Example_data/p53 &> results/p53/log_files/p53_WTb_run1.log
#python main_1.py -a 2 -of=results/p53/p53_WTc_run1 -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_WTc.data -lc 00 -if=Example_data/p53 &> results/p53/log_files/p53_WTc_run1.log

#k5
#python main_1.py -a 2 -of=results/p53/p53_4xk5a_run1 -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_4xk5a.data -lc 00 -if=Example_data/p53 &> results/p53/log_files/p53_4xk5a_run1.log
#python main_1.py -a 2 -of=results/p53/p53_4xk5b_run1 -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_4xk5b.data -lc 00 -if=Example_data/p53 &> results/p53/log_files/p53_4xk5b_run1.log
#python main_1.py -a 2 -of=results/p53/p53_4xk5c_run1 -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_4xk5c.data -lc 00 -if=Example_data/p53 &> results/p53/log_files/p53_4xk5c_run1.log

#k1
#python main_1.py -a 2 -of=results/p53/p53_4xk1a_run1 -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_4xk1a.data -lc 00 -if=Example_data/p53 &> results/p53/log_files/p53_4xk1a_run1.log
#python main_1.py -a 2 -of=results/p53/p53_4xk1b_run1 -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_4xk1b.data -lc 00 -if=Example_data/p53 &> results/p53/log_files/p53_4xk1b_run1.log
#python main_1.py -a 2 -of=results/p53/p53_4xk1c_run1 -i1 p53_model_ref.xml p53_model_exp.xml -i2 p53_ref.data p53_4xk1c.data -lc 00 -if=Example_data/p53 &> results/p53/log_files/p53_4xk1c_run1.log


unset CUDA_DEVICE
