module load python
source activate /cluster/home/jm2716/work/conda_enviroments/Python27_GPU

cd /cluster/home/ld2113/work/Experimental-Design/main

python main_1.py -a 2 -i1 p53_model.xml p53_model_exp.xml -i2 data_p53 data_p53_exp -of=results -lc 00 -if=Example_data > log_repressilator.txt
