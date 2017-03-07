module load python
source activate /cluster/home/jm2716/work/conda_enviroments/Python27_GPU

cd /cluster/home/ld2113/work/Experimental-Design/main

##gE1 (SBML input):
#python main_1.py -a 0 -of=results -i1 rep_test.xml -i2 new_file2 -lc 0 -if=Example_data

##gE2 (local code):
#python main_1.py -a 1 -of=results -i1 input_file_hess1.xml -i2 new_file2 -lc 1 -if=Example_data

##gE3 (SBML input):
python main_1.py -a 2 -i1 p53_model.xml p53_model_exp.xml -i2 data_p53 data_p53_exp -of=results -lc 00 -if=Example_data > log_repressilator.txt
