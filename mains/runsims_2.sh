module load python
source activate /cluster/home/jm2716/work/conda_enviroments/Python27_GPU

cd /cluster/home/saw112/work/run_rep/mains

python main_1.py -a 0 -i1 rep_test.xml -i2 rep_test_data -of=results -lc 0 -if=Example_data > log_repressilator.txt

source deactivate
module unload python
