#!/bin/sh

module load python

source activate /cluster/home/jm2716/work/conda_enviroments/Python27_GPU

cd /cluster/home/jm2716/work/Comp_Proj_GIT_simulation/mains

python main_1.py -a 0 -of=results/Repressilator/repr_paper_run1 -i1 repressilator.xml -i2 repr_paper.data -lc 0 -if=Example_data/Repressilator >~/work/Comp_Proj_GIT_simulation/mains/results/Repressilator/log_files/repr_paper_run1.log 2>&1
python main_1.py -a 0 -of=results/Repressilator/repr_paper_run2 -i1 repressilator.xml -i2 repr_paper.data -lc 0 -if=Example_data/Repressilator >~/work/Comp_Proj_GIT_simulation/mains/results/Repressilator/log_files/repr_paper_run2.log 2>&1
python main_1.py -a 0 -of=results/Repressilator/repr_paper_run3 -i1 repressilator.xml -i2 repr_paper.data -lc 0 -if=Example_data/Repressilator >~/work/Comp_Proj_GIT_simulation/mains/results/Repressilator/log_files/repr_paper_run3.log 2>&1

python main_1.py -a 0 -of=results/Repressilator/repr_server_run1 -i1 repressilator.xml -i2 repr_server.data -lc 0 -if=Example_data/Repressilator >~/work/Comp_Proj_GIT_simulation/mains/results/Repressilator/log_files/repr_server_run1.log 2>&1
python main_1.py -a 0 -of=results/Repressilator/repr_server_run2 -i1 repressilator.xml -i2 repr_server.data -lc 0 -if=Example_data/Repressilator >~/work/Comp_Proj_GIT_simulation/mains/results/Repressilator/log_files/repr_server_run2.log 2>&1
python main_1.py -a 0 -of=results/Repressilator/repr_server_run3 -i1 repressilator.xml -i2 repr_server.data -lc 0 -if=Example_data/Repressilator >~/work/Comp_Proj_GIT_simulation/mains/results/Repressilator/log_files/repr_server_run3.log 2>&1

source deactivate

module unload python
