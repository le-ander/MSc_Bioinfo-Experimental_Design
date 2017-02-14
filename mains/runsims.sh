#!/bin/bash

module load cuda/4.0.17
REPRESSILATOR=$HOME/work/Repressilator
#GPUDIR=$HOME/work/Repressilator/entropy/Model_set1/all
TESTCODE=$HOME/work/Test_code
MYINSTALL=/cluster/home/saw112/work/software


# Need some setup for running cbarnes python
export PATH=/cluster/home/cbarnes/soft/bin/:$PATH

export LD_LIBRARY_PATH=/cluster/home/cbarnes/soft/lib:/cluster/soft/Linux_2.6_64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/cluster/home/cbarnes/soft/lib:${LD_LIBRARY_PATH}

# set python path to read local abcsysbio and cudsim modules
export PYTHONPATH=$TESTCODE/abc-sysbio/abcsysbio
export PYTHONPATH=$PYTHONPATH:$TESTCODE/abc-sysbio/abcsysbio_parser
export PYTHONPATH=$PYTHONPATH:$TESTCODE/abc-sysbio
export PYTHONPATH=$PYTHONPATH:$REPRESSILATOR/cudaSim/cuda-sim-0.06
export PYTHONPATH=$PYTHONPATH:$TESTCODE
export PYTHONPATH=$MYINSTALL/lib/python:$PYTHONPATH

export PATH=/usr/local/cuda/bin/:/cluster/home/cbarnes/soft/bin:${PATH}

python_exe=/cluster/home/cbarnes/soft/bin/python

abcSysBio_exe=$REPRESSILATOR/abc-sysbio/scripts/run-abc-sysbio
entBio_exe=$REPRESSILATOR/abc-sysbio/scripts/cudaEntropy_all_Scott.py
test_exe=$GPUDIR/mains/main_1.py

test_temp=$REPRESSILATOR/abc-sysbio/scripts/test_SCOTT.py
test_temp_2=$REPRESSILATOR/abc-sysbio/scripts/test_SCOTT_2.py
test_temp_3=$REPRESSILATOR/abc-sysbio/scripts/test_SCOTT_3.py
#cd $REPRESSILATOR/entropy/Model_set1/all

if [ -n "$PBS_O_WORKDIR" ]; then
cd $PBS_O_WORKDIR
else
cd $GPUDIR
echo "Setting using GPUDIR"
fi

rm -rf acceptedParticles
rm -rf rejectedParticles

$python_exe -u $entBio_exe -i input_file_repressilator.xml -of=results_1 -cu -lc >log_all.txt

#$test_exe -i rep_test.origin -of=results_1 -cu 

#$python_exe $test_exe -i input_file_repressilator.xml

#qsub [-keo] -q long -lnodes=cuda07:gpus=1 bash_script

#$python_exe $test_exe -a 0 -of=results -i1 rep_test.xml rep_test.xml -i2 test_data.txt test_data.txt -p 10 -ic 10 -s 01 -lc 11
#$python_exe $test_exe -a 0 -of=results -i1 rep_test_1.xml rep_test_2.xml rep_test_3.xml rep_test_4.xml -i2 new_file2 new_file2 new_file2 new_file2 -p 1010 -ic 1100 -s 0001 -lc 0000

#/cluster/home/saw112/work/git_group_project/Example_data/rep_test.xml
#/cluster/home/saw112/work/git_group_project/Example_data/new_file2