#!/bin/bash
module load cuda/4.0.17

REPRESSILATOR=$HOME/work/Repressilator
GPUDIR=$HOME/work/test_code
TESTCODE=$HOME/work/Experimental-Design

# Need some setup for running cbarnes python
export PATH=/cluster/home/cbarnes/soft/bin/:$PATH
export LD_LIBRARY_PATH=/cluster/home/cbarnes/soft/lib:/cluster/soft/Linux_2.6_64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/cluster/home/cbarnes/soft/lib:${LD_LIBRARY_PATH}

# Set python path to read local abcsysbio and cudsim modules
export PYTHONPATH=$REPRESSILATOR/abc-sysbio
export PYTHONPATH=$PYTHONPATH:$REPRESSILATOR/cudaSim/cuda-sim-0.06
export PYTHONPATH=$PYTHONPATH:$TESTCODE

export PATH=/usr/local/cuda/bin/:/cluster/home/cbarnes/soft/bin:${PATH}

python_exe=/cluster/home/cbarnes/soft/bin/python
abcSysBio_exe=$REPRESSILATOR/abc-sysbio/scripts/run-abc-sysbio
entBio_exe=$TESTCODE/cudaEntropy_all_Le.py


cd $GPUDIR

rm -rf acceptedParticles
rm -rf rejectedParticles

$python_exe -u $entBio_exe -i input_file_repressilator_le.xml -of=results_1 -cu -lc >log_all.txt
