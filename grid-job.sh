#!/bin/bash

#$ -N whale
#$ -l h_vmem=4000M
#$ -l mem_free=4000M
#$ -l h_rt=864000
#$ -S /bin/bash
#$ -cwd

unset SGE_ROOT

export LD_LIBRARY_PATH=/home/volatile/glouppe/local/lib:$LD_LIBRARY_PATH
export LD_RUN_PATH=/home/volatile/glouppe/local/lib:$LD_RUN_PATH
export PATH=/home/volatile/glouppe/local/bin:$PATH
export PYTHONPATH=/home/volatile/glouppe/src/scikit-learn:$PYTHONPATH

python grid.py $@


