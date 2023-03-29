#!/bin/bash                                                                                                                     
#$ -N spatial_p_plot_d=3_ep=0
#$ -j yes
#$ -pe openmp 16                                                                                                                
#$ -l h_vmem=1G                                                                                                                 
#$ -l h_rt=02:00:00                                                                                                            
#$ -t 1-128

d=3
ep=0.0

start=0.0052
step=0.0002
repeat=32

p=$(echo "import numpy as np; print(np.round(np.floor(($SGE_TASK_ID-1)/$repeat)*$step+$start,5))" | python3 )

python3 ~/HQAOA/HQAOA.py $PWD $d $p $ep spatial --n_iter 4



