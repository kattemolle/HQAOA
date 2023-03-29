#!/bin/bash                                                                                                                     
#$ -N temporal_p_plot_d=3_ep=1
#$ -j yes
#$ -pe openmp 16                                                                                                                
#$ -l h_vmem=2G                                                                                                                 
#$ -l h_rt=10:00:00                                                                                                            
#$ -t 1-1504

d=3
ep=1.0

start=0.03
step=0.01
repeat=32

p=$(echo "import numpy as np; print(np.round(np.floor(($SGE_TASK_ID-1)/$repeat)*$step+$start,5))" | python3 )

python3 ~/HQAOA/HQAOA.py $PWD $d $p $ep temporal --n_iter 4


