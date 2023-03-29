#!/bin/bash                                                                                                                     
#$ -N gen_landscape_data
#$ -j yes
#$ -pe openmp 10                                                                                                                
#$ -l h_vmem=0.5G                                                                                                                 
#$ -l h_rt=04:00:00                                                                                                            
##$ -t 1-64

python3 generate_landscape_data.py



