#!/usr/bin/env python3

"""
This script is used for making the plots of the AR as a function of the kappa in the manuscript.
Note that in my code kappa is named epsilon.
Run with "$ PYTHONHASHSEED=0 python3 kappa_plot.py" to get consistent hash values in python 3.3.
"""
import sys
sys.path.insert(1, '../')
import numpy as np
import chainer as ch
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.lines import Line2D
import _HQAOA

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 9}) # Latex plot labels.
rc('text', usetex=True)

def get_ep_list(path,instance,d,n_iter,p): # Get the list of available ep values in path+instance at the given d,n_iter and p. Sort the list.
    def filter_data(data,d,n_iter,path,p,n):
        def conditions(line):
            cond=[line[0]['d']==d,
                  line[0]['n_iter']==n_iter,
                  line[0]['direction']==path[-2:-10:-1][::-1] or line[0]['direction']==path[-2:-9:-1][::-1], # Check line is in the right path.
                  line[0]['p']==p,
                  line[1]['n']==n]
            return all(cond)
        filtered=[line for line in data if conditions(line)]
        return filtered

    data=_HQAOA.import_instance_data(path+instance)
    data=filter_data(data,d,n_iter,path,p,n)
    ep_data=list(set([line[0]['ep'] for line in data]))
    ep_data.sort()
    return ep_data

def plot_points(path,instance,d,n_iter,p):
    pltpts=[]
    E0=_HQAOA.get_instance_E0(path+instance)
    ep_list=get_ep_list(path,instance,d,n_iter,p)

    for ep in ep_list:  # Per ep, take the line with the smallest cost
        best_cost=_HQAOA.load_optimal_line(path+instance, p, ep, d)[2]['cost_qaoa']
        best_AR=best_cost/E0
        pltpts.append([ep,best_AR])

    pltpts.sort()
    pltpts=np.array(pltpts).transpose()

    np.save('plot_points/lone_kappa_plot_plot_points_noise_aware_{}.npy'.format(hash((path,instance,d,n_iter,p))),pltpts)

    return pltpts

def plot_points_noise_unaware(unaware_path,instance,n,d,n_iter,direc,p): # Take the optimal angles at p=0, and use them to obtain an AR at other p.
    pltpts=[]
    E0=_HQAOA.get_instance_E0(unaware_path+instance)
    con=_HQAOA.get_instance_con(unaware_path+instance)
    ep_list=get_ep_list(unaware_path,instance,d,n_iter,p) # get the ep for which I ran noiseless optimization.
    for ep in ep_list: # Use angle unaware angles, put them through noisy chan.
        opt_pars=_HQAOA.load_optimal_line(unaware_path+instance, 0, 0, d)[2]['opt_parameters'] # Load optimal parameters at p,ep=0.
        opt_pars=ch.Variable(np.array(opt_pars))
        if direc=='temporal':
            cost=_HQAOA.cost_from_parameters_swap_temporal_quantum_classical(con,n,d,p,ep,opt_pars).data
        elif direc=='spatial':
            cost=_HQAOA.cost_from_parameters_swap_spatial(con,n,d,p,ep,opt_pars).data

        AR=cost/E0
        pltpts.append([ep,AR])

    pltpts.sort()
    pltpts=np.array(pltpts).transpose()

    np.save('plot_points/lone_kappa_plot_plot_points_noise_unaware_{}.npy'.format(hash((unaware_path,instance,n,d,n_iter,direc,p))),pltpts)
    return pltpts

def plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax,_color):
    # If you made new data points, the npy files should be removed. Only in that case the plot points are computed anew.
    fname='plot_points/lone_kappa_plot_plot_points_noise_aware_{}.npy'.format(hash((direc_path,instance,d,n_iter,p)))
    if os.path.isfile(fname):
        print('Loading plot points')
        points=np.load(fname)
    else:
        print('Calculating plot points')
        points=plot_points(direc_path,instance,d,n_iter,p)
    ax.plot(*points,'-o',alpha=1,color=_color,markerfacecolor='none',markersize=7,label=label1,solid_capstyle='round')

    fname='plot_points/lone_kappa_plot_plot_points_noise_unaware_{}.npy'.format(hash((unaware_path,instance,n,d,n_iter,direc,p)))
    if os.path.isfile(fname):
        points=np.load(fname)
        print('Loading plot points')
    else:
        print('Calculating plot points')
        points=plot_points_noise_unaware(unaware_path,instance,n,d,n_iter,direc,p)

    ax.plot(*points,'-o',alpha=1,color=_color,markeredgecolor='none',markersize=3.7,label=label2,solid_capstyle='round')

if __name__=='__main__':
    master_path='../data/SK/SWAP-network/6/' # path to all instance data.
    path_temp=master_path+'temporal/' # Path to temporal noise correlation data.
    path_spat=master_path+'spatial/' # Path to spatial noise correlation data.
    instance='010010100111110' # Instance name.
    unaware_path=path_spat # Where to look for data if p=0, for any ep.
    n_iter=4 # Assert later every data point has used n_iter bassinhopping iterations.
    n_runs=32 # Assert later we have n_runs data points for every kappa=ep.
    n=6 # Assert later data is from n=6 qubits.
    p=0.001 # Error probability at which plots are shown.
    d=3 # The number of cycles to get the data from.

    fig, ax = plt.subplots(figsize=(2.8,3))
    colors=[]
    for i in range(10):
        colors.append(next((ax._get_lines.prop_cycler))['color'])
    ax.set_xlabel('$\kappa$')
    ax.set_ylabel("AR")

    direc_path=path_temp
    direc='temporal'
    label1='Temp.\n aware'
    label2='Temp.\n unaware'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax,colors[1])

    direc_path=path_spat
    direc='spatial'
    label1='Spat.\n aware'
    label2='Spat.\n unaware'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax,colors[2])

    p=0.01 #

    direc_path=path_temp
    direc='temporal'
    label1='_nolegend_'
    label2='_nolegend_'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax,colors[1])

    direc_path=path_spat
    direc='spatial'
    label1='_nolegend_'
    label2='_nolegend_'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax,colors[2])

    # Add extrapolated AR datapoints found with derivatives.py for p=0.001
    temp_AR=[1,0.866452441411907] # [kappa,AR]
    spat_AR=[1,0.8506262123039643]
    uncor_AR=[0,0.8267500487250729]

    # Same for p=0.01
    temp2_AR=[1,0.8175971945193308] # [kappa,AR]
    spat2_AR=[1,0.6593349034399043]
    uncor2_AR=[0,0.4205732676509891]

    pts=np.array([temp_AR,spat_AR,uncor_AR,temp2_AR,spat2_AR,uncor2_AR]).transpose()
    lin=ax.scatter(*pts,marker='x',color='black',s=20,label='Linearized',capstyle='round',zorder=1000,alpha=0.75)

    #ax.legend(loc='upper left', bbox_to_anchor=(0.0,0.81))
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax.legend(handles=[lin],loc='lower right')
    fig.savefig('plots/kappa_plot.pdf',bbox_inches='tight',pad_inches=0.005)
