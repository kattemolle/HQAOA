#!/usr/bin/env python3

"""
Produce plots for the SM. Run with "$ PYTHONHASHSEED=0 python3 plots.py" to get consistent hash values in python 3.3
"""

import sys
sys.path.insert(1, '../')
import numpy as np
import qem
import chainer as ch
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rc
import _HQAOA
import os

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 9})
rc('text', usetex=True)

def get_p_list(path,instance,d,n_iter,ep,n): # Get the list of available p values in path+instance at the given d,n_iter and ep. Sort the list.
    def filter_data(data,d,n_iter,path,ep,n):
        def conditions(line):
            cond=[line[0]['d']==d,
                  line[0]['n_iter']==n_iter,
                  line[0]['direction']==path[-2:-10:-1][::-1] or line[0]['direction']==path[-2:-9:-1][::-1], # Check line is in the right path.
                  line[0]['ep']==ep,
                  line[1]['n']==n]
            return all(cond)
        filtered=[line for line in data if conditions(line)]
        return filtered

    data=_HQAOA.import_instance_data(path+instance)
    data=filter_data(data,d,n_iter,path,ep,n)
    p_data=list(set([line[0]['p'] for line in data]))
    p_data.sort()
    return p_data

def plot_points(unaware_path,path,instance,d,n_iter,ep,n):
    pltpts=[]
    E0=_HQAOA.get_instance_E0(path+instance)
    data=_HQAOA.import_instance_data(path+instance)
    p_list=get_p_list(path,instance,d,n_iter,ep,n)
    if p_list[0]!=0:
        p_list=[0]+p_list # Allways add p as a plot point.

    for p in p_list:  # Per p, take the line with the smallest cost
        if p==0: # For p=0, allways look in the noise_unaware folder, because at p=0, results are the same for any ep and at p=0 I did not recompute for every ep.
            best_line=_HQAOA.load_optimal_line(unaware_path+instance, p, 0, d)
            best_cost=best_line[2]['cost_qaoa']
            best_pars=best_line[2]['opt_parameters']
        else:
            best_line=_HQAOA.load_optimal_line(path+instance, p, ep, d)
            best_cost=best_line[2]['cost_qaoa']
            best_pars=best_line[2]['opt_parameters']
        best_AR=best_cost/E0
        pltpts.append([p,best_AR,best_pars])

    pltpts.sort()
    pltpts=np.array(pltpts,dtype=object)
    pltpts=pltpts.transpose()

    np.save('plot_points/p_plot_plot_points_noise_aware_{}.npy'.format(hash((path,instance,d,n_iter,ep))),pltpts,allow_pickle=True)
    return pltpts

def plot_points_noise_unaware(unaware_path,instance,n,d,n_iter,direc,ep): # Take the optimal angles at p=0, and use them to obtain an AR at other p.
    pltpts=[]
    E0=_HQAOA.get_instance_E0(unaware_path+instance)
    con=_HQAOA.get_instance_con(unaware_path+instance)
    p_list=get_p_list(unaware_path,instance,d,n_iter,ep,n) # get the p for which I ran noise aware optimization.
    if p_list[0]!=0:
        p_list=[0]+p_list # Allways add p as a plot point.

    with ch.no_backprop_mode():
        for p in p_list: # Use optimal angles at p=0, put them through noisy chan.
            opt_pars=_HQAOA.load_optimal_line(unaware_path+instance, 0, 0, d)[2]['opt_parameters'] # Load optimal parameters at p,ep=0.
            opt_pars=ch.Variable(np.array(opt_pars))
            if direc=='temporal':
                cost=_HQAOA.cost_from_parameters_swap_temporal_quantum_classical(con,n,d,p,ep,opt_pars).data
            elif direc=='spatial':
                cost=_HQAOA.cost_from_parameters_swap_spatial(con,n,d,p,ep,opt_pars).data

            AR=cost/E0
            pltpts.append([p,AR])

    pltpts.sort()
    pltpts=np.array(pltpts).transpose()

    np.save('plot_points/p_plot_plot_points_noise_unaware_{}.npy'.format(hash((unaware_path,instance,n,d,n_iter,direc,ep))),pltpts)
    return pltpts

def plotter(direc_path,direc,instance,unaware_path,n_iter,d,ep,label1,label2,ax,_color,man_p_list,n,par_ax):
    # If you made new data points, you should remove the npy files. Only in that case the plot points are computed anew.
    fname='plot_points/p_plot_plot_points_noise_aware_{}.npy'.format(hash((direc_path,instance,d,n_iter,ep)))
    if not os.path.isfile(fname):
        fname='plot_points/p_plot_plot_points_noise_aware_{}.npy'.format(hash(('../../'+direc_path,instance,d,n_iter,ep))) # For backward compatibility.
    if os.path.isfile(fname):
        print('Loading aware plot points')
        points=np.load(fname,allow_pickle=True)
        if len(man_p_list)!=0:
            points=points.transpose()
            points=[point for point in points if point[0] in man_p_list]
            points=np.array(points)
            points=points.transpose()
    else:
        print('Calculating plot points, aware')
        points=plot_points(unaware_path,direc_path,instance,d,n_iter,ep,n)

    ax.plot(points[0],points[1],'-o',alpha=0.75,color=_color,markerfacecolor='none',markersize=7,label=label1,solid_capstyle='round')

    # I have to get the opt angles at p=0.
    opt_unaware_pars=_HQAOA.load_optimal_line(unaware_path+instance, 0, 0, d)[2]['opt_parameters'] # Load optimal parameters at p,ep=0.
    # Calc dist between these angles and the aware angles.
    dists=[]
    for i,_ in enumerate(points[0]):
        dist=_HQAOA.param_dist(opt_unaware_pars,points[2][i])
        dists.append(dist)

    par_ax.plot(points[0],dists,'-o',alpha=0.75,color=_color,markerfacecolor='none',markersize=7,solid_capstyle='round')

    fname='plot_points/p_plot_plot_points_noise_unaware_{}.npy'.format(hash((unaware_path,instance,n,d,n_iter,direc,ep)))
    if not os.path.isfile(fname):
        fname='plot_points/p_plot_plot_points_noise_unaware_{}.npy'.format(hash(('../../'+unaware_path,instance,n,d,n_iter,direc,ep)))
    if os.path.isfile(fname):
        points=np.load(fname)
        print('Loading unaware plot points')
        if len(man_p_list)!=0:
            points=points.transpose()
            points=[point for point in points if point[0] in man_p_list]
            points=np.array(points)
            points=points.transpose()
    else:
        print('Calculating plot points, unaware')
        points=plot_points_noise_unaware(unaware_path,instance,n,d,n_iter,direc,ep)

    ax.plot(*points,'-o',alpha=0.75,color=_color,markeredgecolor='none',markersize=3.7,label=label2,solid_capstyle='round')

def random_guess_AR(path,instance):
    E0=_HQAOA.get_instance_E0(path+instance)
    con=_HQAOA.get_instance_con(path+instance)
    n=6
    d=0
    p=0
    ep=0
    pars=ch.Variable(np.array([-1.,1.,-1.,0.,0.,1.]))
    cost=_HQAOA.cost_from_parameters_swap_temporal_quantum_classical(con,n,d,p,ep,pars).data
    return cost/E0

#random_guess_AR('../../data/SK/SWAP-network/6/spatial/','010010100111110')

def get_ep_list(path,instance,d,n_iter,p,n): # Get the list of available ep values in path+instance at the given d,n_iter and p. Sort the list.
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

def kappa_plot_points(path,instance,d,n_iter,p,n):
    pltpts=[]
    E0=_HQAOA.get_instance_E0(path+instance)
    ep_list=get_ep_list(path,instance,d,n_iter,p,n)

    for ep in ep_list:  # Per ep, take the line with the smallest cost
        best_line=_HQAOA.load_optimal_line(path+instance, p, ep, d)
        best_cost=best_line[2]['cost_qaoa']
        best_pars=best_line[2]['opt_parameters']
        best_AR=best_cost/E0
        pltpts.append([ep,best_AR,best_pars])

    pltpts.sort()
    pltpts=np.array(pltpts,dtype=object)
    pltpts=np.array(pltpts).transpose()

    np.save('plot_points/kappa_plot_plot_points_noise_aware_{}.npy'.format(hash((path,instance,d,n_iter,p))),pltpts,allow_pickle=True)

    return pltpts

def kappa_plot_points_noise_unaware(unaware_path,instance,n,d,n_iter,direc,p): # Take the optimal angles at p=0, and use them to obtain an AR at other p.
    pltpts=[]
    E0=_HQAOA.get_instance_E0(unaware_path+instance)
    con=_HQAOA.get_instance_con(unaware_path+instance)
    ep_list=get_ep_list(unaware_path,instance,d,n_iter,p,n) # get the ep for which I ran noiseless optimization.
    with ch.no_backprop_mode():
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

    np.save('plot_points/kappa_plot_plot_points_noise_unaware_{}.npy'.format(hash((unaware_path,instance,n,d,n_iter,direc,p))),pltpts)
    return pltpts

def kappa_plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax,_color,n,par_ax):
    # If you made new data points, the npy files should be removed. Only in that case the plot points are computed anew.
    fname='plot_points/kappa_plot_plot_points_noise_aware_{}.npy'.format(hash((direc_path,instance,d,n_iter,p)))
    if not os.path.isfile(fname):
        fname='plot_points/kappa_plot_plot_points_noise_aware_{}.npy'.format(hash(('../../'+direc_path,instance,d,n_iter,p)))
    if os.path.isfile(fname):
        print('Loading plot points')
        points=np.load(fname,allow_pickle=True)
    else:
        print('Calculating plot points')
        points=kappa_plot_points(direc_path,instance,d,n_iter,p,n)
    ax.plot(points[0],points[1],'-o',alpha=1,color=_color,markerfacecolor='none',markersize=7,label=label1,solid_capstyle='round')

    # I have to get the opt angles at p=0.
    opt_unaware_pars=_HQAOA.load_optimal_line(unaware_path+instance, 0, 0, d)[2]['opt_parameters'] # Load optimal parameters at p,ep=0.
    # Calc dist between these angles and the aware angles.
    dists=[]
    for i,_ in enumerate(points[0]):
        dist=_HQAOA.param_dist(opt_unaware_pars,points[2][i])
        dists.append(dist)

    par_ax.plot(points[0],dists,'-o',alpha=1,color=_color,markerfacecolor='none',markersize=7,solid_capstyle='round')

    fname='plot_points/kappa_plot_plot_points_noise_unaware_{}.npy'.format(hash((unaware_path,instance,n,d,n_iter,direc,p)))
    if not os.path.isfile(fname):
        fname='plot_points/kappa_plot_plot_points_noise_unaware_{}.npy'.format(hash(('../../'+unaware_path,instance,n,d,n_iter,direc,p)))
    if os.path.isfile(fname):
        points=np.load(fname)
        print('Loading plot points')
    else:
        print('Calculating plot points')
        points=kappa_plot_points_noise_unaware(unaware_path,instance,n,d,n_iter,direc,p)

    ax.plot(points[0],points[1],'-o',alpha=1,color=_color,markeredgecolor='none',markersize=3.7,label=label2,solid_capstyle='round')

def autoscale_y(ax,margin=0.1):
    """
    From DanHickstein on stackoverflow.
    This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims
    """

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def meta_plotter(instance):
    fig, ax = plt.subplots(8,figsize=(3.75,12), gridspec_kw={'height_ratios': [2, 1,2,1,1,.5,1,.5]})
    master_path='../data/SK/SWAP-network/6/' # path to all instance data.
    path_temp=master_path+'temporal/' # Path to temporal noise correlation data.
    path_spat=master_path+'spatial/' # Path to spatial noise correlation data.
    unaware_path=path_spat # Where to look for data if p=0, for any ep.

    #instance='010010100111110' # Instance name.
    n_iter=4 # Assert later every data point has used n_iter bassinhopping iterations.
    n=6 # Assert later data is from n=6 qubits.
    d=3 # The number of cycles to get the data from.

    colors=[]
    for i in range(10):
        colors.append(next((ax[0]._get_lines.prop_cycler))['color'])
    ax[0].set_ylabel("AR")
    ax[0].set_xlabel("$p$")
    ax[0].grid(visible=True)
    ax[1].grid(visible=True)
    ax[1].set_xlabel("$p$")
    ax[1].set_ylabel("$D$")
    ax[1].text(0.005,1-.025,r'\textbf{(a)}',transform=plt.gcf().transFigure,clip_on=False,in_layout=False)
    ax[1].plot([0,1], [1-.305,1-.305], transform=plt.gcf().transFigure, color='gray',clip_on=False,lw=1)


    #man_p_list=np.around(np.arange(0,0.52,0.02),5)
    man_p_list=[] # Plot all points
    #### Global plot ####
    ep=1 # Correlation at which first and second plots are shown.

    direc_path=path_temp
    direc='temporal'
    label1='Temp. cor., aware'
    label2='Temp. cor., unaware'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,ep,label1,label2,ax[0],colors[1],man_p_list,n,ax[1])

    direc_path=path_spat
    direc='spatial'
    label1='Spat. cor., aware'
    label2='Spat. cor., unaware'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,ep,label1,label2,ax[0],colors[2],man_p_list,n,ax[1])

    ep=0 # Correlation at which third plot is shown.

    direc_path=path_spat
    direc='spatial'
    label1='Uncor., aware'
    label2='Uncor., unaware'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,ep,label1,label2,ax[0],colors[0],man_p_list,n,ax[1])

    #ax.legend(loc='upper left', bbox_to_anchor=(0.0,0.81))
    #ax[0].legend()

    ######### Zoomed in plot ##########

    ax[2].set_ylabel("AR")
    ax[2].set_xlabel('$p$')
    ax[2].grid(visible=True)
    ax[3].grid(visible=True)
    ax[3].set_xlabel('$p$')
    ax[3].set_ylabel('$D$')
    ax[3].text(0.005,1-(.305+0.015),r'\textbf{(b)}',transform=plt.gcf().transFigure,in_layout=False)
    ax[3].plot([0,1],[1-.605,1-.605],transform=plt.gcf().transFigure, color='gray',clip_on=False,lw=1)
    ep=1 # Correlation at which first and second plots are shown.

    direc_path=path_temp
    direc='temporal'
    label1='Temp. cor., aware'
    label2='Temp. cor., unaware'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,ep,label1,label2,ax[2],colors[1],man_p_list,n,ax[3])

    direc_path=path_spat
    direc='spatial'
    label1='Spat. cor., aware'
    label2='Spat. cor., unaware'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,ep,label1,label2,ax[2],colors[2],man_p_list,n,ax[3])

    ep=0 # Correlation at which third plot is shown.

    direc_path=path_spat
    direc='spatial'
    label1='Uncor., aware'
    label2='Uncor., unaware'

    plotter(direc_path,direc,instance,unaware_path,n_iter,d,ep,label1,label2,ax[2],colors[0],man_p_list,n,ax[3])

    ax[2].set_xlim(left=-0.0007,right=0.015)
    ax[3].set_xlim(left=-0.0007,right=0.015)
    autoscale_y(ax[2],margin=0.1)

    ### Kappa plot ####

    p=0.001 # Error probability at which plots are shown.

    ax[4].text(0.005,1-(.605+0.015),r'\textbf{(c)}',transform=plt.gcf().transFigure,in_layout=False)
    ax[4].set_xlabel('$\kappa$')
    ax[4].set_ylabel("AR")
    ax[4].grid(visible=True)
    ax[4].set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax[5].set_xlabel('$\kappa$')
    ax[5].set_ylabel("$D$")
    ax[5].grid(visible=True)
    ax[5].set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax[5].plot([0,1],[1-.8,1-.8],transform=plt.gcf().transFigure, color='gray',clip_on=False,lw=1)

    direc_path=path_temp
    direc='temporal'
    label1='Temp.\n aware'
    label2='Temp.\n unaware'

    kappa_plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax[4],colors[1],n,ax[5])

    direc_path=path_spat
    direc='spatial'
    label1='Spat.\n aware'
    label2='Spat.\n unaware'

    kappa_plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax[4],colors[2],n,ax[5])

    p=0.01 #

    ax[3].text(0.005,1-(.8+0.015),r'\textbf{(d)}',transform=plt.gcf().transFigure,in_layout=False)
    ax[6].set_xlabel('$\kappa$')
    ax[6].set_ylabel("AR")
    ax[6].grid(visible=True)
    ax[6].set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax[7].set_xlabel('$\kappa$')
    ax[7].set_ylabel("$D$")
    ax[7].grid(visible=True)
    ax[7].set_xticks([0,0.2,0.4,0.6,0.8,1])

    direc_path=path_temp
    direc='temporal'
    label1='_nolegend_'
    label2='_nolegend_'

    kappa_plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax[6],colors[1],n,ax[7])

    direc_path=path_spat
    direc='spatial'
    label1='_nolegend_'
    label2='_nolegend_'

    kappa_plotter(direc_path,direc,instance,unaware_path,n_iter,d,p,label1,label2,ax[6],colors[2],n,ax[7])

    ### Savefig ###
    fig.tight_layout()
    fig.savefig('plots/plot_{}.pdf'.format(instance),bbox_inches='tight',pad_inches=0.005)

master_path='../data/SK/SWAP-network/6/' # path to all instance data.
path_temp=master_path+'temporal/'

#lst=os.listdir(path_temp)
#lst=[[int(i),i] for i in lst if os.path.isdir(path_temp+i)]
#lst.sort()
#print([i[1] for i in lst])

if __name__=='__main__':
    for instance in os.listdir(path_temp):
        if os.path.isdir(path_temp+instance):
            print('plotting instance',instance)
            meta_plotter(instance)
