"""
Defs. for running HQAOA.py.
"""
class Name: # Simple namespace class
    pass

import numpy # For the cases where GPU==True and we still want to use numpy.
import qem
import chainer as ch
from datetime import datetime
import argparse
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from itertools import product
from matplotlib import rc
import matplotlib as mpl
import os
from time import time

try: # Use GPU if CuPy installation is available. 
    import cupy as xp
    GPU=True
except ImportError:
    import numpy as xp
    GPU=False

def get_command_line_input():
    parser=argparse.ArgumentParser(description='Run QAOA using the quantum emulator qem.')
    
    # Obligatory arguments
    parser.add_argument('path',type=str,help='The path to the folder where output should be placed.')
    parser.add_argument('d',type=int,help='The number of cycles to be used in QAOA.')
    parser.add_argument('p',type=float,help='The probability of error.')
    parser.add_argument('ep',type=float,help='(Parameter proportional to) the correlation strength of the error.')
    parser.add_argument('direction',type=str,help='This must me set to either temporal or spatial. If temporal, errors are time-correlated. If spatial, errors are spatialy correlated.')

    # Optional arguments
    parser.add_argument('--n_iter','-i',type=int,default=0,help='Number of iterations of the basinhopping routine.')
    parser.add_argument('--par_mul',type=str,default='opgt',help="Use one parameter per cycle per gate type if set to 'opgt' (default). Use one parameter per gate instance when set to 'opgi'" )

    # Parse the args.
    cmd_args=parser.parse_args()
    if cmd_args.path[-1]=='/':
        cmd_args.path=cmd_args.path[:-1]

    return cmd_args

def import_instance_data(ipath): # Not tested
    """
    Load data from ipath/output.txt. First line is skipped. 
    """
    with open(ipath+'/output.txt','r') as f:
        f.readline()
        _data=f.readlines()
    data=[]
    for line_nr,line in enumerate(_data):
        if line !='\n':
            try:
                new_line=eval(line.strip())
                data.append(new_line)
            except:
                print('WARNING: corruption found in line {} of {}/output.txt'.format(line_nr+2,ipath))

    return data

def get_instance_E0(ipath): # Not tested
    with open(ipath+'/ground_state.txt','r') as f:
        Elst=f.readlines()
        Elst=[eval(x.strip()) for x in Elst]
        Elst=[x[0] for x in Elst]
        Elst.sort()
        return float(Elst[0])
    
def get_instance_con(ipath): # Not tested
    with open(ipath+'/graph_input.txt', 'r') as file:
        con=eval(file.readline())
    return con

def load_optimal_line(ipath, p, ep, d): # not in test_HQAOA!!!! 
    """
    Load the optimal line from instance path ipath, given the p, ep and d.
    """
    data=import_instance_data(ipath)
    data=[line for line in data if line[0]['ep']==ep and line[0]['p']==p and line[0]['d']==d]
    if len(data)==0:
        print('No entries found at ipath={}, p={},ep={},d={}'.format(ipath,p,ep,d))
    data.sort(key=lambda x: x[2]['cost_qaoa'])
    return data[0]
 
    
def param_dist(pars1,pars2): 
    """
    Returns the euclidian distance between pars1 and pars2 modulo all the where any ambibulity in the angles (caused by symmetries of the cost function) is removed by wrap_parameters. 
    """
    if type(pars1)==ch.Variable:
        _pars1=pars1.data.copy()
    elif type(pars1)==list:
        _pars1=xp.array(pars1).copy()
    elif type(pars1)==xp.ndarray:
        _pars1=pars1.copy()
        
    if type(pars2)==ch.Variable:
        _pars2=pars2.data.copy()
    elif type(pars2)==list:
        _pars2=xp.array(pars2).copy()
    elif type(pars2)==xp.ndarray:
        _pars2=pars2.copy()
        
    _pars1=wrap_parameters(_pars1)
    _pars2=wrap_parameters(_pars2)
    dif=_pars1-_pars2
    
    return xp.linalg.norm(dif)

def wrap_parameters(pars): 
    """
    The SK QAOA cost function has symmetries, as described in the preprint. The fn wrap_parameters() maps all parameters to a range in which truely differeent parameters are different, and were sets of parameters related by symmetries are maped to the same set of paramers. The euclid length is the gammas is minimized. Note that 2*gamma_here=gamma_preprint.
    
    Parameters:
    -----------
    pars : ch.Variable or xp.array
    
    Returns:
    --------
    pars : xp.array
    
    """             
    if type(pars)==ch.Variable:
        pars1=pars.data.copy()
    elif type(pars)==list:
        pars1=xp.array(pars).copy()
    elif type(pars)==xp.ndarray:
        pars1=pars.copy()
        
    r=len(pars1)
    assert r%2==0
        
    def zipper(parsz):
        for i in range(0,r,2):
            parsz[i]=parsz[i]%xp.pi
            if parsz[i]>=xp.pi/2:
                parsz[i]-=xp.pi/2
                parsz[i+1]=-1*parsz[i+1]
                if i<r-2:
                    parsz[i+2]-=xp.pi/2
            parsz[i+1]=parsz[i+1]%xp.pi
        return parsz
    
    def euclid_length_gammas(pars):
        gammas=pars[::2]
        return xp.linalg.norm(gammas)
        
    pars2=-1*pars1
    
    pars1=zipper(pars1)
    pars2=zipper(pars2)
    
    if euclid_length_gammas(pars2)<euclid_length_gammas(pars1):
        return pars2
    else:
        return pars1

       
def transition_CY_channel_classical_quantum(p,ep,edge): # Return the channel composed of 1) a classical transition channel with parameters (p,ep) on bit edge[0], 2) a CY gate on the edge, with edge[0] the control and edge[1] the target.
    re=xp.array([[ep+(1-ep)*(1-p),(1-ep)*(1-p)],[(1-ep)*p,ep+(1-ep)*p]],dtype=xp.float64)
    im=xp.zeros((2,2),dtype=xp.float64)
    array=qem.Array(re,im)
    gate=qem.Gate(array,'Transition'+'('+str(p)+','+str(ep)+')')
    action=qem.Action(edge[0],gate)
    cir=qem.Cir()
    cir.append_action(action)

    action=qem.Action(edge,qem.CY())
    cir.append_action(action)

    chan=qem.Chan()
    chan.append_circuit(cir)

    return chan

def transition_channel(p,ep,q):
    """
    Returns the follwing channel: flip the qubit q with probability p. Reinitialize to [[1-p,0],[0,p]] with probability (1-ep). This implements a markov chain on the state space [0,1] with transition matrix 
    T=
    [[ 1-p+ep*p, 1-p-ep+ep*p],
     [ p-ep*p  , ep+p-ep*p  ]]. 
    This transtition matrix has a steady state [1-p,p] and lag-1 correlation strength ep.
    """
    chan=qem.Chan()
    chan.name='Transition'

    cir=qem.Cir()
    re=xp.sqrt(1-p+ep*p)*xp.array([[1.,0.],[0.,0.]])
    im=xp.zeros((2,2))
    array=qem.Array(re,im)
    gate=qem.Gate(array,'sqrt(1-p+ep*p)|0><0|')
    action=qem.Action(q,gate)
    cir.append_action(action)
    chan.append_circuit(cir)

    if ep!=1: # Condition to prevent evaluating sqrt at incorrectly rounded off argument.
        cir=qem.Cir()
        re=xp.sqrt(1-p-ep+ep*p)*xp.array([[0.,1.],[0.,0.]])
        im=xp.zeros((2,2))
        array=qem.Array(re,im)
        gate=qem.Gate(array,'sqrt(1-p-ep+ep*p)|0><1|')
        action=qem.Action(q,gate)
        cir.append_action(action)
        chan.append_circuit(cir)

    if p!=0 and ep!=1: # Idem
        cir=qem.Cir()
        re=xp.sqrt(p-ep*p)*xp.array(  [[0.,0.],[1.,0.]]  )
        im=xp.zeros((2,2))
        array=qem.Array(re,im)
        gate=qem.Gate(array,'sqrt(p-ep*p)|1><0|')
        action=qem.Action(q,gate)
        cir.append_action(action)
        chan.append_circuit(cir)

    if not (p==0 and ep==0):
        cir=qem.Cir()
        re=xp.sqrt(ep+p-ep*p)*xp.array( [[0.,0.],[0.,1.]] )
        im=xp.zeros((2,2))
        array=qem.Array(re,im)
        gate=qem.Gate(array,'sqrt(ep+p-ep*p)|1><1|')
        action=qem.Action(q,gate)
        cir.append_action(action)
        chan.append_circuit(cir)

    return chan

#def cost(complete_graph,n,mixreg): # Depricated. Uses old imput format with complete_graph instead of connectivity matrix con.
#    """
#    Given the list of edges in 'graph', compute the cost associated with the state of reg.
#
#    Returns
#    -------
#    E : ch.Variable
#    """
#    E=0.
#
#    def add_term(edge):
#        nonlocal E
#        temp_mixreg=qem.MixReg(n)
#        temp_mixreg.psi=mixreg.psi
#        qem.apply_action(qem.Action(edge[:2],qem.Ising(edge[2])),temp_mixreg)
#        qem.apply_channel(qem.Trace(tuple(range(n))),temp_mixreg)
#        E+=temp_mixreg.psi.re
#
#    for edge in complete_graph:
#        assert len(edge)==3 and (len(edge[2])==1 or len(edge[2])==3)
#        add_term(edge)
#
#    return E

def cost_con(con,qord,n,mixreg):
    """
    Given the list of edge weights in the connectivity matrix con, and the qubit order qord, compute the cost associated with the state of reg.

    This function is like _HQAOA.cost, but takes con as an argument instead of complete_graph, and takes an extra argument qord.

    Returns
    -------
    E : ch.Variable
    """
    E=0.
    assert type(qord)==list

    def add_term(i,j): # Add the energy of the effective edge (i,j).
        nonlocal E
        temp_mixreg=qem.MixReg(n)
        temp_mixreg.psi=mixreg.psi
        phys_ed=(qord.index(i),qord.index(j))
        w=con[i][j]
        qem.apply_action(qem.Action(phys_ed,qem.Ising([w])),temp_mixreg)
        qem.apply_channel(qem.Trace(tuple(range(n))),temp_mixreg)
        E+=temp_mixreg.psi.re

    for i in range(n):
        for j in range(i+1,n):
            add_term(i,j)

    return E

def ansatz_phys_swap_sc_temporal_quantum_classical(con,n,d,p,ep,pars):
    '''
    Returns the super circuit that is to be applied to the all-zero state of n mixed quantum bits and n classical bits, modeling temporal noise correlations. The supercircuit is a noisy implementation of QAOA using a swap-network.
    The form of pars determines whether we do a circuit with one parameter per gate type per cycle or one parameter per gate. If it is of the form [a,b,c,...] we do one parameter per cycle per gate type. If it is of the form [(a,b,c,..),(d,e,f,...)], we use one parameter per gate instance.

    '''
    def app_RIsingSWAP(edge,weight,par,psc): # Append RIsing channel to a supercir psc on the litteral qubits edge. Swap the literal qubits in edge.
        cir=qem.Cir()
        gate=qem.RIsingSWAP([weight],par)
        cir.append_action(qem.Action(edge,gate))
        chan=qem.Chan([cir])
        psc.append_channel(chan)

    def app_phys_RIsing(q,par,con,psc): # Append RIsing gate on logical qubit q and q+1 to the physical super circuit. Apply the error to the right qubits.
        eff_ed=[psc.qord[q],psc.qord[q+1]]
        weight=con[eff_ed[0]][eff_ed[1]]
        phys_ed=(q,q+1)
        app_RIsingSWAP(phys_ed,weight,par,psc)
        psc.qord[q],psc.qord[q+1]=psc.qord[q+1],psc.qord[q]
        pee=(q+n,q) # Physical error edge for left logical qubit
        psc.append_channel(transition_CY_channel_classical_quantum(p,ep,pee))
        pee=(q+1+n,q+1) # Same for right logical qubit
        psc.append_channel(transition_CY_channel_classical_quantum(p,ep,pee))

    def app_brickwork(par,con,psc): # Make brickwork SWAP-network structure for evo along Ham. If  par is a list, use one parameter per gate instance.
        ipar=0 # Needed because chainer Variables do not support pop.
        assert len(par.shape)==0 or len(par)==(n-1)*((n-1)+1)/2, 'Wrong number of parameters.'
        for i in range(0,n,2): # Swap network needs n layers
            for q in range(0,n,2): # Even layer
                if len(par.shape)!=0:
                    par_=par[ipar] # (Chainer Variables do not have a pop function.)
                    ipar+=1
                else:
                    par_=par
                app_phys_RIsing(q,par_,con,psc)

            for q in range(1,n-1,2): # Uneven layer
                if len(par.shape)!=0:
                    par_=par[ipar] # (Chainer Variables do not have a pop function.)
                    ipar+=1
                else:
                    par_=par
                app_phys_RIsing(q,par_,con,psc)

    def app_mix(par,psc): # Make mixer structure. If par is a list, use one parameter per gate instance.
        assert len(par.shape)==0 or len(par)==n, 'Wrong number of parameters.'
        for q in range(n):
            if len(par.shape)!=0:
                par_=par[q]
            else:
                par_=par
            app_phys_mix(q,par_,psc)

    def app_phys_mix(q,par,psc): # Append mixer on logical qubit q to the supercir, containing errors.
        phys_q=q
        cir=qem.Cir()
        gate=qem.RX(par)
        cir.append_action(qem.Action(phys_q,gate))
        chan=qem.Chan([cir])
        psc.append_channel(chan)

        pee=(phys_q+n,phys_q)
        psc.append_channel(transition_CY_channel_classical_quantum(p,ep,pee))

    psc=qem.SupCir(n,n)
    psc.qord=list(range(n))

    for phys_q in range(n): # Make init state
        psc.append_channel(qem.H_channel(phys_q))
        psc.append_channel(qem.prepare_classical_mixed_state_channel_classical(p,phys_q+n))

    for i in range(0,2*d,2): # Repeat the cycle d times
        app_brickwork(pars[i],con,psc)
        app_mix(pars[i+1],psc)

    return psc

def ansatz_phys_swap_sc_temporal(con,n,d,p,ep,pars):
    '''
    Returns the physical super circuit that is to be applied to the all-zero state, modeling temporal noise correlations. The supercircuit is a noisy implementation of QAOA using a swap-network. The sc outputted is defined on 2*n qubits, where qubit 0 is a data qubit, qubit 1 is an aux qubit, etc. Note the connectivity matrix con should not be defined in this way.
    '''
    def app_RIsingSWAP(edge,weight,par,psc): # Append RIsing channel to a supercir psc on the litteral qubits edge. Swap the literal qubits in edge.
        cir=qem.Cir()
        gate=qem.RIsingSWAP([weight],par)
        cir.append_action(qem.Action(edge,gate))
        chan=qem.Chan([cir])
        psc.append_channel(chan)

    def app_transitionCY(edge,psc): # Append transition channel to literal qubit edge[1], append CY channel to litteral qubits edge[1] (control) and edge[0] (target):
        psc.append_channel(transition_channel(p,ep,edge[1]))
        psc.append_channel(qem.CY_channel(edge[1],edge[0]))

    def app_phys_RIsing(q,par,con,psc): # Append RIsing gate on logical qubit q and q+1 to the physical super circuit. Apply the error to the right qubits.
        eff_ed=[psc.qord[q],psc.qord[q+1]]
        weight=con[eff_ed[0]][eff_ed[1]]
        phys_ed=(2*q,2*(q+1))
        app_RIsingSWAP(phys_ed,weight,par,psc)
        psc.qord[q],psc.qord[q+1]=psc.qord[q+1],psc.qord[q]
        pee=(2*q,2*q+1) # Physical error edge for left logical qubit
        app_transitionCY(pee,psc)
        pee=(2*q+2,2*q+3) # Same for right logical qubit
        app_transitionCY(pee,psc)

    def app_brickwork(par,con,psc): # Make brickwork SWAP-network structure for evo along Ham.
        for i in range(0,n,2): # Swap network needs n layers
            for q in range(0,n,2): # Even layer
                app_phys_RIsing(q,par,con,psc)
            for q in range(1,n-1,2): # Uneven layer
                app_phys_RIsing(q,par,con,psc)

    def app_phys_mix(q,par,psc): # Append mixer on logical qubit q to the supercir, containing errors.
        phys_q=2*q
        cir=qem.Cir()
        gate=qem.RX(par)
        cir.append_action(qem.Action(phys_q,gate))
        chan=qem.Chan([cir])
        psc.append_channel(chan)

        pee=(phys_q,phys_q+1)
        app_transitionCY(pee,psc)

    psc=qem.SupCir(2*n,0)
    psc.qord=list(range(n))

    for phys_q in range(0,2*n,2): # Make init state
        psc.append_channel(qem.H_channel(phys_q))
        psc.append_channel(qem.prepare_classical_mixed_state_channel(p,phys_q+1))

    for i in range(0,2*d,2): # Repeat the cycle d times
        app_brickwork(pars[i],con,psc)
        for q in range(n):
            app_phys_mix(q,pars[i+1],psc)

    return psc

def ansatz_phys_swap_sc_spatial(con,n,d,p,ep,pars,error_loc='all'):
    '''
    Returns the physical super circuit that is to be applied to the all-zero state, modeling spatial noise correlations. The supercircuit is a noisy implementation of QAOA using a swap-network. The sc outputted is defined on n+1 qubits, where qubit 0 to (n-1) are data qubits, and qubit n is the aux qubit. Note the connectivity matrix con should not be defined in this way.
    Is equal to ansat_phys_sc_temporal, but all aux qubits are now qubit n.
    
    If a error_loc (noise location) is specified, error_loc=list with all entries an int, the error is only applied at the jth interaction with the fluctuator if j is in list. Counting starts at 0.
    
    Note that for injecting temporally correlated errors we also use ansatz_phys_swap_sc_spatial. This is because this function is faster (requires less aux qubits.) And 'temporal correlation' is caused by the manual injection. 
    
    '''
    if error_loc!='all':
        assert type(error_loc)==list, 'error_loc must be list'
        assert all([type(ent)==int for ent in error_loc]), 'All entries of error_loc must be int'
        assert p==1, 'You want noise to occur with certainty if you insert it at a certain location.'
    ic=0 # Interaction count
        
    def app_RIsingSWAP(edge,weight,par,psc): # Append RIsing channel to a supercir psc on the litteral qubits edge. Swap the literal qubits in edge.
        cir=qem.Cir()
        gate=qem.RIsingSWAP([weight],par)
        cir.append_action(qem.Action(edge,gate))
        chan=qem.Chan([cir])
        psc.append_channel(chan)

    def app_transitionCY(edge,psc,error_loc): # Append transition channel to literal qubit edge[1], append CY channel to litteral qubits edge[1] (control) and edge[0] (target):
        nonlocal ic
        psc.append_channel(transition_channel(p,ep,edge[1]))
        if error_loc=='all' or ic in error_loc:
            psc.append_channel(qem.CY_channel(edge[1],edge[0]))
        ic+=1

    def app_phys_RIsingSWAP(q,par,con,psc): # Append RIsing gate on logical qubit q and q+1 to the physical super circuit. Apply the error to the right qubits.
        eff_ed=[psc.qord[q],psc.qord[q+1]]
        weight=con[eff_ed[0]][eff_ed[1]]
        phys_ed=(q,q+1)
        app_RIsingSWAP(phys_ed,weight,par,psc)
        psc.qord[q],psc.qord[q+1]=psc.qord[q+1],psc.qord[q]
        pee=(q,n) # Physical error edge for left logical qubit
        app_transitionCY(pee,psc,error_loc)
        pee=(q+1,n) # Same for right logical qubit
        app_transitionCY(pee,psc,error_loc)

    def app_brickwork(par,con,psc): # Make brickwork SWAP-network structure for evo along Ham.
        for i in range(0,n,2): # Swap network needs n layers
            for q in range(0,n,2): # Even layer
                app_phys_RIsingSWAP(q,par,con,psc)
            psc.append_channel(qem.prepare_classical_mixed_state_channel(p,n)) # Clean up afterwards
            for q in range(1,n-1,2): # Uneven layer
                app_phys_RIsingSWAP(q,par,con,psc)
            psc.append_channel(qem.prepare_classical_mixed_state_channel(p,n))

    def app_phys_mix(q,par,psc,error_loc): # Append mixer on logical qubit q to the supercir, containing errors.
        phys_q=q
        cir=qem.Cir()
        gate=qem.RX(par)
        cir.append_action(qem.Action(phys_q,gate))
        chan=qem.Chan([cir])
        psc.append_channel(chan)

        pee=(phys_q,n)
        app_transitionCY(pee,psc,error_loc)

    psc=qem.SupCir(n+1,0)
    psc.qord=list(range(n))

    for phys_q in range(0,n): # Make init state
        psc.append_channel(qem.H_channel(phys_q))
    psc.append_channel(qem.prepare_classical_mixed_state_channel(p,n))

    for i in range(0,2*d,2): # Repeat the cycle d times
        app_brickwork(pars[i],con,psc)
        for q in range(n):
            app_phys_mix(q,pars[i+1],psc,error_loc)
        psc.append_channel(qem.prepare_classical_mixed_state_channel(p,n))

    return psc

def cost_from_parameters_swap_temporal_quantum_classical(con,n,d,p,ep,pars):
    """
    Return the cost of a state as a fn of the parameters. If enteries of pars are ch.Variable array, use one parameter per gate,

    Returns
    -------
    E : chainer.Variable
    """
    assert type(pars)==ch.Variable or type(pars)==list
    assert len(pars)==2*d

    psc=ansatz_phys_swap_sc_temporal_quantum_classical(con,n,d,p,ep,pars)
    preg=qem.MixReg(n,n)
    qem.apply_supcir(psc,preg)
    for i in range(n):
        qem.apply_channel(qem.ClasTrace([n]),preg)
    E=cost_con(con,psc.qord,n,preg)
    return E

def cost_from_parameters_swap_temporal(con,n,d,p,ep,pars):
    """
    Return the cost of a state as a fn of the parameters. 

    Returns
    -------
    E : chainer.Variable
    """
    assert type(pars)==ch.Variable
    assert len(pars)==2*d

    psc=ansatz_phys_swap_sc_temporal(con,n,d,p,ep,pars)
    preg=qem.MixReg(2*n)
    qem.apply_supcir(psc,preg)
    auxqubits=list(range(1,2*n,2))
    qem.apply_channel(qem.Trace(auxqubits),preg)
    E=cost_con(con,psc.qord,n,preg)

    return E

def cost_from_parameters_swap_spatial(con,n,d,p,ep,pars,error_loc='all'):
    """
    Return the cost of a state as a fn of the parameters. If error_loc=list is specified, the CY is only aplied at qunit-fluctuator interaction number j if j is in list. Note that if error_loc!='all', p must be set to 1. 

    Returns
    -------
    E : chainer.Variable
    """
    assert type(pars)==ch.Variable
    assert len(pars)==2*d
    if error_loc!='all':
        assert p==1, 'If you insert an error manually, you want it to be inserted with certainty.'

    psc=ansatz_phys_swap_sc_spatial(con,n,d,p,ep,pars,error_loc=error_loc)
    preg=qem.MixReg(n+1)
    qem.apply_supcir(psc,preg)
    auxqubits=(n,)
    qem.apply_channel(qem.Trace(auxqubits),preg)
    E=cost_con(con,psc.qord,n,preg)

    return E

def infidelity_from_parameters_swap_temporal(con,n,d,p,ep,pars,grdsp):
   """
   Return the infidelity of a state, as given by 'con','d', 'p', 'ep', and 'pars', with the true ground state as described by the groundspace 'grdsp'. The groundspace should be of the form e.g. [[E0,[1,1,-1]],[E0,[-1,-1,1]]].
   """
   assert type(pars)==ch.Variable
   assert len(pars)==2*d
   assert type(con)==numpy.ndarray

   # Create a reg with the right state
   psc=ansatz_phys_swap_sc_temporal(con,n,d,p,ep,pars)
   preg=qem.MixReg(2*n)
   qem.apply_supcir(psc,preg)
   auxqubits=tuple(range(1,2*n,2))
   qem.apply_channel(qem.Trace(auxqubits),preg)
   reg=preg
   del preg

   # Note that, upon measurement of rho, the prob that we obtain one of the optimal solutions {|GS_i>}_i is sum_i <GS_i|rho|GS_i> = sum_i rho_{x_i,x_i}, where x_i is the bitstring associated with |GS_i>. So to compute the prob of obtaining an optimal state upon measurement of all qubits, we just need to sum the corresponding terms on the diagonal of rho.

   grdspst=[i[1] for i in grdsp] # Ground space states only.

   for i in range(len(grdspst)): # Convert from +/- 1 conventon to 0/1 convention.
       for j in grdspst[i]:
           assert type(j)==int and (j==1 or j==-1), "Wrong entries found in the description of states in the ground space."
       grdspst[i]=[(-el+1)//2 for el in grdspst[i]]

    
    # Reorder the states in the grdsp according to the order of the qubits induced by the swaps. 
    
   ord_grdspst=[]
   for state in grdspst:
        ordered=[state[q] for q in psc.qord]
        ord_grdspst.append(ordered)
    

   prob=0.
   for st in ord_grdspst:
       ent=(*st,*st)
       qem.round_assert(reg.psi.im.array[ent],0.)
       prob+=reg.psi.re.array[ent]

   infidel=1-prob # My def of infidelity.

   return infidel

def infidelity_from_parameters_swap_temporal_quantum_classical(con,n,d,p,ep,pars,grdsp):
   """
   Return the infidelity of a state, as given by 'con','d', 'p', 'ep', and 'pars', with the true ground state as described by the groundspace 'grdsp'. The groundspace should be of the form e.g. [[E0,[1,1,-1]],[E0,[-1,-1,1]]]. This uses the new quantum_classical functionality of qem.
   """
   assert type(pars)==ch.Variable
   assert len(pars)==2*d
   assert type(con)==numpy.ndarray

   # Create a reg with the right state
   psc=ansatz_phys_swap_sc_temporal_quantum_classical(con,n,d,p,ep,pars)
   preg=qem.MixReg(n,n)
   qem.apply_supcir(psc,preg)
   for i in range(n):
       qem.apply_channel(qem.ClasTrace([n]),preg)
   reg=preg
   del preg

   # Note that, upon measurement of rho, the prob that we obtain one of the optimal solutions {|GS_i>}_i is sum_i <GS_i|rho|GS_i> = sum_i rho_{x_i,x_i}, where x_i is the bitstring associated with |GS_i>. So to compute the prob of obtaining an optimal state upon measurement of all qubits, we just need to sum the corresponding terms on the diagonal of rho.

   grdspst=[i[1] for i in grdsp] # Ground space states only.

   for i in range(len(grdspst)): # Convert from +/- 1 conventon to 0/1 convention.
       for j in grdspst[i]:
           assert type(j)==int and (j==1 or j==-1), "Wrong entries found in the description of states in the ground space."
       grdspst[i]=[(-el+1)//2 for el in grdspst[i]]

    
    # Reorder the states in the grdsp according to the order of the qubits induced by the swaps. 
   ord_grdspst=[]
   for state in grdspst:
        ordered=[state[q] for q in psc.qord]
        ord_grdspst.append(ordered)
    
   prob=0.
   for st in ord_grdspst:
       ent=(*st,*st)
       qem.round_assert(reg.psi.im.array[ent],0.)
       prob+=reg.psi.re.array[ent]

   infidel=1-prob # My def of infidelity.

   return infidel

def infidelity_from_parameters_swap_spatial(con,n,d,p,ep,pars,grdsp):
    """
   Return the infidelity of a state, as given by 'con','d', 'p', 'ep', and 'pars', with the true ground state as described by the groundspace 'grdsp'. The groundspace should be of the form e.g. [[E0,[1,1,-1]],[E0,[-1,-1,1]]].
    """
    assert type(pars)==ch.Variable
    assert len(pars)==2*d

    # Create a reg with the right state
    psc=ansatz_phys_swap_sc_spatial(con,n,d,p,ep,pars)
    preg=qem.MixReg(n+1)
    qem.apply_supcir(psc,preg)
    auxqubits=(n,)
    qem.apply_channel(qem.Trace(auxqubits),preg)
    reg=preg
    del preg

    # Note that, upon measurement of rho, the prob that we obtain one of the optimal solutions {|GS_i>}_i is sum_i <GS_i|rho|GS_i> = sum_i rho_{x_i,x_i}, where x_i is the bitstring associated with |GS_i>. So to compute the prob of obtaining an optimal state upon measurement of all qubits, we just need to sum the corresponding terms on the diagonal of rho.

    grdspst=[i[1] for i in grdsp] # Ground space states only.

    for i in range(len(grdspst)): # Convert from +/- 1 conventon to 0/1 convention.
        for j in grdspst[i]:
            assert type(j)==int and (j==1 or j==-1), "Wrong entries found in the description of states in the ground space."
        grdspst[i]=[(-el+1)//2 for el in grdspst[i]]

    # Reorder the states in the grdsp according to the order of the qubits induced by the swaps.   
    ord_grdspst=[]
    
    for state in grdspst:
        ordered=[state[q] for q in psc.qord]
        ord_grdspst.append(ordered)
    
    prob=0.
    for st in ord_grdspst:
        ent=(*st,*st)
        qem.round_assert(reg.psi.im.array[ent],0.)
        prob+=reg.psi.re.array[ent]

    infidel=1-prob # My def of infidelity.

    return infidel

def run_QAOA(cmd_args,run_args):
    """
    Run QAOA. Returns namespace instance with atributes n_fn_calls, local_min_list,local_min_par_list,local_min_accept_list,init_par,cost_qaoa,opt_parameters.
    """
    qaoa_out=Name()
    qaoa_out.n_fn_calls=0
    qaoa_out.local_min_list=[]
    qaoa_out.local_min_par_list=[]
    qaoa_out.local_min_accept_list=[]

    def ccost(pars): # Pars go in as numpy array.
        nonlocal qaoa_out
        nonlocal cmd_args
        nonlocal run_args
        tmp=Name()
        pars=ch.Variable(xp.array(pars))

        if cmd_args.direction=='temporal':
            cost=cost_from_parameters_swap_temporal_quantum_classical(run_args.con,run_args.n,cmd_args.d,cmd_args.p,cmd_args.ep,pars)
        elif cmd_args.direction=='spatial':
            cost=cost_from_parameters_swap_spatial(run_args.con,run_args.n,cmd_args.d,cmd_args.p,cmd_args.ep,pars)
        cost.backward()
        grad=pars.grad
        qaoa_out.n_fn_calls+=1
        print('.',end='',flush=True) #Progress indicator. One dot per function call. Here one function call is defined as one forward and one backward evaluation.
        
        if run_args.GPU==True:
            cost=cost.array.get()
            g=g.get()
        elif run_args.GPU==False:
            cost=cost.array

        return cost, grad

    def callback(x,f,accept):
        nonlocal qaoa_out
        print('\nNew local min for', vars(cmd_args))
        print('cost=',float(f),'accepted=',accept,'parameters=',list(x))
        qaoa_out.local_min_list.append(float(f))   
        qaoa_out.local_min_par_list.append(list(x))
        qaoa_out.local_min_accept_list.append(accept)

    qaoa_out.init_par=numpy.random.rand(cmd_args.d*2)/1000-1/2000  #! This is truely different from

    sol=scipy.optimize.basinhopping(ccost,qaoa_out.init_par,minimizer_kwargs={'jac':True},niter=cmd_args.n_iter,callback=callback)
    qaoa_out.cost_qaoa=float(sol.fun)
    qaoa_out.opt_parameters=sol.x.tolist()
    qaoa_out.init_par=list(qaoa_out.init_par)

    return qaoa_out

def generate_random_SK_model_SWAP_network_graph_input(path,n):
    '''
    Generates a graph_input.txt for a SWAP-network QAOA ansatz solving for the ground state of a random SK model. Note path is relative to the path the python script is called from. Note graph_input.txt has different contents compared to graph_input.txt in HVQE. Here, we just store the matrix that defines the SK graph.
    '''
    con=numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i+1,n):
            con[i,j]=numpy.random.randint(2)*2-1
            con[j,i]=con[i,j]

    weights=[con[i,j] for i in range(n) for j in range(i+1,n)]
    weights=[(-weight+1)//2 for weight in weights]
    string=''
    for weight in weights:
        string+=str(weight)
    fldr_name=path+'/'+string

    try:
        os.mkdir(fldr_name)
    except FileExistsError:
        print("!!! Attempt of creating",fldr_name,'twice!!! Prob. of this happening was low, but it should be accounted for in the weighing of results.')

    file_name=fldr_name+'/'+'graph_input.txt'
    with open(file_name,'w') as f:
        f.write(str(con.tolist()))

    print('Created the directory', fldr_name, 'with graph input for a random SK model on',n,'spins')

def Ising_ground_state(con):
    """
    Brute-force solve for the states with the lowest energy. Note that if degeneraceis occur, all states belonging to that single cost are returned ,in e.g. (4 qubits) the form [cost, [1,0,1,1],same cost, [0,1,0,0]]. The list is a regular python list.
    """
    assert type(con)==numpy.ndarray
    n=con.shape[0]

    def all_bitstrings():
          return list(map(list, product([1, -1], repeat=n)))

    def ccost(bitstr):
        cost=0.
        for i in range(n):
            for j in range(i+1,n):
                cost+=bitstr[i]*bitstr[j]*con[i][j]
        return cost

    best=[]
    for x in all_bitstrings():
        cost=ccost(x)
        if len(best)==0:
            best.append([cost,x])
        elif all(cost < y for y in [pair[0] for pair in best]):
            best=[[cost,x]]
        elif all(cost <= y for y in [pair[0] for pair in best]):
            best.append([cost,x])

    return best

def export_Ising_ground_state(path): # Compute the ground state of con, as defined in path/ground_state.txt, and put the result in path/graph_input.txt
    # Load the ansatz from graph_input.txt

    if path[-1]=='/':
        path=cmd_args.path[:-1]

    with open(path+'/graph_input.txt', 'r') as file:
        con=eval(file.readline())
        listq=type(con)==list
        rowq=all(type(row)==list for row in con)
        entq=all(type(ent)==int for row in con for ent in row)
        con=numpy.array(con,dtype=int) # Convert to numpy.array
        twodq=(len(con.shape)==2)
        squareq=(con.shape[0]==con.shape[1])
        orthq=(con.transpose()==con)
        assert all([listq,rowq,entq,twodq,squareq]), 'Input file improperly formatted.'

    print('')
    print('--------------------------------------------------')
    print('Computing the Ising ground space of',path+'/graph_input.txt')
    start=time()
    out=Ising_ground_state(con)
    if qem.GPU==True: qem.sync()
    end=time()
    print('Solution found in',end-start,'seconds')
    print('Ground space =',out)
    print('Output saved in',path+'/ground_state.txt')
    print('--------------------------------------------------')
    print(' ')

    ## Write gs energy to disk
    with open(path+"/ground_state.txt", "w") as f:
        for el in out:
            f.write(str(el) + "\n")
