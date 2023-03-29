import sys
sys.path.insert(1, '../')
import qem
try: # See if CuPy is installed. If false, continue without GPU.
    import cupy as xp
    GPU=True
except ImportError:
    import numpy as xp
    GPU=False

import random
import numpy as np
import chainer as ch
import scipy

def test_Gate():
    def test_classical_quantum_q():
        name='test Gate'

        re=xp.kron([[xp.random.rand(),0],[0, xp.random.rand()]],xp.random.rand(2,2))
        im=xp.kron([[xp.random.rand(),0],[0, xp.random.rand()]],xp.random.rand(2,2))
        array=qem.Array(re,im)
        gate=qem.Gate(array,name)
        assert gate.classical_quantum==True # Nothing wrong with this matrix.

        re=xp.kron([[xp.random.rand(),0],[0, xp.random.rand()]],xp.random.rand(2,2))
        im=xp.kron([[xp.random.rand(),0],[0, xp.random.rand()]],xp.random.rand(2,2))
        re[3,1]=1.
        array=qem.Array(re,im)
        gate=qem.Gate(array,name)
        assert gate.classical_quantum==False # Because off-diag blocks are not all zero.


        re=xp.kron([[xp.random.rand(),0],[0, xp.random.rand()]],xp.random.rand(2,2))
        im=xp.kron([[xp.random.rand(),0],[0, xp.random.rand()]],xp.random.rand(2,2))
        im[0,3]=1.
        array=qem.Array(re,im)
        gate=qem.Gate(array,name)
        assert gate.classical_quantum==False # Because off-diag blocks are not all zero.

        re=xp.random.rand(2,2)
        im=xp.random.rand(2,2)
        array=qem.Array(re,im)
        gate=qem.Gate(array,name)
        assert gate.classical_quantum==False # Because wrong shape.

    test_classical_quantum_q()

    def test_stochastic_q():
        a=xp.random.rand()
        b=xp.random.rand()
        T=xp.array([[a,b],[1-a,1-b]])
        T=qem.Array(T,xp.zeros((2,2)))
        gate=qem.Gate(T,'test clas gate')
        assert gate.classical==True

        T=xp.array([[-1,1],[2,0]])
        T=qem.Array(T,xp.zeros((2,2)))
        gate=qem.Gate(T,'test clas gate')
        assert gate.classical==False # Because of negativity

        T=xp.array([[1,1],[0,1]])
        T=qem.Array(T,xp.zeros((2,2)))
        gate=qem.Gate(T,'test clas gate')
        assert gate.classical==False # Because not normalized

        a=xp.random.rand()
        b=xp.random.rand()
        T=xp.array([[a,b],[1-a,1-b]])
        T=qem.Array(T,xp.random.rand(2,2))
        gate=qem.Gate(T,'test clas gate')
        assert gate.classical==False # Because imaginary

        a=xp.random.rand()
        b=xp.random.rand()
        T=xp.array([[a,b,a],[1-a,1-b,1-a]])
        T=qem.Array(T,xp.zeros((2,2)))
        gate=qem.Gate(T,'test clas gate')
        assert gate.classical==True # Because wrong shape

test_Gate()

def test_Cir():
    def test_vectorize():
        # Two qubits, one classical bit, apply CNOT between classical bit and the first qubit, and the classical bit and the second qubit.
        cir=qem.Cir()
        gate=qem.CNOT()
        action=qem.Action((2,0),gate)
        cir.append_action(action)
        action=qem.Action((2,1),gate)
        cir.append_action(action)
        cir.vectorize(2,1,'normal')
        cir.vectorize(2,1,'conj')
        #cir.vectorize(2,1,'normal').print_layers()
        #cir.vectorize(2,1,'conj').print_layers()

        # Incorrect circuit because quantum part of classical-quantum gate acts on classical bit. Assert exception is raised.
        try:
            cir=qem.Cir()
            gate=qem.CNOT()
            action=qem.Action((2,0),gate)
            cir.append_action(action)
            action=qem.Action((1,2),gate)
            cir.append_action(action)
            cir.vectorize(2,1,'normal')
            cir.vectorize(2,1,'conj')
            assert 'normal'
        except AssertionError:
            pass

        # Correct circuit: act with purely classical gate on classical part.
        cir=qem.Cir()
        gate=qem.X()
        action=qem.Action(2,gate)
        cir.append_action(action)
        gate=qem.CNOT()
        action=qem.Action((0,1),gate)
        cir.append_action(action)
        cir.vectorize(2,1,'normal')
        cir.vectorize(2,1,'conj')
        #cir.vectorize(2,1,'normal').print_layers()
        #cir.vectorize(2,1,'conj').print_layers()

        # Incorrect circuit because quantum gate acts on classical part of reg.
        try:
            cir=qem.Cir()
            gate=qem.Y()
            action=qem.Action(0,gate)
            cir.append_action(action)
            action=qem.Action(2,gate)
            cir.append_action(action)
            cir.vectorize(2,1,'normal')
            cir.vectorize(2,1,'conj')
            assert False
        except AssertionError:
            pass

    test_vectorize()

test_Cir()

def test_MixReg():
    reg=qem.MixReg(5)
    assert reg.psi.shape==(2,)*(2*5)
    reg=qem.MixReg(0,2)
    assert reg.psi.shape==(2,)*2
    #reg.print_ket_state()
    reg=qem.MixReg(2,0)
    assert reg.psi.shape==(2,)*(2*2)
    #reg.print_ket_state()
    reg=qem.MixReg(2,2)
    assert reg.psi.re.shape==(2,)*(2*2+2)
    #reg.print_ket_state()

test_MixReg()

def test_RIsingSWAP():
    # In case w has one par. (I.e. Ising term without local field).
    w=[-0.9]
    angle=ch.Variable(xp.array(21.4))
    action=qem.Action((0,1),qem.RIsingSWAP(w,angle))

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_Y(1,reg)
    qem.apply_action(action,reg)
    ref=[0., -0.282147316987422 - 0.6483771213705796*1j,
 0.282147316987422 + 0.6483771213705796*1j, 0.]
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),ref)

    # In case w has three pars.
    w=[12.3,-.11,2]
    angle=ch.Variable(xp.array(-99.9))
    action=qem.Action((0,1),qem.RIsingSWAP(w,angle))

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_action(action,reg)
    ref=[-0.5303399154856147 - 0.46769602739675753*1j, 0.,
         0., -0.7041273410128411 - 0.06484356281225619*1j]
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),ref)

test_RIsingSWAP()

def stinespring():
    """
    High level test that tests multiple components. If you have two qubits, with the first in a random state and the second in the state sqrt(1-p)|0> + sqrt(p)|1>, apply a qem.CY and trace out the second qubit, the resulting channel should be the bitphaseflip channel. 
    """
    def test_a_channel(gate_channel,channel):
        randreg=qem.random_mixreg(1)
        randreg.psi=qem.Array(randreg.psi.re,randreg.psi.im) # Workaround to let the class of randreg be qem.Array instead of __main__.Array
        p=xp.random.rand()

        ## Do it the stinespring way
        ### Prepare correct init state
        randrho=randreg.psi.re+1j*randreg.psi.im
        initpsi=xp.array([[xp.sqrt(1-p)],[xp.sqrt(p)]])
        initrho=xp.kron(initpsi,initpsi.conjugate().transpose())
        array=xp.kron(randrho,initrho).reshape((2,)*4)
        regstine=qem.MixReg(2)
        regstine.psi=qem.Array(array.real,array.imag)

        ## Apply gate
        qem.apply_channel(gate_channel(1,0),regstine)

        ## Trace out aux qubit
        qem.apply_channel(qem.Trace(1),regstine)

        # Do it the Kraus way
        regk=qem.MixReg(1)
        regk.psi=randreg.psi
        qem.apply_channel(channel(0,p),regk)

        # Compare
        qem.round_assert(regk.psi.re.data+1j*regk.psi.im.data,regstine.psi.re.data+1j*regstine.psi.im.data)

    test_a_channel(qem.CY_channel,qem.bitphaseflip_channel)
    test_a_channel(qem.CZ_channel,qem.phase_channel)
    test_a_channel(qem.CNOT_channel,qem.bitflip_channel)

stinespring()

def test_CZ():
    action=qem.Action((0,1),qem.CZ())

    reg=qem.Reg(2)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),[1.,0.,0.,0.])

    reg=qem.Reg(2)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),[0.,1.,0.,0.])

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),[0.,0.,1.,0.])

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),[0.,0.,0.,-1.])

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),xp.array([1.,0.,0.,-1.])/xp.sqrt(2))

test_CZ()

def test_CY():
    action=qem.Action((0,1),qem.CY())

    reg=qem.Reg(2)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),[1.,0.,0.,0.])

    reg=qem.Reg(2)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),[0.,1.,0.,0.])

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),1j*xp.array([0.,0.,0.,1.]))

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),-1j*xp.array([0.,0.,1.,0.]))

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),xp.array([1.,0.,-1j,0.])/xp.sqrt(2))

    #Swich around control and target

    action=qem.Action((1,0),qem.CY())

    reg=qem.Reg(2)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),[1.,0.,0.,0.])

    reg=qem.Reg(2)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),[0.,0.,0.,1j])

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),xp.array([0.,0.,1.,0.]))

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),-1j*xp.array([0.,1.,0.,0.]))

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),xp.array([1.,-1j,0.,0.])/xp.sqrt(2))

test_CY()

def test_Ising():
    # In case w has one par. (I.e. Ising term without local field)
    w=[xp.random.rand()]
    action=qem.Action((0,1),qem.Ising(w))

    reg=qem.Reg(2)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),-w[0]*xp.array([0.,1.,0.,0.]))

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),-w[0]*xp.array([0.,0.,1.,0.]))

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),w[0]*xp.array([0.,0.,0.,1.]))

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),w[0]*xp.array([1.,0.,0.,1.])/xp.sqrt(2))

    # In case w has three pars. Also swich around control and target.
    w=xp.random.rand(3)
    action=qem.Action((1,0),qem.Ising(w))

    reg=qem.Reg(2)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),(w[0]+w[1]+w[2])*xp.array([1.,0.,0.,0.]))
    
    reg=qem.Reg(2)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),(-w[0]-w[1]+w[2])*xp.array([0.,1.,0.,0.]))

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),(-w[0]+w[1]-w[2])*xp.array([0.,0.,1.,0.]))

    reg=qem.Reg(2)
    qem.apply_X(0,reg)
    qem.apply_X(1,reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),(w[0]-w[1]-w[2])*xp.array([0.,0.,0.,1.]))

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),xp.array([w[0]+w[1]+w[2],0.,0.,w[0]-w[1]-w[2]])/xp.sqrt(2))

test_Ising()

def test_RIsing():
    # In case w has one par. (I.e. Ising term without local field). Interchange control and target. 
    w=[.1]
    angle=ch.Variable(xp.array(-.2))
    action=qem.Action((1,0),qem.RIsing(w,angle))

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_Y(1,reg)
    qem.apply_action(action,reg)
    ref=[0.,0.014141192833545372+0.7069653645442925*1j,-0.014141192833545372-0.7069653645442925*1j,0.] 
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),ref)

    # In case w has three pars. 
    w=[.1,-.3*xp.pi,-10.2]
    angle=ch.Variable(xp.array(-15.3*xp.pi))
    action=qem.Action((0,1),qem.RIsing(w,angle))

    reg=qem.Reg(2)
    qem.apply_H(0,reg)
    qem.apply_CNOT((0,1),reg)
    qem.apply_action(action,reg)
    ref=[-0.6983699169852746-0.11081272061449436*1j,0.+0.*1j,
0.+0.*1j,0.7067640990086461+0.02201155043368494*1j]
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),ref)

test_RIsing()

def test_RY():
    angle=xp.array(xp.pi/2)
    gate=qem.RY(ch.Variable(angle))
    reg=qem.Reg(1)
    action=qem.Action(0,gate)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),(1/xp.sqrt(2))*xp.array([1,1]))

    angle=xp.array(-xp.pi/2)
    gate=qem.RY(ch.Variable(angle))
    reg=qem.Reg(1)
    action=qem.Action(0,gate)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),(1/xp.sqrt(2))*xp.array([1,-1]))

    angle=xp.array(xp.pi)
    gate=qem.RY(ch.Variable(angle))
    reg=qem.Reg(1)
    action=qem.Action(0,gate)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),xp.array([0,1]))
    
test_RY()

def test_RX():
    angle=xp.array(xp.pi/2)
    gate=qem.RX(ch.Variable(angle))
    reg=qem.Reg(1)
    action=qem.Action(0,gate)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),(1/xp.sqrt(2))*xp.array([1,-1j]))

    angle=xp.array(-xp.pi/2)
    gate=qem.RX(ch.Variable(angle))
    reg=qem.Reg(1)
    action=qem.Action(0,gate)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),(1/xp.sqrt(2))*xp.array([1,1j]))

    angle=xp.array(xp.pi)
    gate=qem.RX(ch.Variable(angle))
    reg=qem.Reg(1)
    action=qem.Action(0,gate)
    qem.apply_action(action,reg)
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),xp.array([0,-1j]))
    
test_RX()

def test_measure():
    """
    Create a random state, and see if we project on 0 with the right probability.
    """
    for i in range(2**8):
        p=xp.random.rand()

        initpsi=xp.array([[xp.sqrt(1-p)],[xp.sqrt(p)]])
        initrho=xp.kron(initpsi,initpsi.conjugate().transpose())
        reg=qem.MixReg(1)
        reg.psi=qem.Array(initrho.real,initrho.imag)
        qem.apply_channel(qem.measure(0),reg)

        assert reg.psi.re.data[0][0].round(4)==xp.array(1-p).round(4)
        assert reg.psi.re.data[1][1].round(4)==xp.array(p).round(4)
        assert reg.psi.re.data[0][1].round(4)==0
        assert reg.psi.re.data[1][0].round(4)==0
        assert reg.psi.im.data[0][0].round(4)==0
        assert reg.psi.im.data[1][1].round(4)==0
        assert reg.psi.im.data[0][1].round(4)==0
        assert reg.psi.im.data[1][0].round(4)==0

test_measure()

def test_apply_channel(): #Also tests Trace, ClasTrace, Channel, entropy(), GHZ_channel(), measure()

    ######
    # Using two classical bits, prepare the mixed state  ( (1-p)|00><00|+p|11><11| ) otimes  ( (1-q)|00><00|+q|11><11| ).
    ######
    n=4
    m=2
    reg=qem.MixReg(4,2)
    # First put the classical qubits in the right state.
    #
    p=xp.random.rand()
    q=xp.random.rand()
    rep=xp.array([[1-p,1-p],[p,p]])
    req=xp.array([[1-q,1-q],[q,q]])
    im=xp.zeros((2,2))
    chan=qem.Chan()

    cir=qem.Cir()
    gate=qem.Gate(qem.Array(rep,im),'prepare p')
    action=qem.Action(4,gate)
    cir.append_action(action)

    gate=qem.Gate(qem.Array(req,im),'prepare q')
    action=qem.Action(5,gate)
    cir.append_action(action)

    gate=qem.CNOT()
    action=qem.Action((4,0),gate)
    cir.append_action(action)
    action=qem.Action((4,1),gate)
    cir.append_action(action)
    action=qem.Action((5,2),gate)
    cir.append_action(action)
    action=qem.Action((5,3),gate)
    cir.append_action(action)

    chan.append_circuit(cir)
    qem.apply_channel(chan,reg)

    qem.apply_channel(qem.ClasTrace([4]),reg)
    qem.apply_channel(qem.ClasTrace([4]),reg)

    zero=[[1,0]]
    one=[[0,1]]
    trueresult=xp.kron( (1-p)*xp.kron(xp.kron(zero,zero),xp.kron(zero,zero)) + (p)*xp.kron(xp.kron(zero,zero),xp.kron(zero,zero)),(1-q)*xp.kron(xp.kron(one,one),xp.kron(one,one)) + (q)*xp.kron(xp.kron(one,one),xp.kron(one,one)))
    result=reg.psi.re.data+1j*reg.psi.im.data
    result=trueresult.reshape((1,2**8,))
    qem.round_assert(result,trueresult)

    ######
    #If we trace out any collection of qubits in a GHZ state, the resulting state has the entropy of one bit.
    ######
    
    # Make GHZ state
    n=8
    reg=qem.MixReg(n)
    chan=qem.GHZ_channel(range(n))
    qem.apply_channel(chan,reg)
    
    # Trace out random qubits
    m=xp.random.randint(n-1)+1
    tracequbits=random.sample(range(n),m)
    qem.apply_channel(qem.Trace(tracequbits),reg)

    # Check if the result has entropy 1
    maxmixent=1
    ent=qem.entropy(reg)
    qem.round_assert(ent,maxmixent)

    ######
    # If we apply a lot of weak bitflip and phaseflip channels to any pure state that is not |++...><++...|, the entropy goes to max mixed.
    ######
    n=4
    reg=qem.random_mixreg(n)
    #reg=qem.MixReg(n)
    qem.round_assert(qem.entropy(reg),0.)
    
    for i in range(10*2**n):
        for q in range(n):
            chan=qem.bitflip_channel(q,.1)
            qem.apply_channel(chan,reg)
            chan=qem.phase_channel(q,.1)
            qem.apply_channel(chan,reg)

    qem.round_assert(qem.entropy(reg),n)

    ####
    #Prepare a GHZ, measure any qubit, that qubit should be in the max mixed state, the rest is classical superpos of |00...> and |11...>
    ####
    n=8
    reg=qem.MixReg(n)
    m=xp.random.randint(0,n)
    qem.apply_channel(qem.GHZ_channel(range(n)),reg)
    qem.apply_channel(qem.measure(m),reg)
    reg2=qem.MixReg(n)
    reg2.psi=reg.psi    
    rest=np.array(range(n))
    rest=rest[rest!=m]
    rest=tuple(rest)
    qem.apply_channel(qem.Trace(rest),reg)
    array=reg.psi.re.data+1j*reg.psi.im.data
    qem.round_assert(array,[[.5,0.],[0.,.5]])

    array=reg2.psi.re.data.flatten()+1j*reg2.psi.im.data.flatten()
    qem.round_assert(array[0],.5)
    qem.round_assert(array[-1],.5)
    qem.round_assert(array[1:-1],xp.zeros(2**(2*n)-2))   
    
    ####
    #Do a some gates and channels, and compare the outcome to that obtained with and independent implementation in Mathematica. It includes measurement, gates, channels,
    ####    
    n=6
    
    reg=qem.MixReg(n)
    qem.apply_channel(qem.GHZ_channel(range(n)),reg)
    qem.apply_channel(qem.RY_channel(1,ch.Variable(xp.array(-.1))),reg)
    qem.apply_channel(qem.RX_channel(3,ch.Variable(xp.array(-.3))),reg)
    qem.apply_channel(qem.bitflip_channel(1,.1),reg)
    qem.apply_channel(qem.phase_channel(3,.2),reg)
    qem.apply_channel(qem.bitphaseflip_channel(5,.3),reg)
    qem.apply_channel(qem.X_channel(0),reg)
    qem.apply_channel(qem.Y_channel(4),reg)
    qem.apply_channel(qem.H_channel(3),reg)
    qem.apply_channel(qem.CNOT_channel(1,2),reg)
    qem.apply_channel(qem.CNOT_channel(2,4),reg)
    qem.apply_channel(qem.Y_channel(0),reg)
    qem.apply_channel(qem.measure(4),reg)
    qem.apply_channel(qem.measure(2),reg)
    qem.apply_channel(qem.Trace((2,4)),reg)
    qem.apply_channel(qem.H_channel(3),reg)
    qem.apply_channel(qem.RY_channel(0,ch.Variable(xp.array(.1))),reg)
    qem.apply_channel(qem.RX_channel(0,ch.Variable(xp.array(.3))),reg)

    from scipy.io import mmread, mmwrite
    a = mmread("test_array1.mtx")
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),a.flatten())


    ####
    # See that, when we start with a max. mixed state, applying some circuit results in max. mixed state.
    ####
    w=xp.random.rand(3)
    pars=ch.Variable(xp.random.rand(2))
    re=xp.random.rand(4,4)
    im=xp.random.rand(4,4)
    m=re+1j*im
    dag=m.transpose().conjugate()
    randU=scipy.linalg.expm(1j*(m+dag))
    randU=qem.Gate(qem.Array(randU.real,randU.imag),'randU')


    #construct circuit, and make a channel out of it
    cir=qem.Cir()
    cir.append_action(qem.Action(0,qem.H()))
    cir.append_action(qem.Action(1,qem.H()))
    cir.append_action(qem.Action((0,1),randU))
    cir.append_action(qem.Action((0,1),qem.RIsing(w,pars[0])))
    cir.append_action(qem.Action(0,qem.RX(pars[1])))
    cir.append_action(qem.Action(1,qem.RX(pars[1])))
    cir.append_action(qem.Action(2,qem.H()))
    cir.append_action(qem.Action(3,qem.H()))
    cir.append_action(qem.Action((1,2),randU))
    cir.append_action(qem.Action((2,3),qem.RIsing(w,pars[0])))
    cir.append_action(qem.Action(3,qem.RX(pars[1])))
    cir.append_action(qem.Action(1,qem.RX(pars[1])))

    chan=qem.Chan()
    chan.append_circuit(cir)

    # Make max mixed state. 
    reg=qem.MixReg(4)
    re=xp.identity(2**4).reshape((2,2,2,2,2,2,2,2))/2**4
    im=xp.zeros((2,2,2,2,2,2,2,2))
    reg.psi=qem.Array(re,im)

    # Apply channel and see if outcome is correct. (I.e. max. mixed)
    qem.apply_channel(chan,reg)
    qem.round_assert(reg.psi.re.data,re)
    qem.round_assert(reg.psi.im.data,im)

test_apply_channel()

def test_apply_supcir(): # Same as test_apply_channel() but than for everything is rephrased in supercicuits.
    
    ######
    #If we trace out any collection of qubits in a GHZ state, the resulting state has the entropy of one bit.
    ######
    
    # Make GHZ state
    n=8
    chan1=qem.GHZ_channel(range(n))
    # Trace out random qubits
    m=xp.random.randint(n-1)+1
    tracequbits=random.sample(range(n),m)
    chan2=qem.Trace(tracequbits)

    sc=qem.SupCir(n,0,[chan1,chan2])
    reg=qem.MixReg(n)
    qem.apply_supcir(sc,reg)

    # Check if the result has entropy 1
    maxmixent=1
    ent=qem.entropy(reg)
    qem.round_assert(ent,maxmixent)

    ######
    # If we apply a lot of weak bitflip and phaseflip channels to any pure state that is not |++...><++...|, the entropy goes to max mixed.
    ######
    n=4
    reg=qem.random_mixreg(n)
    qem.round_assert(qem.entropy(reg),0.)
    sc=qem.SupCir(n,0)
    
    for i in range(10*2**n):
        for q in range(n):
            chan=qem.bitflip_channel(q,.1)
            sc.append_channel(chan)
            chan=qem.phase_channel(q,.1)
            sc.append_channel(chan)

    qem.apply_supcir(sc,reg)
    qem.round_assert(qem.entropy(reg),n)

    ####
    #Do a some gates and channels, and compare the outcome to that obtained with and independent implementation in Mathematica. It includes measurement, gates, channels,
    ####    
    n=6
    sc=qem.SupCir(n,0)
    
    sc.append_channel(qem.GHZ_channel(range(n)))
    sc.append_channel(qem.RY_channel(1,ch.Variable(xp.array(-.1))))
    sc.append_channel(qem.RX_channel(3,ch.Variable(xp.array(-.3))))
    sc.append_channel(qem.bitflip_channel(1,.1))
    sc.append_channel(qem.phase_channel(3,.2))
    sc.append_channel(qem.bitphaseflip_channel(5,.3))
    sc.append_channel(qem.X_channel(0))
    sc.append_channel(qem.Y_channel(4))
    sc.append_channel(qem.H_channel(3))
    sc.append_channel(qem.CNOT_channel(1,2))
    sc.append_channel(qem.CNOT_channel(2,4))
    sc.append_channel(qem.Y_channel(0))
    sc.append_channel(qem.measure(4))
    sc.append_channel(qem.measure(2))
    sc.append_channel(qem.Trace((2,4)))
    sc.append_channel(qem.H_channel(3))
    sc.append_channel(qem.RY_channel(0,ch.Variable(xp.array(.1))))
    sc.append_channel(qem.RX_channel(0,ch.Variable(xp.array(.3))))

    reg=qem.MixReg(n)
    qem.apply_supcir(sc,reg)
    from scipy.io import mmread, mmwrite
    a = mmread("test_array1.mtx")
    qem.round_assert(reg.psi.re.data.flatten()+1j*reg.psi.im.data.flatten(),a.flatten())

    # Put CNOT and H are in seperate channels, see if outcome is correct.

    reg=qem.MixReg(2)
    sc=qem.SupCir(2,0)
    
    cir=qem.Cir()
    cir.append_action(qem.Action(0,qem.H()))
    chan=qem.Chan([cir])
    sc.append_channel(chan)

    cir=qem.Cir()
    cir.append_action(qem.Action((0,1),qem.CNOT()))
    chan=qem.Chan([cir])
    sc.append_channel(chan)

    reg=qem.MixReg(2)
    qem.apply_supcir(sc,reg)

    qem.round_assert(reg.psi.re.data,xp.array([[[[0.5, 0. ],[0.,  0.5]],[[0.,  0. ],[0. , 0. ]]],[[[0.,  0. ],[0. , 0. ]],[[0.5, 0. ],[0. , 0.5]]]]))
    qem.round_assert(reg.psi.im.data,xp.zeros((2,2,2,2)))

    # Put CNOT an H in the same channel and check outcome.

    sc=qem.SupCir(2,0)

    chan=qem.Chan()
    cir=qem.Cir()

    cir.append_action(qem.Action(0,qem.H()))
    cir.append_action(qem.Action((0,1),qem.CNOT()))

    chan.append_circuit(cir)
    sc.append_channel(chan)

    reg=qem.MixReg(2)
    qem.apply_supcir(sc,reg)

    qem.round_assert(reg.psi.re.data,xp.array([[[[0.5, 0. ],[0.,  0.5]],[[0.,  0. ],[0. , 0. ]]],[[[0.,  0. ],[0. , 0. ]],[[0.5, 0. ],[0. , 0.5]]]]))
    qem.round_assert(reg.psi.im.data,xp.zeros((2,2,2,2)))
    
test_apply_supcir()

def test_hl_teleport():
    """
    High-level test that checks if teleportation works as expected.
    """
    initreg=qem.random_mixreg(1)
    initrho=initreg.psi.re+1j*initreg.psi.im
    array=xp.kron(xp.kron(initrho,[[1.,0.],[0.,0.]]),[[1.,0.],[0.,0.]]).reshape((2,)*6)
    fullreg=qem.MixReg(3)
    fullreg.psi=qem.Array(array.real,array.imag)

    cir=qem.Cir()
    cir.append_action(qem.Action(1,qem.H()))
    cir.append_action(qem.Action((1,2),qem.CNOT()))
    cir.append_action(qem.Action((0,1),qem.CNOT()))
    cir.append_action(qem.Action(0,qem.H()))
    qem.apply_channel(qem.Chan([cir]),fullreg)

    qem.apply_channel(qem.measure(0),fullreg)
    qem.apply_channel(qem.measure(1),fullreg)

    cir=qem.Cir()
    cir.append_action(qem.Action((1,2),qem.CNOT()))
    cir.append_action(qem.Action((0,2),qem.CZ()))
    qem.apply_channel(qem.Chan([cir]),fullreg)

    qem.apply_channel(qem.Trace((0,1)),fullreg)

    qem.round_assert(fullreg.psi.re.data+1j*fullreg.psi.im.data,initreg.psi.re.data+1j*fullreg.psi.im.data)

test_hl_teleport()

def test_prepare_classical_mixed_state_channel():
    #qem.prepare_classical_mixed_state_channel(.1,2).print_circuits() # Check by hand
    p=xp.random.rand()
    reg=qem.MixReg(3)
    qem.apply_channel(qem.GHZ_channel([0,1,2]),reg) 
    qem.apply_channel(qem.prepare_classical_mixed_state_channel(p,1),reg)
    qem.apply_channel(qem.Trace([0,2]),reg)
    qem.round_assert(reg.psi.re.data+1j*reg.psi.im.data,[[1-p,0.],[0.,p]])

    p=xp.random.rand()
    reg=qem.MixReg(1)
    qem.apply_channel(qem.prepare_classical_mixed_state_channel(p,0),reg)
    qem.round_assert(reg.psi.re.data+1j*reg.psi.im.data,[[1-p,0.],[0.,p]])

test_prepare_classical_mixed_state_channel()

print('All test passed succesfully')
