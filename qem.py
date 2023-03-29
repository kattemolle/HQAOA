try: # See if CuPy is installed. If false, continue without GPU.
    import cupy as xp
    print('CuPy installation found, continuing using GPU acceleration.')
    GPU=True
except ImportError:
    print('No CuPy installation found, continuing without GPU acceleration.')
    import numpy as xp
    GPU=False

import numpy #For the cases where GPU==True and we still want to use NumPy.
import chainer as ch
import scipy.sparse.linalg
import scipy.linalg
from copy import deepcopy
import os
from functools import reduce

#try:
#    os.sched_setaffinity(0,range(1000)) # Reset the CPU affinity; allow Python to use all CPUs that are available. 
#    print('having the CPUs', os.sched_getaffinity(0), 'available.')
#except AttributeError:
#    pass

# Only use when timing: may slow down computation if used improperly 
if GPU==True:
    sync=xp.cuda.stream.get_current_stream().synchronize

# Classes

class Array:
    """
    Chainer does not natively support complex numbers. We need Chainer for automatic differentiation. This class defines a new type of array that stores the real part and imaginary part separately, as two real arrays. If an array is purely real (imaginary) the imaginary part (real part) should be stored as an array of the same shape containing zeros. The real and imaginary part should be either NumPy/CuPy arrays or `chainer.Variable`s. 

    Attributes
    ----------
    re : chainer.Variable, numpy.ndarray or cupy.ndarray

    im : chainer.Variable, numpy.ndarray or cupy.ndarray

    shape : tuple

    """
    def __init__(self,re,im):
        assert re.shape==im.shape, 'Array.re and array.im must have same shape.'
        self.re=re
        self.im=im
        self.shape=self.re.shape
            
    def __str__(self):
        return str(self.re)+'\n + i* \n'+str(self.im)

    def __add__(self, other):
        return Array(self.re+other.re,self.im+other.im)

    def __sub__(self,other):
        return Array(self.re-other.re,self.im-other.im)

    def __rmul__(self,other): # Element-wise product of two objects of type Array of a float and an Array.
        if type(other)==xp.float64 or type(other)==float:
            return Array(other*self.re, other*self.im)
            
        elif type(other)==Array:
            return Array(self.re*other.re-self.im*other.im, self.re*other.im+self.im*other.re)

        else:
            TypeError('Wrong type of multiplication of an Array by an Array or float.')
    
    def dagger(self): # Return complex conjugated array. Dagger is a misnomer, but the function is kept for backward compatibility. 
        return Array(self.re,-self.im)


    def conjugate(self): # Return complex conjugated array
        return Array(self.re,-self.im)

    def reshape(self,shape): # Not this always creates a copy, so use with care.
        return Array(self.re.reshape(shape),self.im.reshape(shape))
            
    def do_dagger(self): # Apply dagger, return None
        self.im=-self.im


class Reg:
    """
    The quantum register class. Holds the current wave function of the register as a qem.Array array of shape (2,)*n. Upon creation, the wave function is initialized as the wave function of |00...0>.

    Parameters
    ----------
    n : int
      The number of qubits of the register.

    Attributes
    ----------
    n : int
      The number of qubits of the register.

    psi : qem.Array 
      The wave function of the register. A quantum state $\sum_{ab} \psi_{ab...} |ab...>$ is stored in such a way that psi[a,b,...] holds value $\psi_{ab...}$. Is initialized to the wavefunction of |00...0>

    Methods
    -------
    reset()
        Reset the state of the reg to |00...0>.

    print_ket_state()
        Print the state of the reg to stdout in ket notation, in e.g. the form (0.707107+0j)|00> + (0.707107+0j)|11>.

    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> print(reg.psi)
    [[1. 0.]
     [0. 0.]]
     + i* 
    [[0. 0.]
     [0. 0.]]
    >>> reg.print_ket_state()
    psi = (1+0j)|00>
    """
    def __init__(self,n):
        self.n=n
        self.psi=xp.zeros((2,)*self.n,dtype=xp.float64)
        self.psi[(0,)*self.n]=1.0
        self.psi=Array(self.psi, xp.zeros((2,)*self.n,dtype=xp.float64))

    def reset(self):
        """
        Reset the wavefunction to that of |00..0>.
        """
        del self.psi
        self.psi=xp.zeros((2,)*self.n,dtype=xp.float64)
        self.psi[(0,)*self.n]=1.0
        self.psi=Array(self.psi, xp.zeros((2,)*self.n,dtype=xp.float64))

    def print_ket_state(self):
        get_bin=lambda x: format(x, 'b').zfill(self.n) # Def function that converts ints to the binary representation and outputs it as string.
        re=self.psi.re
        im=self.psi.im
        if type(re)==ch.Variable:
            re=re.data
        if type(im)==ch.Variable:
            im=im.data        
        psi=re+1j*im
        psi=psi.flatten()
        psi=numpy.around(psi,6)
        out='psi = '
        for i in range(len(psi)):
            if psi[i]!=0:
                out+=str(psi[i])+'|'+get_bin(i)+'>'+' + '
        out = out[:-3]
        print(out)
        
class EmptyReg:
    """
    Same as Reg but initializes with empty Array as state.
    """
    def __init__(self,n):
        self.n=n
        self.psi=Array(xp.array([0.]),xp.array([0.]))

    def reset(self):
        """
        Resets the wavefunction to that of |00..0>. No parameters.
        """
        del self.psi
        self.psi=Array(xp.array([0.]),xp.array([0.]))

class EmptyMixReg:
    """
    Same as MixReg but initializes with empty Array as state.
    """
    def __init__(self,*args):
        if len(args)==1:
            self.n=args[0]
            self.m=0
        elif len(args)==2:
            self.n,self.m=args
        else:
            raise ValueError('EmptyMixReg takes one or two arguments.')
        self.psi=Array(xp.array([0.]),xp.array([0.]))

    def reset(self):
        """
        Resets the wavefunction to that of |00..0>. No parameters.
        """
        del self.psi
        self.psi=Array(xp.array([0.]),xp.array([0.]))

class Gate:
    """
    The Gate's parameter can be either a qem.Array of shape (2**m,2**m) or (2,)*m, where m is the number of qubits the gate acts upon. In converting matrices to tensors of shape (2,)*m, we use the convention that [000...,000...] refers to the upper right entry of the matrix. In the tensor format, this correspronds to the entry [0,0,0,...,0,0,0,...].

    Parameters
    ----------
    array : qem.Array of shape (2**m,2**m) or (2,)*m with m the number of qubits the gate acts upon.
    name : string 
        Name of the gate used for printing the gate.

    Attributes
    ----------
    self.array: qem.Array
        The input qem.Array, but then qem.Array.re and qem.Array.im are reshaped to shape (2,)*m.

    self.name : string

    Methods:
    --------
    get_matrix() : return the array but reshaped to (2**m,2**m), irrespective of the shape that was used to construct the object.

    __rmul__() : defines multiplication of a gate by a float. 
    """
    def __init__(self,array,name):
        assert type(array)==Array
        self.array=array
        self.name=name
        if array.shape!=(2,2,2,2) and array.shape!=(2,2):
            totaldim=reduce((lambda x, y: x * y), array.shape)
            self.array=array.reshape((2,)*int(numpy.log2(totaldim)))
        self.classical_quantum=self.classical_quantum_q()
        self.classical=self.classical_q()

    def __rmul__(self,other):
        if type(other)==xp.float64 or type(other)==float:
            gate=Gate(other*self.array,str(other)+'*'+self.name)
            return gate
        else:
            TypeError('Wrong type in multiplication of a gate.')
            
    def get_matrix(self):
        return self.array.reshape((2**(len(self.array.shape)//2),2**(len(self.array.shape)//2)))

    def classical_quantum_q(self): # Return true if the gate can be interpreted as a classical-quantum two qubit gate.
        if self.array.shape!=(2,)*4:
            return False
        else:
            re=self.get_matrix().re
            im=self.get_matrix().im
            if type(re)==ch.Variable:
                re=re.data
            if type(im)==ch.Variable:
                im=im.data
            ar=re+1j*im
            left_down=ar[2:,:2]
            top_right=ar[:2,2:]
            zeros=[[0,0],[0,0]]
            if (xp.array_equal(left_down,zeros) and xp.array_equal(top_right,zeros)):
                return True
            else:
                return False

    def classical_q(self): # Return true if the array beloning to this gate is a 2x2 column stochastic matrix
        if self.array.shape!=(2,2): # Must be right shape
            return False
        re=self.get_matrix().re
        im=self.get_matrix().im
        if type(re)==ch.Variable:
            re=re.data
        if type(im)==ch.Variable:
            im=im.data
        if not xp.array_equal(im,xp.zeros((2,2))): # Must be real
            return False
        if xp.around(re[0,0]+re[1,0],5)!=1 or xp.around(re[0,1]+re[1,1],5)!=1: # Columns must be normalized.
            return False
        elif not  all(re[i,j]>=0 for i in [0,1] for j in [0,1]): # Entries must be positive
            return False
        else:
            return True

# Predefined gates that are children of the Gates with preset matrices and names.

class ID(Gate):
    def __init__(self):
        self.matrix=Array(xp.array([[1.,0.],[0.,1.]]),xp.array([[0.,0.],[0.,0.]]))
        Gate.__init__(self,self.matrix,'ID')

class X(Gate):
    matrix=Array(xp.array([[0.,1.],[1.,0.]]),xp.array([[0.,0.],[0.,0.]]))
    def __init__(self):
        Gate.__init__(self,self.matrix,'X')

class Y(Gate):
    matrix=Array(xp.array([[0.,0.],[0.,0.]]),xp.array([[0.,-1.],[1.,0.]]))
    def __init__(self):
        Gate.__init__(self,self.matrix,'Y')

class Z(Gate):
    matrix=Array(xp.array([[1.,0.],[0.,-1.]]),xp.array([[0.,0.],[0.,0.]]))
    def __init__(self):
        Gate.__init__(self,self.matrix,'Z')

class H(Gate):
    matrix=Array(xp.array([[1.,1.],[1.,-1.]]/numpy.sqrt(2)),xp.array([[0.,0.],[0.,0.]]))
    def __init__(self):
        Gate.__init__(self,self.matrix,'H')


class RY(Gate): #e^{- i angle Y / 2}
    """
    qem.Gate, where you rotate around the y axis by angle 'angle'.

    Parameters
    ----------
    angle : Chainer.Variable

    """
    def __init__(self,angle):
        assert type(angle)==ch.Variable
        Gate.__init__(self,self.RY_matrix(angle),'RY('+str(angle.data)+')')

    def RY_matrix(self,angle):
        r1=ch.functions.cos(angle/2)*xp.array([[1.,0.],[0.,0.]])
        r2=ch.functions.cos(angle/2)*xp.array([[0.,0.],[0.,1.]])
        r3=-ch.functions.sin(angle/2)*xp.array([[0.,1.],[0.,0.]])
        r4=ch.functions.sin(angle/2)*xp.array([[0.,0.],[1.,0.]])
        re=r1+r2+r3+r4
        im=xp.zeros((2,2),dtype=xp.float64)
        return Array(re,im)


class RX(Gate): #e^{- i angle X / 2}
    def __init__(self,angle):
        assert type(angle)==ch.Variable
        self.angle=angle
        Gate.__init__(self,self.RX_matrix(angle),'RX('+str(angle.data)+')')

    def RX_matrix(self,angle):
        r1=ch.functions.cos(angle/2)*xp.array([[1.,0.],[0.,0.]])
        r2=ch.functions.cos(angle/2)*xp.array([[0.,0.],[0.,1.]])
        re=r1+r2

        i1=-ch.functions.sin(angle/2)*xp.array([[0.,1.],[0.,0.]])
        i2=-ch.functions.sin(angle/2)*xp.array([[0.,0.],[1.,0.]])
        im=i1+i2

        return Array(re,im)


# Two-qubit gates

class CNOT(Gate):
    matrix=Array(xp.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]]),xp.zeros((4,4),dtype=xp.float64))
    def __init__(self):
        Gate.__init__(self,self.matrix,'CNOT')

class SWAP(Gate):
    matrix=Array(xp.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]]),xp.zeros((4,4),dtype=xp.float64))
    def __init__(self):
        Gate.__init__(self,self.matrix,'SWAP')

class Heisenberg:
    """
    The Heisenberg gate, with matrix (XX + YY + ZZ)/4. Gets special treatment in run_VQE.
    """
    pass
        
class Heisenberg_exp:
    """
    This is not like the other gates for reasons of speed. It is treated differently in `qem.apply_action`.

    Parameters
    ----------
    angle : chainer.Variable
    """
    def __init__(self,angle):
        self.angle=angle

class prepare_singlet(Gate):
    """
    This is a non-unitary operation that prepares the two-body ground state of (XX+YY+ZZ)/4, the singlet state, if acted on qubits in the state |00>.
    """
    array=Array( xp.array([[0,0,0,0],[1,0,0,0],[-1,0,0,0],[0,0,0,0]])/xp.sqrt(2) , xp.zeros((4,4)) )
    def __init__(self):
        Gate.__init__(self,self.array,'prepare_singlet' )
        

class CY(Gate):
    """
    Controlled Y, control on the left qubit.
    """
    def __init__(self):     
        re=xp.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        im=xp.array([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,-1.],[0.,0.,1.,0.]])
        matrix=Array(re,im)
        Gate.__init__(self,matrix,'CY')
        
class CZ(Gate):
    """
    Controlled Z, control on the left qubit.
    """
    def __init__(self):     
        re=xp.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]])
        im=xp.array([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        matrix=Array(re,im)
        Gate.__init__(self,matrix,'CZ')


class Ising(Gate):
    """
    Ising([w01]) creates a qem.Gate with matrix w01 Z_0 \otimes Z_1. Ising([w01,w0,w1]) creates a qem.Gate with matrix w01 Z_0 \otimes Z_1 + w0 Z_0 + w1 Z_1.
    """
    
    def __init__(self,w):
        if len(w)==1:
            w=[w[0],0.,0.]
            
        re=xp.array([[w[0] + w[1] + w[2], 0., 0., 0.], [0., -w[0] + w[1] - w[2], 0., 0.], [0., 
                                                                                  0., -w[0] - w[1]+ w[2], 0.], [0., 0., 0., w[0] - w[1] - w[2]]])
        im=xp.zeros((4,4),dtype=xp.float64)
        self.matrix=Array(re,im)
        Gate.__init__(self,self.matrix,'Ising('+str(w)+')')


class RIsing(Gate):
    """
    RIsing(w) creates the qem.Gate with matrix e^{- i H angle}, where H = w01 Z_0 \otimes Z_1 if w=[w01], and H=w01 Z_0 \otimes Z_1 + w0 Z_0 + w1 Z_1 if w=[w01,w0,w1]. Angle must be a Chaniner.Variable.
    """
    def __init__(self,w,angle):
        assert type(w)==list or type(w)==xp.ndarray or type(w)==tuple
        assert len(w)==1 or len(w)==3
        assert type(angle)==ch.Variable
        self.angle=angle
        Gate.__init__(self,self.RIsing_matrix(w,angle),'RIsing(angle='+str(angle.data)+', w='+str(w)+')')

    def RIsing_matrix(self,w,angle):
        if len(w)==1:
            w=[w[0],0.,0.]
            
        w01,w0,w1=w
        
        r1=ch.functions.cos(angle*(w01+w0+w1))*xp.array([1.,0.,0.,0.])
        r2=ch.functions.cos(angle*(-w01+w0-w1))*xp.array([0.,1.,0.,0.])
        r3=ch.functions.cos(angle*(w01+w0-w1))*xp.array([0.,0.,1.,0.])
        r4=ch.functions.cos(angle*(-w01+w0+w1))*xp.array([0.,0.,0.,1.])
        re=ch.functions.reshape(ch.functions.concat((r1,r2,r3,r4),axis=0),(4,4))

        i1=-ch.functions.sin(angle*(w01+w0+w1))*xp.array([1.,0.,0.,0.])
        i2=-ch.functions.sin(angle*(-w01+w0-w1))*xp.array([0.,1.,0.,0.])
        i3=ch.functions.sin(angle*(w01+w0-w1))*xp.array([0.,0.,1.,0.])
        i4=ch.functions.sin(angle*(-w01+w0+w1))*xp.array([0.,0.,0.,1.])
        im=ch.functions.reshape(ch.functions.concat((i1,i2,i3,i4),axis=0),(4,4))

        return Array(re,im)
    
class RIsingSWAP(Gate):
    """
    RIsing(w,angle) creates the qem.Gate with matrix SWAP*e^{- i H angle} (So first rotation by H and then the SWAP gate), where H = w01 Z_0 \otimes Z_1 if w=[w01], and H=w01 Z_0 \otimes Z_1 + w0 Z_0 + w1 Z_1 if w=[w01,w0,w1]. Angle must be a Chaniner.Variable.
    """
    def __init__(self,w,angle):
        assert type(w)==list or type(w)==xp.ndarray or type(w)==tuple
        assert len(w)==1 or len(w)==3
        assert type(angle)==ch.Variable
        self.angle=angle
        Gate.__init__(self,self.RIsingSWAP_matrix(w,angle),'RIsingSWAP(angle='+str(angle.data)+', w='+str(w)+')')

    def RIsingSWAP_matrix(self,w,angle):
        if len(w)==1:
            w=[w[0],0.,0.]

        w01,w0,w1=w

        r1=ch.functions.cos(angle*(w01+w0+w1))*xp.array([1.,0.,0.,0.])
        r2=ch.functions.cos(angle*(w01+w0-w1))*xp.array([0.,0.,1.,0.])
        r3=ch.functions.cos(angle*(-w01+w0-w1))*xp.array([0.,1.,0.,0.]) # The effect of the SWAP is to exchange rows 2 and 3 as compared to RIsing.
        r4=ch.functions.cos(angle*(-w01+w0+w1))*xp.array([0.,0.,0.,1.])
        re=ch.functions.reshape(ch.functions.concat((r1,r2,r3,r4),axis=0),(4,4))

        i1=-ch.functions.sin(angle*(w01+w0+w1))*xp.array([1.,0.,0.,0.])
        i2=ch.functions.sin(angle*(w01+w0-w1))*xp.array([0.,0.,1.,0.])
        i3=-ch.functions.sin(angle*(-w01+w0-w1))*xp.array([0.,1.,0.,0.])
        i4=ch.functions.sin(angle*(-w01+w0+w1))*xp.array([0.,0.,0.,1.])
        im=ch.functions.reshape(ch.functions.concat((i1,i2,i3,i4),axis=0),(4,4))

        return Array(re,im)



class Action:
    """ 
    An Action is a combination of a `qem.Gate' and the qubit ints this gate should act upon. Creation of an Action object does not apply the gate to the qubit. To do this, use `qem.apply_action()`. 

    Parameters
    ----------
    qubits : tuple of ints or int.
        The qubits the gate should act upon. This is a tuple of one int for single-qubit gates, and a tuple of two ints for tow-qubit gates. If it is an int, it is converted internally to a tuple.

    gate : qem.Gate
    """
    def __init__(self, qubits,gate):
        if type(qubits) is int:
            self.qubits=(qubits,)
        else:
            self.qubits=qubits
        self.gate=gate
        
class Layer:
    """
    A layer is a list of actions (`qem.action`). We imagine all the actions in a layer would be excequted simultaniously on a quantum computer. This behaviour is, however, not enforced, and a layer can have multiple actions acting on the same qubit, but in that case care should be taken to ensure that gates are applied in the right order.
    
    Parameters
    ----------
    *args : Arguments
        - None 
          self.actions is initialized as an empty list. 

        - list : list
          List of actions (`qem.action`).

    Methods
    -------
        append_action(action)
            Append the action to the layer.

        print_actions()
            Print all gates in layer in human readable format.

    Examples
    --------
    
    """
    def __init__(self,*args):
        if len(args)==0:
            self.actions=[]
        if len(args)==1:
            self.actions=args[0]

    def append_action(self, action):
        self.actions.append(action)

    def print_actions(self):
        print('        Printing layer, with actions:')
        print('        ==============')
        first=True
        for action in self.actions:
            if first==False:
                print('        --------------')
            first=False
            print('        '+action.gate.name, list(action.qubits))
        print('        ==============')

class Cir:
    """
    A circuit is has list of layers (qem.layer). Every layer is a list of actions (qem.action). Every action has a qubits list and a gate (qem.gate). 
    The circuit is initialized with as having one layer with no actions. 

    Attributes
    ----------
    layers : list
        List containing layers (`qem.layer`). 
    
    Methods
    -------
    append_layer(*args)
        append a new layer to self.layers. If no args are given, the new Layer is empty. If one argument is given, it should be a list of actions. Then this list of actions is the new layer. 
    
    append_action(action) 
        Append an Action to the last layer.

    print_layers()
        Print all layers in human-readable format. 

    __rmul__()
        Defines multiplication of a circuit by a float. Multiplies all gates in a circuit by a float.

    complex_conjugate()
        Return self.circuit, but where all gates of the circuit are complex conjugate. Note that the order of the gates is not changed.  

    translate(q)
        Return the circuit, but translated by q qubits. 
 
    Examples
    --------
    
    >>> import qem
    >>> cir=qem.Cir() # Create circuit with one empty layer
    >>> cir.append_action(qem.Action((1),qem.H())) # Append a Hadamard action to the layer. 
    >>> cir.append_layer() # Append a new empty layer to the circuit.
    >>> cir.append_action(qem.Action((0,1),qem.CNOT())) # Append a CNOT action to the new emtpy layer.

    The circuit now contains the layers
    >>> print(cir.layers)
    [<qem.Layer object at 0x101a163950>, <qem.Layer object at 0x1018f9abd0>]

    Of which the 0th layer contains the actions
    >>>print(cir.layers[0].actions)
    [<qem.Action object at 0x101a1638d0>]

    Of which the 0th action has the gate with name
    >>> print(cir.layers[0].actions[0].gate.name)
    H

    Multiply the circuit by a number. (To be used in the construction of Kraus operators.)
    >>>import qem
    >>>cir=qem.Cir()
    >>>cir.append_action(qem.Action(0,qem.X()))
    >>>cir.append_action(qem.Action(1,qem.X()))
    >>>cir2=.1*cir
    >>>cir2.print_layers()
    Printing circuit, with layers:
    ==============
        Printing layer, with actions:
        ==============
        0.1*X [0]
        --------------
        0.1*X [1]
        ==============
    ==============

    >>>print(cir2.layers[0].actions[0].gate.array)
    [[0.  0.1]
     [0.1 0. ]]
     + i* 
    [[0. 0.]
     [0. 0.]]

    Complex conjugate of a circuit.
    >>>import qem
    >>>cir=qem.Cir()
    >>>cir.append_action(qem.Action(0,qem.X()))
    >>>cir.append_layer()
    >>>cir.append_action(qem.Action(1,qem.Y()))
    >>>cir=cir.complex_conjugate()
    >>>cir.print_layers()
    >>>print('The complex conjugate of X:\n', cir.layers[0].actions[0].gate.array)
    >>>print('The complex conjugate of Y:\n',cir.layers[1].actions[0].gate.array)
       Printing circuit, with layers:
       ==============
           Printing layer, with actions:
           ==============
           X conjugate [0]
           ==============
       --------------
           Printing layer, with actions:
           ==============
           Y conjugate [1]
           ==============
       ==============

    The complex conjugate of X:
    [[0. 1.]
    [1. 0.]]
    + i* 
    [[-0. -0.]
    [-0. -0.]]
    The complex conjugate of Y:
    [[0. 0.]
    [0. 0.]]
    + i* 
    [[-0.  1.]
    [-1. -0.]]


    Translate a circuit. (Note the cir.translate method is called.)
    >>>import qem
    >>>cir=qem.Cir()
    >>>cir.append_action(qem.Action(0,qem.X()))
    >>>cir.append_layer()
    >>>cir.append_action(qem.Action(1,qem.Y()))
    >>>cir=cir.complex_conjugate().translate(4)
    >>>cir.print_layers()
        Printing circuit, with layers:
        ==============
            Printing layer, with actions:
            ==============
            X conjugate [4]
            ==============
        --------------
            Printing layer, with actions:
            ==============
            Y conjugate [5]
            ==============
        ==============
    """
    def __init__(self):
        self.layers=[Layer()]

    def __rmul__(self,other):
        if type(other)==xp.float64 or type(other)==float:
            cirprime=Cir()
            first=True
            for layer in self.layers:  # Construct new cir containing gates multiplied by a number.
                if first==False:
                    cirprime.append_layer()
                    first=False
                for action in layer.actions:
                    cirprime.append_action(Action(action.qubits,other*action.gate))

            return cirprime

        else:
            TypeError('Wrong type encountered in multiplication of circuit by float.')

    def append_layer(self,*args):
      if len(args)==0:
          self.layers.append(Layer())
      elif len(args)==1:
          assert type(args[0])==list, 'Argument must be a list of actions.'
          self.layers.append(args[0])
      else:
          raise Exception('Zero or one arguments expected but recieved '+str(len(args)))
  
    def append_action(self,action):
        self.layers[-1].append_action(action)

    def print_layers(self):
        print('    Printing circuit, with layers:')
        print('    ==============')
        first=True
        for layer in self.layers:
            if first==False:
                print('    --------------')
            first=False
            layer.print_actions()
        print('    ==============')
        print()

    def complex_conjugate(self):
        cirprime=Cir()
        first=True

        for layer in self.layers:
            if first==True:
                first=False
            elif first==False:
                cirprime.append_layer()
            for action in layer.actions:
                new_array=action.gate.array.conjugate()
                new_gate=Gate(new_array,action.gate.name+' conjugate')
                cirprime.append_action(Action(action.qubits,new_gate))

        return cirprime

    def translate(self,q):
        cirprime=Cir()
        first=True
        for layer in self.layers:
            if first==True:
                first=False
            else:
                cirprime.append_layer()
            for action in layer.actions:
                trans_qubits=tuple((qubit+q for qubit in action.qubits))
                cirprime.append_action(Action(trans_qubits,action.gate))

        return cirprime

    def vectorize(self,n,m,part):
        # Prepare the circuit for application to a MixReg.psi.
        # n. The number of qubits of the reg
        # m. The number of classical bits of the reg
        # conj. if False, vectorize the circuit that would normally be left multiplied with rho. If True, vectorize the circuit that would normally be right multiplied by rho.
        #
        # If a single-(qu)bit gate is applied to a (qu)bit with index q>=
        #
        # For all actions in the circuit,check the qubits of the action (p,q). If p and or q are larger then, or equal to, n, they are replaced by p+n (or q+n).
        if part=='normal':
            cirprime=Cir()
            first=True
            for layer in self.layers:
                if first==True:
                    first=False
                else:
                    cirprime.append_layer()
                for action in layer.actions:
                    if len(action.qubits)==1:
                        p=action.qubits[0]
                        if p>=n:
                            assert action.gate.classical
                            assert p<n+m
                            p+=n
                        cirprime.append_action(Action(p,action.gate))

                    elif len(action.qubits)==2:
                        p,q=action.qubits
                        assert q<n, 'Second qubit index larger than number of available qubits. In case of a classical-quantum gate, the second index must always act on a qubit, not a classical bit'
                        assert p<n+m

                        if p>=n:
                            assert action.gate.classical_quantum
                            p+=n
                        cirprime.append_action(Action((p,q),action.gate))

                    else:
                        raise ValueError('Cir.vectorize can only handle one- and two-qubit gates.')
            return cirprime

        elif part=='conj': # Complex conjuate and translate the circuit. Note that if a gate acts ONLY on a mixed classical bit, this gate should be REMOVED. This is because it must only be applied once, and it has already be applied in the normal circuit. Classical-quantum gates do need to be applied twice. This is not in conflict with the precious because Classical-quantum gates do not change the state of the classical mixed bit. This bit is only used as control bit in classical-quantum gates.
            #
            #It is because the classical Kraus operators act from one side. the quantum Ktraus ops act from two sides.
            #
            #Actualy, what I call a classical gate above is already a classical channel, where you do certain operations with a certain probability. But because a purely classical channel is mathematically the same as a quantum single qubit gate up to unitarity (i.e. a 2x2 matrix), we call a classical channel a classical gate above. A purely classical channel is implemented as a (non-unitary) quantum gate and is therefore a Gate object.
            #
            #Such classical quantum channels may be put together in a circuit with other quantum gates. We have to be careful in implementing the conjugate circuit, because only the quantum and classical-quantum part must be conjugated and applied.
            #
            #The only problem remaining is that, at the moment, I cannot create channels with more than one classical kraus operator. This is because I explicitly check if a gate is tochastic whenever that gate is applied to the purely classical part of the reg. If there are multiple classical Kraus operators, the sum of the classical Kraus operators is stochastic, but, because of the weighting, not the classical Kraus operators seperaterly.
            cirprime=Cir()
            first=True
            for layer in self.layers:
                if first==True:
                    first=False
                elif first==False:
                    cirprime.append_layer()
                for action in layer.actions:
                    if not(len(action.qubits)==1 and action.qubits[0]>=n): # Only add gates to the new circ if not a purely classical
                        new_array=action.gate.array.conjugate()
                        new_gate=Gate(new_array,action.gate.name+' conjugate')
                        new_qubits=tuple((qubit+n for qubit in action.qubits))
                        cirprime.append_action(Action(new_qubits,new_gate))
            return cirprime
        else:
            raise ValueError("part must be 'normal' or 'conj'")

class MixReg:
    """
    (Potentially mixed-state) quantum register that holds density matrices rho, on n qubits and m classical mixed bits. MixReg(n) is internally stored as a Reg with 2n+m qubits. That is, the vectorized form of MixReg(n) is stored. The vectorized form a a density matrix rho is optained (in the tensor notation picture), by putting all the legs on the right of rho below the legs on the left of rho, without changing their order. We denote this state by vec(rho). We implement a gate on qubits (p,q) of rho by applying the gate on qubits (p,q) of vec(rho), and the complex conjugate of that gate (not the Hermitian conjugate!) on qubits (p+n,q+n) of vec(rho). To compute the expectation value of an operator on qubits (q_l,...,q_m), we apply that operator to qubits l,...,m of vec(rho) only. We then contract the leg of q_j of vec(rho) with the leg of q_{j+n} of vec(rho) for all j.

 If m>0, we have n mixed qubits (legs 0 up to and including 2n-1) and m classical bits (legs 2n up to and including 2n+m-1). A 1-bit mixed classical mixed state has entries [1-p,p].
    If only 1 parameter is given, it is assumed m=1.


    Attributes
    ----------
    psi : qem.Array
        See Reg.psi

    Methods
    -------
    print_ket_state()
        Like Reg.print_ket_state, but modified to print density matrices in human readable format. Mainly for debugging. 

    Examples
    --------
    >>> import qem
    >>> reg=qem.MixReg(2)
    >>> reg.print_ket_state()
    psi = (1+0j)|00><00|
    """
    def __init__(self,*args):
        if len(args)==1:
            self.n=args[0]
            self.m=0
        elif len(args)==2:
            self.n,self.m=args
        else:
            raise ValueError('MixReg takes one or two arguments.')
        re=xp.zeros((2,)*(2*self.n+self.m),dtype=xp.float64)
        re[(0,)*(2*self.n+self.m)]=1.0
        im=xp.zeros((2,)*(2*self.n+self.m),dtype=xp.float64)
        self.psi=Array(re,im)

    def print_ket_state(self):
        get_bin=lambda x: format(x, 'b').zfill(2*n+m) # Def function that converts ints to the binary representation and outputs it as string.
        n=self.n
        m=self.m
        re=self.psi.re
        im=self.psi.im
        if type(re)==ch.Variable:
            re=re.data
        if type(im)==ch.Variable:
            im=im.data
        psi=re+1j*im
        psi=psi.flatten()
        psi=xp.around(psi,6)
        out='rho = '
        for i in range(len(psi)):
            if psi[i]!=0:
                bi=get_bin(i)
                out+=str(psi[i])+'(|'+bi[:n]+'><'+bi[n:2*n]+'|) otimes |'+bi[2*n:]+'> + '
        out = out[:-3]
        print(out)

class Chan:
    """
    a Chan ('Channel') has a list of circuits. Every circuit acts as one Kraus operator when applied to a MixReg. Should always act on a MixReg. Can optionally be initialized with a list of Cirs.
    
    Attributes
    ----------
    cirs : list 
        List of circuits

    Methods
    -------
    append_circuit(cir) 
        Append the Cir cir to the list of circuits. 

    print_circuits()
        Print the gates in every circuit. 

    Examples
    --------
    Make a bitflip channel by adding circuits in two different ways. 
    >>> import qem
    >>> import numpy as np
    >>> p=.1 # Probability of doing a bitflip
    >>> cir=qem.Cir()
    >>> cir.append_action(qem.Action(1,qem.X()))
    >>> cir=np.sqrt(p)*cir
    >>> chan=qem.Chan([cir])
    >>> cir=qem.Cir()
    >>> cir.append_action(qem.Action(1,qem.ID()))
    >>> cir=np.sqrt(1-p)*cir
    >>> chan.append_circuit(cir)
    >>> chan.print_circuits()
      Printing channel, with circuits:
      ==============
        Printing circuit, with layers:
        ==============
            Printing layer, with actions:
            ==============
            0.31622776601683794*X [1]
            ==============
        ==============

        Printing circuit, with layers:
        ==============
            Printing layer, with actions:
            ==============
            0.9486832980505138*ID [1]
            ==============
        ==============
    """
    def __init__(self,*args):
        if len(args)==0:
            self.cirs=[]
        elif len(args)==1:
            assert type(args[0])==list or type(args[0])==tuple, 'Optional argument must be a list containing objects of type Cir.'
            for cir in args[0]:
                assert type(cir)==Cir, 'Optional argument must be a list containing objects of type Cir.'
            self.cirs=args[0]

        self.name=None
    

    def append_circuit(self,cir):
        self.cirs.append(cir)

    def print_circuits(self):
        print('  Printing channel, with circuits:')
        print('  ==============')
        for cir in self.cirs:
            cir.print_layers()

class Trace:
    """
    A special kind of channel that traces out a qubit and removes it. In principle, this could be written as a kraus map, but the following is easier. Must always act on mixed reg. Note that a Trace introduces a shift of the qubit indices.
    """
    def __init__(self,qubits): # qubits=qubits to trace out
        if type(qubits)==int: 
            self.qubits=(qubits,)
        elif type(qubits)==list:
            self.qubits=tuple(qubits)
        else:
            assert type(qubits)==tuple
            self.qubits=qubits    
    
    def print_circuits(self):
        print('  Printing channel:')
        print('    Trace out qubits', self.qubits)


class ClasTrace:
    """
    A special kind of channel that traces out a classical mixed bit.
    """
    def __init__(self,bits): # qubits=classical qubits to trace out
        if type(bits)==int:
            self.bits=(bits,)
        elif type(bits)==list:
            self.bits=tuple(bits)
        else:
            assert type(bits)==tuple
            self.bits=bits

    def print_circuits(self):
        print('  Printing channel:')
        print('    Trace out classical bits', self.bits)

class PrintState(Chan):
    """
    A special type of Channel, that tells the simulator (via apply_action) to print the current state of the reg. Quantumly, it is allways an ID gate on the 0th qubit. 
    """
    def __init__(self,*args):
        if len(args)==0:
            self.message=None
            self.style='markup'
        else:
            assert type(args[0])==str
            self.message=args[0]
            self.style='markup'
        if len(args)==2:
            assert type(args[1])==str
            self.style=args[1]

    def print_circuits(self):
        print('  Printing channel:')
        print('    PrintState called here')
        
class SupCir:
    """
    The supercircuit class. A supercircuit is a circuit containing superoperators (qem.Chan), where the first n locations are qubits, and the following m locations are classical mixed bits.

    Examples
    --------
    >>>bitflip=qem.bitflip_channel(0,.1)
    >>>bitphaseflip=qem.bitphaseflip_channel(1,.2)
    >>>sc=qem.SupCir([bitflip,bitphaseflip])
    """
    def __init__(self,*args):
        if len(args)==2:
            self.n=args[0]
            assert type(self.n)==int
            self.m=args[1]
            assert type(self.m)==int
            self.chans=[]
        elif len(args)==3:
            assert type(args[2])==list or type(args[2])==tuple, 'Optional argument must be a list containing objects of type qem.Chan.'
            for chan in args[2]:
                assert type(chan)==Chan or type(chan)==PrintState or type(chan)==Trace, 'Optional argument must be a list containing objects of type Chan.'
            self.chans=args[2]
        else:
            raise ValueError('SupCir must receive either 2 or 3 arguments.')

    def append_channel(self,chan):
        self.chans.append(chan)

    def print_channels(self):
        print('Printing super circuit, with channels:')
        print('==============')
        for chan in self.chans:
            chan.print_circuits()

# Functions

def round_assert(a,b):
    """
    Assert the xp arrays a and b are equal up to 4 decimal places.
    """
    return xp.testing.assert_array_equal(xp.around(a,decimals=5),xp.around(b,decimals=5))


def tensordot(a,b,c):
    """
    Return the tensordot of two objects from the class `Array`. Also see `numpy.tensordot`.
    """
    assert type(a)==type(b)==Array
    _im=a.im
    if type(_im)==ch.Variable:
            _im=_im.data
    zeros=xp.zeros(_im.shape)
    if xp.array_equal(_im,zeros):
        re=ch.functions.tensordot(a.re,b.re,c)
        im=ch.functions.tensordot(a.re,b.im,c)
    else:
        re=ch.functions.tensordot(a.re,b.re,c)-ch.functions.tensordot(a.im,b.im,c)
        im=ch.functions.tensordot(a.re,b.im,c)+ch.functions.tensordot(a.im,b.re,c)

    return Array(re,im)

def run(cir,reg):
    """
    Run the circuit cir on the register reg, thereby changing the quantum state of the register. 

    Paramters
    ---------
    cir : qem.Circuit
    reg : qem.Reg

    Examples
    --------
    Create a GHZ state on 8 qubits.
    >>> import qem
    >>> reg=qem.Reg(8)
    >>> cir=qem.Cir()
    >>> cir.append_action(qem.Action((0),qem.H()))
    >>> for i in range(7):
    ...     cir.append_layer()
    ...     cir.append_action(qem.Action((i,i+1),qem.CNOT()))
    >>> qem.run(cir,reg)
    >>> reg.print_ket_state()
    psi = (0.707107+0j)|00000000> + (0.707107+0j)|11111111>
    """                           
    for layer in cir.layers:
        for action in layer.actions:
            apply_action(action,reg)

def apply_channel(chan,mixreg):
    """
    Apply the Chan chan to the MixReg mixreg.


    Examples
    --------
    >>> reg=qem.MixReg(1)
    >>> qem.apply_channel(qem.bitflip_channel(0,.5),reg)
    >>> reg.print_ket_state()     
    """
    assert type(mixreg)==MixReg
    if type(chan)==Chan:
        n=mixreg.n
        m=mixreg.m
        runningsum=EmptyMixReg(n,m)
        #mixreg.psi.re.flags.writeable=False
        #mixreg.psi.im.flags.writeable=False
        for cir in chan.cirs:
            mixregprime=EmptyMixReg(n,m)
            mixregprime.psi=mixreg.psi
            run(cir.vectorize(n,m,'normal'),mixregprime)
            run(cir.vectorize(n,m,'conj'),mixregprime)
            runningsum.psi+=mixregprime.psi
        #mixreg.psi.re.flags.writeable=True
        #mixreg.psi.im.flags.writeable=True
        mixreg.psi=runningsum.psi
    elif type(chan)==Trace:
        ntrq=len(chan.qubits)
        re=xp.identity(2**ntrq).reshape((2,)*(ntrq*2))
        im=xp.zeros((2**ntrq,2**ntrq)).reshape((2,)*(ntrq*2))
        ID=Array(re,im)
        translatedqs=tuple((chan.qubits[i]+mixreg.n for i in range(ntrq)))
        axes=chan.qubits+translatedqs
        mixreg.psi=tensordot(ID,mixreg.psi,(range(ntrq*2),axes))
        mixreg.n=mixreg.n-ntrq
    elif type(chan)==ClasTrace:
        assert all(q>=mixreg.n for q in chan.bits), 'Classical trace must act on classical bits.'
        assert len(chan.bits)==1, 'Temporarily only able to trace out a single classical bit at once.'
        ntrq=len(chan.bits)
        re=xp.array([1.,1.])
        im=xp.array([0.,0.])
        ones=Array(re,im)
        transqs=tuple((chan.bits[i]+mixreg.n for i in range(ntrq)))
        for transq in transqs:
            mixreg.psi=tensordot(ones,mixreg.psi,((0,),(transq,)))
            mixreg.m-=1

    elif type(chan)==PrintState:
        print('Printing current state')
        if chan.message!=None:
            print(chan.message)
        if chan.style=='markup':
            mixreg.print_ket_state()
        if chan.style=='flat':
            print(mixreg.psi.re.data.flatten()+1j*mixreg.psi.im.data.flatten())
        else:
            raise ValueError('chan must be of type Chan, Trace, ClasTrace, or PrintState')

def apply_supcir(supcir,mixreg):
    for chan in supcir.chans:
        apply_channel(chan, mixreg)

def moveaxis(a,b,c):
    """
    Of the `qem.Array' a, move axis a to position b. Also see `numpy.moveaxis`.
    """
    re=ch.functions.moveaxis(a.re,b,c)
    im=ch.functions.moveaxis(a.im,b,c)
    return Array(re,im)
  
def apply_action(action, reg):
    """
    Applies the action to the register reg, thereby chainging the resister's state (reg.psi).

    Parameters
    ----------
    action : qem.Action

    reg : qem.Reg

    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> action=qem.Action((0),qem.H())
    >>> qem.apply_action(action,reg)
    >>> action=qem.Action((0,1),qem.CNOT())
    >>> qem.apply_action(action,reg)
    >>> print(reg.psi)
    variable([[0.70710678 0.        ]
              [0.         0.70710678]])
     + i* 
    variable([[0. 0.]
              [0. 0.]])
    """
    # qem.Heisenberg_exp gets a special treatment. This way it is faster. Note e^(-i angle/4) * e^(-i angle (XX+YY+ZZ)/4)=cos(angle/2) Id - i sin(angle/2) SWAP.
    if type(action.gate)==Heisenberg_exp:
        angle=action.gate.angle
        reg_id=EmptyReg(reg.n)
        reg_SWAP=EmptyReg(reg.n)

        reg_id.psi.re=ch.functions.cos(angle/2.)*reg.psi.re
        reg_id.psi.im=ch.functions.cos(angle/2.)*reg.psi.im

        # Multiply SWAP term with sin
        reg_SWAP.psi.re=ch.functions.sin(angle/2.)*reg.psi.re
        reg_SWAP.psi.im=ch.functions.sin(angle/2.)*reg.psi.im

        # Multiply SWAP term with -i
        c=reg_SWAP.psi.re 
        reg_SWAP.psi.re=reg_SWAP.psi.im
        reg_SWAP.psi.im=-c

        # Do the SWAP
        reg_SWAP.psi.re=ch.functions.swapaxes(reg_SWAP.psi.re,*action.qubits)
        reg_SWAP.psi.im=ch.functions.swapaxes(reg_SWAP.psi.im,*action.qubits)

        # Add the SWAP term to the identity term
        reg.psi=reg_id.psi+reg_SWAP.psi

    # Also the gate Heisenberg() gets special treatment, very much like Heisenberg_exp. Note (XX+YY+ZZ)/4=SWAP/2-Id/4.
    elif type(action.gate)==Heisenberg:
        reg_id=EmptyReg(reg.n)
        reg_SWAP=EmptyReg(reg.n)

        reg_id.psi.re=-reg.psi.re/4
        reg_id.psi.im=-reg.psi.im/4

        reg_SWAP.psi.re=reg.psi.re/2
        reg_SWAP.psi.im=reg.psi.im/2

        reg_SWAP.psi.re=ch.functions.swapaxes(reg_SWAP.psi.re,*action.qubits)
        reg_SWAP.psi.im=ch.functions.swapaxes(reg_SWAP.psi.im,*action.qubits)

        # Add the SWAP term to the identity term
        reg.psi=reg_id.psi+reg_SWAP.psi
        
    else:
        n_legs=len(action.gate.array.shape)
        lower_legs=range(n_legs//2,n_legs)
        reg.psi=tensordot(action.gate.array,reg.psi,(lower_legs,action.qubits))
        reg.psi=moveaxis(reg.psi,range(n_legs//2),action.qubits)
            
def ground_state(g,k,return_state=False):
    """
    Compute the k lowest energies of the Heisenberg model defined on the graph g. (If k=1 only the ground state energy is computed.) The nodes of the graph need not be integers or coordinates (as is the case for test_graph_input.edges_fig() and related functions). If return_state=True, also the whole state vector is returned. 

    Optionally, a 'weight' attribute can be set for edges, which we will call w_e here. Then the Hamiltonain will read \sum_e w_e (X_e1 X_e2 + Y_e1 Y_e2 + Z_e1 Z_e2)/4. w_e defaults to 1 for the edges where no weight is given. 
    
    Parameters
    ----------
    g : list
        The graph on which the Heisenberg model is defined as a list of edges, where every edge is of the form (int,int).

    k : int
        The energy is computed of the k states with the lowest energy. For k=1 only the ground state energy is computed.

    return_state : Bool (optional)
        If true, also the whole state vector is returned.

    Returns
    -------
    w : numpy.ndarray (dtype=numpy.float64)
        Array containing the k lowest eigenvalues in increasing order.
        If return_state==True, also the state vectors are returned. Then the output is equal to that of scipy.linalg.eighs. This means in this case w=[b,v] with b the array containing the k lowest eigenvalues, and v an array representing the k eigenvectors. The column v[:, i] is the eigenvector corresponding to the eigenvalue w[i]. Note the ground state is retrned as a flat array (as opposed to the shape of e.g. qem.Reg.psi and functions as qem.basis_state()).
    

    Notes
    -----
    This function uses a Lanszos algorithm (ARPACK via scipy.sparse.linalg.eigsh) to compute the energy memory-efficiently. The storage of the complete Hamiltonian, even as a sparse matrix, can be very costly. Therefore, the Hamiltonian is suplied to scipy.linalg.eighs as a callable. That is, a function that receives the vector r and returns H.r (the Hamiltonian applied to r).
    
    """
    heisenberg_tensor_real=xp.array([[1,0,0,0],[0,-1,2,0],[0,2,-1,0],[0,0,0,1]],dtype=xp.float64)/4
    heisenberg_tensor_real=heisenberg_tensor_real.reshape((2,2,2,2)) # Note heisenberg tensor is real. So also the Hamiltonian, and any vector during the Lanszos algo will be real.

    nodes=[node for edge in g for node in edge]
    nodes=set(nodes)
    n=len(nodes)
    del nodes
    
    def Hv(v):
        v=xp.array(v,dtype=xp.float64)
        v=v.reshape((2,)*n)
        vp=xp.zeros((2,)*n,dtype=xp.float64)
        for edge in g:
            new_term=xp.tensordot(heisenberg_tensor_real,v,((2,3),edge))
            new_term=xp.moveaxis(new_term,(0,1),edge)
            vp+=new_term
        vp=vp.flatten()
        if GPU==True:
            vp=xp.asnumpy(vp)
        return vp

    H=scipy.sparse.linalg.LinearOperator((2**n,2**n),matvec=Hv)
    output=scipy.sparse.linalg.eigsh(H,k,which='SA',maxiter=numpy.iinfo(numpy.int32).max)
    if return_state==False:
        return output[0]
    else:
        return output


# Gates as functions

def apply_prepare_singlet(qubits,reg):
    action=Action(qubits, prepare_singlet())
    apply_action(action,reg)

def apply_H(qubits,reg):
    """
    Apply the Hadamard gate to control (int) of the register.

    Parameters
    ----------
    qubits : tuple containing one int
        The number of the qubit the gate is to be applied to
    
    reg : qem.reg
        The register the gate H is to be applied to.
    
    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> qem.apply_H(0,reg)
    >>> print(reg.psi)
    variable([[0.70710678 0.        ]
              [0.70710678 0.        ]])
    + i* 
    variable([[0. 0.]
              [0. 0.]])
    """
    action=Action(qubits, H())
    apply_action(action,reg)

def apply_X(qubits,reg):
    """
    Apply the X gate to reg. See `qem.apply_H`.
    """
    action=Action(qubits, X())
    apply_action(action,reg)

def apply_Y(qubits,reg):
    """
    Apply the Y gate to qubits reg. See `qem.apply_H`.
    """
    action=Action(qubits, Y())
    apply_action(action,reg)

def apply_Z(qubits,reg):
    """
    Apply the Z gate to reg. See `qem.apply_H`.
    """
    action=Action(qubits, Z())
    apply_action(action,reg)

def apply_CNOT(qubits,reg):
    """
    Apply the CNOT gate to reg. Qubits is a tuple of the form (int,int), containing the control and target qubit number (in that order).
    ----------
    qubits : tuple (int,int)
        Tuple containing the control and target qubit numner (in that order).

    reg : qem.reg
        The register the CNOT is to be applied to.

    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> qem.apply_H((0),reg)
    >>> qem.apply_CNOT((0,1),reg)
    >>> print(reg.psi)
    variable([[0.70710678 0.        ]
              [0.         0.70710678]])
    + i* 
    variable([[0. 0.]
              [0. 0.]])
    """
    action=Action(qubits, CNOT())
    apply_action(action,reg)

def Heisenberg_energy(g,reg):
    """
    Compute < reg.psi | H | reg.psi >, with H the Heisenberg Hamiltonian defined on the graph g.

    Parameters
    ----------
    g : list of edges or networkx.Graph 
        List of edges of the form (int,int) that define the graph. If it is a networkx.Graph object, this graph should already be mapped to ints (containing the 'old' node attribute completeness, but not required). In that case, the edges can additionally specify an edge attribute called 'weight'. This means that for this edge, the Hamiltonian is weight*(XX+YY+ZZ)/4.

    reg : qem.reg
        The resister containing the state for which the expectation value of the Hamiltonian is to be computed.   

    Retruns
    -------
    energy : chainer.Variable

    Example
    -------
    Compute the expectation value of the energy of the Neel state |0101> on a square. 
    >>> import qem
    >>> import numpy as np
    >>> edges=[(0,1),(1,2),(2,3),(3,0)]
    >>> reg=qem.Reg(4)
    >>> qem.apply_X(1,reg)
    >>> qem.apply_X(3,reg)
    >>> print(qem.Heisenberg_energy(edges,reg))
    variable(-1.)

    Compare this to the ground state energy of the Heisenberg model on the square.
    >>> qem.ground_state(edges,1)
    >>> print(qem.ground_state(g,1)[0].round())
    -2.0
    """
    E=0.
    reg_prime=EmptyReg(reg.n)
    gate=Heisenberg()
    for edge in g:
        reg_prime.psi=reg.psi 
        action=Action(edge,gate)
        apply_action(action,reg_prime)
        reg_prime.psi.do_dagger()
        E_term=tensordot(reg_prime.psi,reg.psi, (range(reg.n),range(reg.n)))
        E+=E_term.re
          
    return E


def X_channel(q):
    """
    X_channel(q) returns the channel where an X is applied to qubit q.   
    """
    cir=Cir()
    cir.append_action(Action(q,X()))
    chan=Chan([cir])
    return chan

def Y_channel(q):
    """
    Y_channel(q) returns the channel where an Y is applied to qubit q.   
    """
    cir=Cir()
    cir.append_action(Action(q,Y()))
    chan=Chan([cir])
    return chan

def Z_channel(q):
    """
    Z_channel(q) returns the channel where an Z is applied to qubit q.   
    """
    cir=Cir()
    cir.append_action(Action(q,Z()))
    chan=Chan([cir])
    return chan


def H_channel(q):
    """
    H_channel(q) returns the channel where a H is applied to qubit q.
    """
    cir=Cir()
    cir.append_action(Action(q,H()))
    chan=Chan([cir])
    return chan

def CNOT_channel(c,t):
    """
    CNOT_channel(c,t) returns the channel where a CNOT is applied with control qubit c and target qubit t.
    """
    cir=Cir()
    cir.append_action(Action((c,t),CNOT()))
    chan=Chan([cir])
    return chan

def CY_channel(c,t):
    """
    CY_channel(c,t) returns the channel where a CY is applied with control qubit c and target qubit t
    """
    cir=Cir()
    cir.append_action(Action((c,t),CY()))
    chan=Chan([cir])
    return chan

def CZ_channel(c,t):
    """
    CZ_channel(c,t) returns the channel where a CZ is applied with control qubit c and target qubit t.
    """
    cir=Cir()
    cir.append_action(Action((c,t),CZ()))
    chan=Chan([cir])
    return chan

def GHZ_channel(qubits):
    """
    GHZ_channel(qubits) returns the channel that would generate the (pure) GHZ-state on the qubits 'qubits'  when applied to a MixReg in the all-zero state. 
    Parameters
    ----------
    qubits : list or tuple of ints
    """
    cir=Cir()
    cir.append_action(Action(qubits[0],H()))
    for i in range(len(qubits)-1):
        cir.append_action(Action((qubits[i],qubits[i+1]),CNOT()))

    chan=Chan([cir])

    return chan

def RX_channel(q,angle):
    """
    RX_channel(q) returns the channel where a RX is applied to qubit q, with angle 'angle'.
    """
    assert type(angle)==ch.Variable
    cir=Cir()
    cir.append_action(Action(q,RX(angle)))
    chan=Chan([cir])
    return chan

def RY_channel(q,angle):
    """
    RY_channel(q) returns the channel where a RY is applied to qubit q, with angle 'angle'.
    """
    assert type(angle)==ch.Variable
    cir=Cir()
    cir.append_action(Action(q,RY(angle)))
    chan=Chan([cir])
    return chan

def compute_s(reg):
    """
    Checks if the state of the register is an eigenstate of the total spin operator S^2. If it is not an eigenstate, it returns None. If it is an eigenstate, it returns the quantum number S, defined by S^2|psi>=s(s+1)|psi>, with |psi> the eigenstate. 
    """
    #Check that the norm of the state of the register is unity
    norm=tensordot(reg.psi.dagger().flatten(),reg.psi.flatten(),((0),(0)))
    norm=xp.sqrt(norm.re.array**2+norm.im.array**2)
    assert xp.around(norm,5)==1., 'State of the register is not normalized.'
    reg_prime=Reg(reg.n)
    reg_prime.psi=Array(xp.zeros(reg.psi.shape), xp.zeros(reg.psi.shape))

    for i in range(reg.n):
        for j in range(reg.n):
            reg_prime_prime=deepcopy(reg)
            apply_X(j,reg_prime_prime)
            apply_X(i,reg_prime_prime)
            reg_prime.psi=reg_prime.psi + reg_prime_prime.psi

    for i in range(reg.n):        
        for j in range(reg.n):
            reg_prime_prime=deepcopy(reg)
            apply_Y(j,reg_prime_prime)
            apply_Y(i,reg_prime_prime)
            reg_prime.psi=reg_prime.psi + reg_prime_prime.psi

    for i in range(reg.n):
        for j in range(reg.n):
            reg_prime_prime=deepcopy(reg)
            apply_Z(j,reg_prime_prime)
            apply_Z(i,reg_prime_prime)
            reg_prime.psi=reg_prime.psi + reg_prime_prime.psi

    inner=tensordot(reg.psi.dagger().flatten(),reg_prime.psi.flatten(),((0),(0)))
    norm=tensordot(reg_prime.psi.dagger().flatten(),reg_prime.psi.flatten(),((0),(0)))
    norm=xp.sqrt(norm.re.array**2+norm.im.array**2)
    if xp.around(norm,5)==0.:
        print('State of register is eigenstate of the total spin operator, with s=0')
        return 0.
    elif xp.around(xp.sqrt(inner.re.array**2+inner.im.array**2)/norm,5)!=1.:
        print('State of register is not an eigenstate of the total spin operator')
        return None
    elif xp.around(xp.sqrt(inner.re.array**2+inner.im.array**2)/norm,5)==1.:
        print('State of register is eigenstate of the total spin operator, with')
        s=-1/2+1/2*xp.sqrt(1+4*norm)
        print('s=',s)
        return s
    else:
        raise ValueError()

def expectation(cir,reg):
    """
    Returns the expectation value <psi|U|psi>, where psi is the state of the register (=reg.psi) and U is the unitary induced by the circuit.
    
    Parameters
    ----------
    cir : qem.Circuit
      
    reg : qem.Reg

    Returns
    -------
    ex : qem.Array (with qem.re a chainer.Variable with shape (), likewise for qem.im)
    

    Examples
    --------
    Compute the expectation value <psi|Z_0 Z_1|psi> with |psi>=|0000>:
    >>> import qem
    >>> reg=qem.Reg(4)
    >>> cir=qem.Cir()
    >>> cir.append_action(qem.Action((0),qem.Z()))
    >>> cir.append_action(qem.Action((1),qem.Z()))
    >>> print(qem.expectation(cir,reg))
    variable(1.)
    + i* 
    variable(0.)
    """
    reg_prime=Reg(reg.n)
    reg_prime.psi=reg.psi
    run(cir,reg_prime)
    reg_prime.psi.do_dagger()
    ex=tensordot(reg.psi, reg_prime.psi,(range(reg.n),range(reg.n)))
    return ex

def spin_spin_correlation(i,j,reg):
    """
    Calculates the spin_spin correlation ( < S_i . S_j > - < S_i > . < S_j > ), with S=(X,Y,Z)/2.

    Parameters
    ----------
    i : int
        Number of the first spin.

    j : int
        Number of the second spin.

    reg : qem.Reg
        The quantum register hosting the spins i and j.

    Returns 
    -------
    c : qem.Array
        The value of the correlation as a qem.Array object (with c.re a chainer.Variable of shape () and likewise for c.im .)
    
    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> qem.apply_H((0),reg)
    >>> qem.apply_CNOT((0,1),reg)
    >>> cor=qem.spin_spin_correlation(0,1,reg)
    >>> print(cor)
    variable(0.25)
    + i* 
    variable(0.)
    """
    #  ( < S_i . S_j > - < S_i > . < S_j > ) = 1/4( <X_i X_j> + <Y_i Y_j> + <Z_i Z_j> - <X_i><X_j> - <Y_i><Y_j> - <Z_i><Z_j> )
    def double(i,j,gate):
        cir=Cir()
        cir.append_action(Action((i),gate))
        cir.append_action(Action((j),gate))
        return expectation(cir,reg)

    def single(i,j,gate):
        cir=Cir()
        cir.append_action(Action((i),gate))
        return expectation(cir,reg)*expectation(cir,reg)

    c=Array(xp.array(1/4.),xp.array(0.)) * ( double(i,j,X()) + double(i,j,Y()) + double(i,j,Z()) - single(i,j,X()) - single(i,j,Y()) - single(i,j,Z()) )

    return c

def infidelity(reg,reg_prime):
    """
    Computes the infidelity 1- |< reg.psi | reg_prime.psi >|^2  
    
    Returns
    -------
    inf : chainer.Variable with cupy array of shape ().
    """
    inner=tensordot(reg.psi.dagger(),reg_prime.psi,(range(reg.n),range(reg.n))) # Returns a qem.Array filled with ch.Variable s with cupy data shape ().
    inner_sq=inner*inner.dagger() # Same
    inf=1-inner_sq.re # The infidelity as 1 minus the SQUARED overlap. Cost is now a chainer Variable of shape ()
    return inf


def bitflip_channel(q,p):
    """
    bitflipchannel(q,p) returns bitflip channel on qubit q. The bitflip occurs with probability p.

    Examples
    --------
    >>> import qem
    >>> chan=qem.bitflip_channel(5,.1)
    >>> chan.print_circuits()
    Printing channel, with circuits:
    ==============
      Printing circuit, with layers:
      ==============
          Printing layer, with actions:
          ==============
          0.9486832980505138*ID [5]
          ==============
      ==============

      Printing circuit, with layers:
      ==============
          Printing layer, with actions:
          ==============
          0.31622776601683794*X [5]
          ==============
      ==============
    
    """
    cir1=Cir()
    cir1.append_action(Action(q,xp.sqrt(1-p)*ID()))

    cir2=Cir()
    cir2.append_action(Action(q,xp.sqrt(p)*X()))

    chan=Chan([cir1,cir2])
    
    return chan


def bitphaseflip_channel(q,p):
    """
    Return bitphaseflip channel on qubit q. It applies a Y with probability p to qubit q. Also see bitflip_channel()
    """
    cir1=Cir()
    cir1.append_action(Action(q,xp.sqrt(1-p)*ID()))

    cir2=Cir()
    cir2.append_action(Action(q,xp.sqrt(p)*Y()))

    chan=Chan([cir1,cir2])
    
    return chan

def phase_channel(q,p):
    """
    Return phaseflip channel on qubit q. It applies a Z with probability p to qubit q. Also see bitflip_channel()
    """
    cir1=Cir()
    cir1.append_action(Action(q,xp.sqrt(1-p)*ID()))

    cir2=Cir()
    cir2.append_action(Action(q,xp.sqrt(p)*Z()))

    chan=Chan([cir1,cir2])
    
    return chan

def entropy(mixreg):
    """
    entropy(mixreg) returns the entropy=-sum_i rho_i log(rho_i), where rho_i are the eigenvalues of the state rho of mixrreg.
    """
    re=mixreg.psi.re
    im=mixreg.psi.im
    if type(re)==type(im)==ch.Variable:
        re=re.array
        im=im.array
    a=re+1j*im
    a=a.reshape((2**mixreg.n,)*2)
    round_assert(a,a.conjugate().transpose())
    round_assert(xp.trace(a),1.)
    eigs=xp.linalg.eigvalsh(a).flatten()
    assert eigs.all()>=0
    eigs=eigs[eigs>=1e-10]
    eigs=xp.multiply(eigs,-xp.log2(eigs))
    ent=xp.sum(eigs)
    return ent
    

def random_mixreg(n):
    """
    Return a MixReg of n qubits in a ramdom pure state. Note it is not Haar random!
    """
    reg=MixReg(n)
    state=xp.random.rand(2**n)+1j*xp.random.rand(2**n)
    state=xp.array([state/xp.sqrt(xp.matmul(state.conjugate(),state))])
    state=xp.kron(state,state.conjugate().transpose())
    state=state.reshape((2,)*(2*n))
    reg.psi=Array(state.real,state.imag)
    return reg

def measure(q):
    """
    Returns the channel that measures qubit q in the computational basis. It stores the result as qubit with state |0><0| or |1><1| with the right probabilities. (I.e. as a mixed classical state.)
    """
    cir1=Cir()
    array=Array(xp.array([[1.,0.],[0.,0.]]),xp.array([[0.,0.],[0.,0.]]))
    gate=Gate(array,"Measurement: project on |0><0|")
    cir1.append_action(Action(q,gate))

    cir2=Cir()
    array=Array(xp.array([[0.,0.],[0.,1.]]),xp.array([[0.,0.],[0.,0.]]))
    gate=Gate(array,"Measurement: project on |1><1|")
    cir2.append_action(Action(q,gate))

    chan=Chan([cir1,cir2])
    
    return chan


def prepare_classical_mixed_state_channel_classical(p,q):
    """
    Returns the channel that throws away *classical bit* q, and reinitializes the state of that bit with [[1-p,p]]. Destroys correlations with the rest of the register.
    """
    re=xp.array([[1-p,1-p],[p,p]],dtype=xp.float64)
    im=xp.zeros((2,2),dtype=xp.float64)
    array=Array(re,im)
    gate=Gate(array,'clas_mix('+str(p)+')')
    action=Action(q,gate)
    cir=Cir()
    cir.append_action(action)
    chan=Chan()
    chan.append_circuit(cir)
    return chan

def prepare_classical_mixed_state_channel(p,q):
    """
    Returns the channel that throws away *qubit* q, and reinitializes the state of that qubit with [[1-p,0],[0,p]]. Destroys entranglement with the rest of the register.
    """
    chan=Chan()

    cir=Cir()

    re=xp.sqrt(1-p)*xp.array([[1.,0.],[0.,0.]])
    im=xp.zeros((2,2))
    array=Array(re,im)
    gate=Gate(array,'sqrt(1-p)|0><0|')
    action=Action(q,gate)
    cir.append_action(action)

    chan.append_circuit(cir)


    cir=Cir()

    re=xp.sqrt(1-p)*xp.array([[0.,1.],[0.,0.]])
    im=xp.zeros((2,2))
    array=Array(re,im)
    gate=Gate(array,'sqrt(1-p)|0><1|')
    action=Action(q,gate)
    cir.append_action(action)

    chan.append_circuit(cir)


    cir=Cir()

    re=xp.sqrt(p)*xp.array(  [[0.,0.],[1.,0.]]  )
    im=xp.zeros((2,2))
    array=Array(re,im)
    gate=Gate(array,'sqrt(p)|1><0|')
    action=Action(q,gate)
    cir.append_action(action)

    chan.append_circuit(cir)


    cir=Cir()

    re=xp.sqrt(p)*xp.array( [[0.,0.],[0.,1.]] )
    im=xp.zeros((2,2))
    array=Array(re,im)
    gate=Gate(array,'sqrt(p)|1><1|')
    action=Action(q,gate)
    cir.append_action(action)

    chan.append_circuit(cir)

    return chan

    
