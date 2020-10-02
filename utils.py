import qiskit
from qiskit import Aer, execute
from qiskit import QuantumCircuit
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time


class Gate:
    def __init__(self, gate_type, target, control, param=None):
        self.gate_type = gate_type
        self.target = target
        self.control = control
        self.param = param
        
    def to_qiskit(self, circuit, param):
        if self.gate_type == 'h':
            circuit.h(self.target)
        elif self.gate_type == 'x':
            circuit.rx(param, self.target)
        elif self.gate_type == 'y':
            circuit.ry(param, self.target)
        elif self.gate_type == 'z':
            circuit.rz(param, self.target)
            
        elif self.gate_type == 'pauli_x':
            circuit.x(self.target)
        elif self.gate_type == 'pauli_y':
            circuit.y(self.target)
        elif self.gate_type == 'pauli_z':
            circuit.z(self.target)
        
        elif self.gate_type == 'xx':
            circuit.rxx(param, self.control, self.target)
        elif self.gate_type == 'yy':
            circuit.ryy(param, self.control, self.target)
        elif self.gate_type == 'zz':
            circuit.rzz(param, self.target, self.control)
        elif self.gate_type == 'cx':
            circuit.cx(self.control, self.target)
        elif self.gate_type == 'cz':
            circuit.cz(self.control, self.target)
        elif self.gate_type == 'cy':
            circuit.cy(self.control, self.target)
        elif self.gate_type == 'crx':
            circuit.crx(param, self.control, self.target)
        elif self.gate_type == 'cry':
            circuit.cry(param, self.control, self.target)
        elif self.gate_type == 'crz':
            circuit.crz(param, self.control, self.target)
        return circuit


    def to_unitary(self, n, param=None, decimals=False):
        c, t = self.control, self.target
        theta = param or self.param
        if theta is None:
            theta = 0.

        # one qubit gates
        if len(self.gate_type) == 1:
            if self.gate_type == 'x':
                non_trivial = np.array([[complex(np.cos(theta/2.),0.), complex(0., -np.sin(theta/2.))],
                                       [complex(0., -np.sin(theta/2.)), complex(np.cos(theta/2.),0.)]])
            elif self.gate_type == 'y':
                non_trivial = np.array([[complex(np.cos(theta/2.), 0.), complex(-np.sin(theta/2.), 0.)],
                                       [complex(np.sin(theta/2.), 0.), complex(np.cos(theta/2.), 0.)]])

            elif self.gate_type == 'z':
                non_trivial = np.array([[complex(1., 0.), complex(0., 0.)],
                                       [complex(0., 0.), complex(np.cos(theta), np.sin(theta))]])

            elif self.gate_type == 'h':
                non_trivial = np.array([[complex(1.,0.), complex(1.,0.)],
                                        [complex(1.,0.), complex(-1.,0.)]]) / np.sqrt(2.)

            unitary = tensor_product([
                np.identity(2**t, dtype=complex), non_trivial, np.identity(2**(n-t-1), dtype=complex)
            ])

        # pauli gates
        elif self.gate_type == 'pauli_x':
            non_trivial = np.array([[complex(0., 0.), complex(1., 0.)],[complex(1., 0.), complex(0., 0.)]])
            
            unitary = tensor_product([
                np.identity(2**t, dtype=complex), non_trivial, np.identity(2**(n-t-1), dtype=complex)
            ])

        elif self.gate_type == 'pauli_y':
            non_trivial = np.array([[complex(0., 0.), complex(0., -1.)],[complex(0., 1.), complex(0., 0.)]])
            
            unitary = tensor_product([
                np.identity(2**t, dtype=complex), non_trivial, np.identity(2**(n-t-1), dtype=complex)
            ])
        
        elif self.gate_type == 'pauli_z':
            non_trivial = np.array([[complex(1., 0.), complex(0., 0.)],[complex(0., 0.), complex(-1., 0.)]])

            unitary = tensor_product([
                np.identity(2**t, dtype=complex), non_trivial, np.identity(2**(n-t-1), dtype=complex)
            ])

        # controlled gates
        else:
            # currently only configured for zz
            if self.gate_type == 'cx':
                upper_diagonal = 0.
                lower_diagonal = 0.
                upper_off_diagonal = 1.
                lower_off_diagonal = 1.

            elif self.gate_type == 'cz':
                upper_diagonal = 1.
                lower_diagonal = -1.
                upper_off_diagonal = 0.
                lower_off_diagonal = 0.


            elif self.gate_type == 'xx':
                upper_diagonal = complex(np.cos(theta),0.)
                lower_diagonal = complex(np.cos(theta),0.)
                upper_off_diagonal = complex(0.,-np.sin(theta))
                lower_off_diagonal = complex(0.,-np.sin(theta))

            elif self.gate_type == 'yy':
                upper_diagonal = complex(np.cos(theta),0.)
                lower_diagonal = complex(np.cos(theta),0.)
                upper_off_diagonal = complex(-np.sin(theta),0.)
                lower_off_diagonal = complex(np.sin(theta),0.)

            elif self.gate_type == 'zz':
                diagonal_entry = complex(np.cos(theta), +np.sin(theta))
                #lower_diagonal = complex(np.cos(theta), np.sin(theta))
                upper_off_diagonal = 0.
                lower_off_diagonal = 0.

            # plug them in (use symmetry of zz gate)
            upper = max(c, t)
            lower = min(c, t)

            non_trivial = np.identity(2**(upper-lower+1), dtype=complex)
            for i in range(2**(upper-lower)):
                if i%2 == 1:
                    non_trivial[i, i] = diagonal_entry
            for i in range(2**(upper-lower), 2**(upper-lower+1)):
                if i%2 == 0:
                    non_trivial[i, i] = diagonal_entry

            """
            if c > t:
    
                non_trivial = np.identity(2**(abs(c-t)+1), dtype=complex)
                for x in range(2**(c-t-1)):
                    non_trivial[2*x+1,2*x+1] = upper_diagonal
                    non_trivial[2*x+1, 2*x + 2**(c-t)+1] = upper_off_diagonal
                for x in range(2**(c-t-1)):
                    non_trivial[2*x + 2**(c-t) +1 , 2*x + 2**(c-t) + 1] = lower_diagonal
                    non_trivial[2*x + 2**(c-t) + 1, 2*x + 1] =  lower_off_diagonal
        

            elif t > c:
    
                non_trivial = tensor_product([np.array([
                    [1.,0.],[0.,0.]],dtype=complex),np.identity(2**(t-c))])
                                             
                for x in range(2**(t-c-1)):
                    index = 2**(abs(t-c)) + 2*x 
                    non_trivial[index, index] = upper_diagonal
                    non_trivial[index+1, index+1] = lower_diagonal
                    non_trivial[index, index+1] = upper_off_diagonal
                    non_trivial[index+1, index] = lower_off_diagonal
            """

            trivial_lower = np.identity(2**(min(c,t)))
            trivial_upper = np.identity(2**(n-max(c,t)-1))

            # plug trivial and non-trivial parts together
            unitary = tensor_product([trivial_lower, non_trivial, trivial_upper])

            """rounding the result"""
            if not decimals == False:
                np.set_printoptions(formatter={'complex-kind': '{:.2f}'.format})

        return unitary


                

class Circuit:
    def __init__(self, n, init='random', backend='qasm_simulator'):
        self.n = n
        self.gate_list = []
        self.params = []
        self.param_config = []
        self.init_strat = init
        self.backend = Aer.get_backend(backend)
        
    
    def to_qiskit(self, params=None, gates=None, param_config=None, fb=False, measure=True):
        if fb == False:
            circuit = QuantumCircuit(self.n, self.n)
        elif fb:
            circuit = QuantumCircuit(self.n+1, self.n+1)
            
        param_config = param_config or self.param_config
        params = params or self.params
        gates = gates or self.gate_list
        
        for j, config in enumerate(param_config):
            if config == 'coll':
                param = params[0]
                for i in range(self.n):
                    gate = gates[i]
                    circuit = gate.to_qiskit(circuit, param)
                gates = gates[self.n:]
                params = params[1:]

            elif config == 'ind_layer':
                for i in range(self.n):
                    gate = gates[i]
                    param = params[i]
                    circuit = gate.to_qiskit(circuit, param)
                gates = gates[self.n:]
                params = params[self.n:]
                
            elif config == 'ind_gate':
                gate = gates[0]
                param = params[0]
                circuit = gate.to_qiskit(circuit, param)
                gates = gates[1:]
                params = params[1:]

            elif config == 'none_layer':
                for i in range(self.n):
                    gate = gates[i]
                    circuit = gate.to_qiskit(circuit, -1)
                gates = gates[self.n:]
                
            elif config == 'none_gate':
                gate = gates[0]
                circuit = gate.to_qiskit(circuit, -1)
                gates = gates[1:]

        if fb == False and measure:
            circuit.measure([i for i in range(self.n)], [i for i in range(self.n)])
        elif fb is True:
            circuit.measure([self.n], [self.n])
        return circuit
        
    
    def add_layer(self, layer_type, param_config):
        if param_config == 'coll':
            self.params += [rd.random()*2*np.pi]
        elif param_config == 'ind':
            param_config = 'ind_layer'
            self.params += [rd.random()*2*np.pi for _ in range(self.n)]
        elif param_config == 'none':
            param_config = 'none_layer'
        self.param_config += [param_config]
            
        if layer_type == 'xx':
            self.gate_list += [Gate('xx', i, (i+1)%self.n) for i in range(self.n)]
        elif layer_type == 'yy':
            self.gate_list += [Gate('yy', i, (i+1)%self.n) for i in range(self.n)]
        elif layer_type == 'zz':
            self.gate_list += [Gate('zz', i, (i+1)%self.n) for i in range(self.n)] 
        elif layer_type == 'crx':
            self.gate_list += [Gate('crx', i, (i+1)%self.n) for i in range(self.n)]
        elif layer_type == 'cry':
            self.gate_list += [Gate('cry', i, (i+1)%self.n) for i in range(self.n)]
        elif layer_type == 'crz':
            self.gate_list += [Gate('crz', i, (i+1)%self.n) for i in range(self.n)] 
        elif layer_type == 'x':
            self.gate_list += [Gate('x', i, None) for i in range(self.n)]
        elif layer_type == 'y':
            self.gate_list += [Gate('y', i, None) for i in range(self.n)]
        elif layer_type == 'z':
            self.gate_list += [Gate('z', i, None) for i in range(self.n)]
        elif layer_type == 'h':
            self.gate_list += [Gate('h', i, None) for i in range(self.n)]
        elif layer_type == 'cx':
            self.gate_list += [Gate('cx', i, (i+1)%self.n) for i in range(self.n)]
        elif layer_type == 'cy':
            self.gate_list += [Gate('cy', i, (i+1)%self.n) for i in range(self.n)]
        elif layer_type == 'cz':
            self.gate_list += [Gate('cz', i, (i+1)%self.n) for i in range(self.n)]
        
        
    def add_gate(self, gate_type, target, control, param='random'):
        if param == 'random':
            self.param_config.append('ind_gate')
            self.params.append(2 * np.pi * rd.random())
        elif param == 'none':
            self.param_config.append('none_gate')
        else:
            self.param_config.append('ind_gate')
            self.params.append(param)
        self.gate_list.append(Gate(gate_type, target, control))
        
    
    def get_param_config(self):
        return self.gate_list, self.param_config


""" ---------------- LINEAR ALGEBRA TOOLS -------------------"""


def get_unitaries(circuit, params=None, mode=None, shift=None, derivs=None):
    """
    returns the list of unitary matrices
    corresponding to the deconstructed circuit
    
    input:
    circuit of type 'Circuit' as defined above

    modes:
    "params": return unitaries with params as in 'params' argument
    "upper_shift": return unitaries with params corresponding to 
                   finite differences shift
    "lower_shift": same but with lower shift
    
    output:
    list of complex 2**n times 2**n matrices, scheduler

    """    

    unitaries = []

    if mode == 'params':
        params = deepcopy(params)
    elif mode == 'shift':
        params = [shift for i in range(len(circuit.params))]

    #backend = Aer.get_backend('unitary_simulator')
    gates = deepcopy(circuit.gate_list)

    """get derivative insertions to determine fb"""
    
    if mode == 'deriv':
        
        for gate in derivs:
            unitaries.append(gate.to_unitary(circuit.n))

        return unitaries


    for i, conf in enumerate(circuit.param_config): 
        
        if conf == 'coll':
            unitary = np.identity(2**circuit.n, dtype=complex)
            param = params[0]
            for j in range(circuit.n):
                gate = gates[j]
                unitary = np.matmul(gate.to_unitary(circuit.n, param=param), unitary)
            gates = gates[circuit.n:]
            params = params[1:]

            unitaries.append(unitary)

        elif conf == 'ind_layer':

            for j in range(circuit.n):
                
                gate = gates[j]
                param = params[j]
                unitary = gate.to_unitary(circuit.n, param=param)
                unitaries.append(unitary)
            
            gates = gates[circuit.n:]
            params = params[circuit.n:]

        else:
            raise 'please use layerwise architectures with parameters'

    return unitaries




def get_rotations(n, axis='x', angle=None):
    """
    returns a unitary matrix as complex numpy array
    corresponding to various circuit rotations:
    
    axis 'x' rotates x axis into the computational basis
    axis 'y' same for y axis
    axis '45' rotates the circuit between the Bloch-sphere axes
    """

    backend = Aer.get_backend('unitary_simulator')
    circuit = QuantumCircuit(n, n)
    if axis == 'x':
        for qubit in range(n):
            circuit.h(qubit)

    elif axis == 'y':
        for qubit in range(n):
            circuit.rz(np.pi, qubit)
            circuit.rx(-np.pi/2, qubit)

    elif axis == '45' or axis == '45_inv':
        """refers to the 45 degree Hamiltonian rotation"""
        for qubit in range(n):
            circuit.rz(np.pi/4., qubit)
            circuit.rx(np.pi/4., qubit)

    elif axis == 'identity':
        pass 


    elif axis == 'gradual_rot' or axis == 'gradual_rot_inv':
        for qubit in range(n):
            circuit.rz(angle, qubit)
            circuit.rx(angle, qubit)

    unitary = execute(circuit, backend).result().get_unitary()
    
    if axis.find('inv') > -1:
        unitary = adj(unitary)

    return unitary


def get_init_state(n, zeros=False):

    if zeros:
        state = np.array([complex(0., 0.) for _ in range(2**n)])
        return state

    state = [complex(1., 0.)]
    state += [complex(0., 0.) for _ in range(2**n - 1)]
    return np.array(state)    



def tensor_product(args):
    hold = 1
    for tensor in args:
        hold = np.kron(hold, tensor)
    return hold


"""return adjoint numpy matrix"""
def adj(matrix):
    return np.conj(matrix.transpose())


def rc(matrix, decimals=2):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix[i, j] = complex(round(np.real(matrix[i, j]), decimals), round(np.imag(matrix[i, j]), decimals))
    return matrix

def r(matrix, dec=2):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix[i,j] = round(matrix[i,j], dec)
    return matrix
     

def rcv(array, decimals=2):
    hold = []
    for a in array:
        hold.append(complex(round(np.real(a), decimals), round(np.imag(a), decimals)))
    return hold













"""--------------- PLOTTING ---------------------------------"""


def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)


def convert_complex_to_float(matrix):
    hold = np.array([np.array([0. for _ in range(len(matrix[0]))]) for _ in range(len(matrix[0]))])
    if isinstance(matrix[0][0], complex):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                    hold[i][j] = np.real(matrix[i][j])

    return hold


    
def plot_fubini(matrices, iterations, savename=''):
    
    fig, axs = plt.subplots(1, len(matrices), figsize=(20, 2))
    for i in range(len(matrices)):
        axs[i].imshow(matrices[i], interpolation='nearest', vmin=0, vmax=1)
        axs[i].set_title('iter: '+str(iterations[i]))

        numrows, numcols = matrices[i].shape

        axs[i].format_coord = format_coord
    plt.tight_layout()
    plt.show()
    fig.savefig('fubini_'+savename+'.png', bbox_inches="tight",  dpi=1000)


def plot_score(score, savename=''):
    
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    axs.plot([i for i in range(len(score))], score, c="royalblue")
    axs.set_title('normed energy during optimization')
    axs.set_xlabel('iteration')
    axs.set_ylabel('normed energy')

    plt.show()
    fig.savefig('saves/score_'+savename+'.png', bbox_inches="tight",  dpi=1000)
    
