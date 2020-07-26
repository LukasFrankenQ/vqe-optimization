import qiskit
from qiskit import Aer
from qiskit import QuantumCircuit
import random as rd
import numpy as np
import matplotlib.pyplot as plt

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
            circuit.rzz(param, self.control, self.target)
        elif self.gate_type == 'cx':
            circuit.cx(self.control, self.target)
        elif self.gate_type == 'cz':
            circuit.cz(self.control, self.target)
        elif self.gate_type == 'cy':
            print(self.control, self.target, param)
            circuit.cy(self.control, self.target)
        elif self.gate_type == 'crx':
            circuit.crx(param, self.control, self.target)
        elif self.gate_type == 'cry':
            circuit.cry(param, self.control, self.target)
        elif self.gate_type == 'crz':
            circuit.crz(param, self.control, self.target)
        return circuit
        
                

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
    fig.savefig('saves/fubini_'+savename+'.png', bbox_inches="tight",  dpi=1000)


def plot_score(score, savename=''):
    
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    axs.plot([i for i in range(len(score))], score, c="royalblue")
    axs.set_title('normed energy during optimization')
    axs.set_xlabel('iteration')
    axs.set_ylabel('normed energy')

    plt.show()
    fig.savefig('saves/score_'+savename+'.png', bbox_inches="tight",  dpi=1000)
    
