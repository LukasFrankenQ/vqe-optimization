import numpy as np
import random as rd
from qiskit import execute, QuantumCircuit
from copy import deepcopy


class Hamiltonian:
    def __init__(self, n, hamiltonian_type='sk'):
        self.n = n
        self.hamiltonian_type = hamiltonian_type
        self.matrix = self.get_matrix()
        self.max_value, self.min_value = self.get_boundaries()
        
        
    def get_matrix(self):
        if self.hamiltonian_type == 'sk':
            vals = [-1., 1.]
            h = np.array([[0. for _ in range(k+1)]+[rd.choice(vals) for _ in range(self.n-k-1)] 
                          for k in range(self.n)])
            
        elif self.hamiltonian_type == 'single_qubit_z':
            return None
        
        elif self.hamiltonian_type == 'transverse_ising':
            n = self.n
            unity = np.identity(2)
            z = np.array([[1., 0.], [0., -1.]])

            h = np.zeros((2**n, 2**n))

            for i in range(n):
                matrix = 1.
                if i == n-1:
                    matrix = np.kron(matrix, z)
                    for _ in range(i-1):
                        matrix = np.kron(matrix, unity)
                else:
                    for _ in range(i):
                        matrix = np.kron(matrix, unity)
        
                matrix = np.kron(matrix, z)
                if i < n-1:
                    matrix = np.kron(matrix, z)
                for _ in range(n-2-i):
                    matrix = np.kron(matrix, unity)
                h += matrix
                 
        return h
    

    def get_boundaries(self):
        if self.hamiltonian_type == 'sk':
            zero = '0'
            minimum = np.inf
            maximum = -np.inf
   
            for i in range(2**self.n):
                conf = bin(i)[2:]
                conf = conf + zero*(self.n-len(conf))
                conf = list(conf)
                conf = 2*np.array([float(conf[i]) for i in range(len(conf))]) - 1.
                energy = np.dot(conf, np.matmul(self.matrix, conf))
                if energy < minimum:
                    minimum = energy
                if energy > maximum:
                    maximum = energy

        elif self.hamiltonian_type == 'single_qubit_z' or self.hamiltonian_type == 'transverse_ising':
            minimum = -1. * self.n 
            maximum = 1. * self.n 
                    
        return maximum, minimum

    
    """takes measured dictionary and return value between 0 and 1"""
    def eval_dict(self, results, hamiltonian_type=None):
        hamiltonian_type = hamiltonian_type or self.hamiltonian_type
        
        avg = 0.
        total = 0.
        for meas, count in results.items():
            meas = 2*np.array([float(meas[i]) for i in range(len(meas))]) - 1.
            
            if hamiltonian_type == 'sk' or hamiltonian_type == "transverse_ising":
                energy = np.dot(meas, np.matmul(self.matrix, meas))
            elif hamiltonian_type == 'single_qubit_z':
                energy = np.dot(np.ones(self.n), meas)
                
            avg += float(count) * energy
            total += float(count)
        avg /= total
        return (self.max_value - avg) / (self.max_value - self.min_value)
    
    
    """Evaluating cost for a parameter configuration for a Hamiltonian with multiple terms"""
    def multiterm(self, circuit, params, reps=100, J=0., t=1.):
        if self.hamiltonian_type == "transverse_ising":
            energy = 0.
            """compute configuration score on first term"""
            curr_circuit = circuit.to_qiskit(params=deepcopy(params))
            result = execute(curr_circuit, circuit.backend, shots=reps).result().get_counts()

            for meas, count in result.items():
                meas = 2*np.array([float(meas[i]) for i in range(len(meas))]) - 1.
                shift = np.array(list(meas)[1:] + [list(meas)[0]])
                energy += np.dot(meas, shift) * (-J) * float(count) / float(reps)
            
            """compute configuration score on second term"""
            circuit = deepcopy(circuit)
            circuit.add_layer('h', 'none')
            curr_circuit = circuit.to_qiskit(params=deepcopy(params))
            result = execute(curr_circuit, circuit.backend, shots=reps).result().get_counts()
            for meas, count in result.items():
                meas = 2*np.array([float(meas[i]) for i in range(len(meas))]) - 1.
                energy += np.dot(np.ones(self.n), meas) * (-t) * float(count) / float(reps)
        
        return (self.max_value - energy) / (self.max_value - self.min_value)
            
            
            
            