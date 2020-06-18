import numpy as np
import random as rd

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

        elif self.hamiltonian_type == 'single_qubit_z':
            minimum = -1. * self.n 
            maximum = 1. * self.n 
                    
        return maximum, minimum

    
    """takes measured dictionary and return value between 0 and 1"""
    def eval_dict(self, results):
        avg = 0.
        total = 0.
        for meas, count in results.items():
            meas = 2*np.array([float(meas[i]) for i in range(len(meas))]) - 1.
            
            if self.hamiltonian_type == 'sk':
                energy = np.dot(meas, np.matmul(self.matrix, meas))
            elif self.hamiltonian_type == 'single_qubit_z':
                energy = np.dot(np.ones(self.n), meas)
                
            avg += float(count) * energy
            total += float(count)
        avg /= total
        return (self.max_value - avg) / (self.max_value - self.min_value)

