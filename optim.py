from copy import deepcopy
from qiskit import Aer, execute
import numpy as np

from physics import Hamiltonian
from utils import Circuit, Gate

class Optimizer:
    def __init__(self, x=None, circuit=None, grad_reps=100, fubini_reps=100, max_iter=100, lr=0.1):
        self.x0 = x
        self.circuit = circuit
        self.rep_count = 0
        self.grad_reps = 100
        self.lr = lr
        self.fubini_reps = fubini_reps
        
    
    def get_gradient(self, obj, x):
        """
        Determining vanilla gradient using the Gradient shift rule as in
        https://arxiv.org/abs/1811.11184
        
        Input: 
        callable 'obj' cost function
        list of current parameter configuration 'x'
        
        Output:
        gradient as list with dimension equal to 'x'
        """
        x = x or self.x0
        deriv = np.zeros(len(x))
        deriv = [0. for _ in range(len(x))]
        for i in range(len(x)):
            upper_shift = deepcopy(x)
            upper_shift[i] += np.pi/2        
            
            circuit_upper = self.circuit.to_qiskit(params=upper_shift)
            result_upper = execute(circuit_upper, self.circuit.backend, shots=self.grad_reps).result().get_counts()
            self.rep_count += self.grad_reps
            
            lower_shift = deepcopy(x)
            lower_shift[i] -= np.pi/2
            
            circuit_lower = self.circuit.to_qiskit(params=lower_shift)
            result_lower = execute(circuit_lower, self.circuit.backend, shots=self.grad_reps).result().get_counts()
            self.rep_count += self.grad_reps
        
            deriv[i] = self.lr * (obj(result_upper) - obj(result_lower))
        
        return deriv
    
    
    def get_vanilla_fubini_term1(self, params, draw=False):
        """
        compute naive fubini-study metric for parameters 'params'.
        Estimate expectation values according to figure 2 in
        Y. Li  and  S. Benjamin:   
        Efficient  Variational Quantum  Simulator  Incorporating  Active  Error  Minimization
        Phys. Rev. X, 7:021050, Jun 2017
        
        Input:
        list of current parameter configuration 'params'
        
        output:
        np.array of dim(params) \times dim(params) with fb_metric
        """
        gates, param_config = self.circuit.get_param_config()
        gate_assign, config_assign, counts = get_fubini_protocol(self.circuit.n, param_config)
        
        num_params = len(params)
        reps = self.fubini_reps
        fb = np.zeros((num_params, num_params))
        
        """computing the first term"""
        for i, (con_one, a_one, c_one) in enumerate(zip(config_assign, gate_assign, counts)):
            for j, (con_two, a_two, c_two) in enumerate(zip(config_assign[i:], gate_assign[i:], counts[i:])):
                
                idx_one = con_one
                idx_two = con_two# + i
                val = 0.
                
                """specify array position of inserted gates"""
                if param_config[con_one].find('layer') > -1 or param_config[con_one] == 'coll':
                    shift_one = self.circuit.n
                else:
                    shift_one = 1
                if param_config[con_two].find('layer') > -1 or param_config[con_two] == 'coll':
                    shift_two = self.circuit.n
                else:
                    shift_two = 1
                
                """sums arise for parameters applied to sums of generators"""
                for sums_one in range(c_one):
                    for sums_two in range(c_two):
                
                        deriv_gates = deepcopy(gates)
                        deriv_config = deepcopy(param_config)
                
                        """insert additional gate for second(!) derivative parameter and ancilla manipulation"""
                        insert_gates, insert_config = get_derivative_insertion(
                                                            deriv_gates[a_two+sums_two], self.circuit.n)
                        
                        deriv_gates = deriv_gates[:a_two+shift_two] + insert_gates + deriv_gates[a_two+shift_two:]
                        deriv_config = deriv_config[:idx_two+1] + insert_config + deriv_config[idx_two+1:]                       
                
                        """insert additional gate for first derivative parameter"""
                        insert_gates, insert_config = get_derivative_insertion(
                                                            deriv_gates[a_one+sums_one], self.circuit.n)
        
                        insert_gates = [Gate('pauli_x', self.circuit.n, 0)] + insert_gates + [Gate('pauli_x', self.circuit.n, 0)]
                        insert_config = ['none_gate'] + insert_config + ['none_gate']
                
                        deriv_gates = deriv_gates[:a_one+shift_one] + insert_gates + deriv_gates[a_one+shift_one:]
                        deriv_config = deriv_config[:idx_one+1] + insert_config + deriv_config[idx_one+1:]
                       

                        """insert gates to manipulate the ancilla"""
                        ancilla_gates = [Gate('h', self.circuit.n, 0)]#, Gate('pauli_x', self.circuit.n, 0)]
                        ancilla_config = ['none_gate']#, 'none_gate']
                        ancilla_gate_final = [Gate('h', self.circuit.n, 0)]
                        ancilla_config_final = ['none_gate']
                        
                        
                        deriv_gates = ancilla_gates + deriv_gates + ancilla_gate_final
                        deriv_config = ancilla_config + deriv_config + ancilla_config_final
                        
                        deriv_circuit = self.circuit.to_qiskit(
                                                fb=True,
                                                params=params,
                                                param_config=deriv_config,
                                                gates=deriv_gates
                                                )
                        if draw:
                            print(deriv_circuit)

                        result = execute(
                                deriv_circuit, 
                                self.circuit.backend, 
                                shots=reps
                                    ).result().get_counts()
                        #print(result)
                        self.rep_count += reps
                        
                        result = result['0'*(self.circuit.n+1)] / reps
                        val += 2 * result - 1
                        
                fb[i, j+i] = val
                fb[j+i, i] = val
                
                
        """computing the second term"""
        """to be added"""
        
        
        

        return fb
                
                
            
             

def get_fubini_protocol(n, param_config):
    """create protocol, assigning parameters to gates via listing the gates indexes"""
    gate_assign = []
    config_assign = []
    counts = []
    gate_count = 0
    for idx, config in enumerate(param_config):
        if config == 'none_layer':
            gate_count += n
        elif config == 'none_gate':
            gate_count += 1
        elif config == 'coll':
            counts += [n]
            gate_assign += [gate_count]
            config_assign += [idx]
            gate_count += n
        elif config == 'ind_layer':
            counts += [1 for _ in range(n)]
            gate_assign += [gate_count + i for i in range(n)]
            config_assign += [idx for i in range(n)]
            gate_count += n
        elif config == 'ind_gate':
            counts += [1]
            gate_assign += [gate_count]
            config_assign += [idx]
            gate_count += 1
    return gate_assign, config_assign, counts
                

    
def get_derivative_insertion(gate, n):
    target = gate.target
    gate_type = gate.gate_type
    if gate_type == 'x':
        return [Gate('cx', target, n)], ['none_gate']
    elif gate_type == 'y':
        return [Gate('cy', target, n)], ['none_gate']
    elif gate_type == 'z':
        return [Gate('cz', target, n)], ['none_gate']
    elif gate_type == 'xx':
        return [Gate('cx', target, n), Gate('cx', (target+1)%n, n)], ['none_gate', 'none_gate']
    elif gate_type == 'yy':
        return [Gate('cy', target, n), Gate('cy', (target+1)%n, n)], ['none_gate', 'none_gate']
    elif gate_type == 'zz':
        return [Gate('cz', target, n), Gate('cz', (target+1)%n, n)], ['none_gate', 'none_gate']

    
    
    

    