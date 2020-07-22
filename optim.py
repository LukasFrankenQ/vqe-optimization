from copy import deepcopy
from qiskit import Aer, execute
from qiskit import QuantumCircuit
import numpy as np

from physics import Hamiltonian
from utils import Circuit, Gate

class Optimizer:
    def __init__(self, x=None, circuit=None, n=4, grad_reps=100, fubini_reps=100, max_iter=100, lr=1., rot_circuit=False):
        self.x0 = x
        self.n = n
        self.circuit = circuit
        self.rep_count = 0
        self.grad_reps = 100
        self.lr = lr
        self.fubini_reps = fubini_reps
        self.rot_circuit = rot_circuit
        if self.rot_circuit:
            """define initial rotation"""
            self.init_circuit = QuantumCircuit(self.n+1, self.n+1)
            for qubit in range(self.n):
                self.init_circuit.rz(np.pi/4., qubit)
                self.init_circuit.ry(np.pi/4., qubit)
        
    
    def get_gradient(self, H, x, reps=None):
        """
        Determining vanilla gradient using the Gradient shift rule as in
        https://arxiv.org/abs/1811.11184
        
        Input: 
        callable 'obj' cost function
        list of current parameter configuration 'x'
        
        Output:
        gradient as list with dimension equal to 'x'
        """
        """assuming eigenvalue spectrum {+- 1} => r = max{+-1} and resulting in shift s:"""
        r = 1.
        s = np.pi / (4 * r)
        
        x = x or self.x0
        deriv = [0. for _ in range(len(x))]

        reps = reps or self.grad_reps
        
        for i in range(len(x)):      
            
            if (
                H.hamiltonian_type == "transverse_ising" or 
                H.hamiltonian_type == "spin_chain" or 
                H.hamiltonian_type == "rot_single_qubit_z"
                ):
                
                upper_shift = deepcopy(x)
                upper_shift[i] += s
                
                lower_shift = deepcopy(x)
                lower_shift[i] -= s
                
                upper_val = H.multiterm(self.circuit, upper_shift, reps=reps)
                lower_val = H.multiterm(self.circuit, lower_shift, reps=reps)
                
                deriv[i] = r * (upper_val - lower_val) * self.lr
                
            else:
                upper_shift = deepcopy(x)
                upper_shift[i] += s  
                
                circuit_upper = self.circuit.to_qiskit(params=upper_shift)
                result_upper = execute(circuit_upper, self.circuit.backend, shots=reps).result().get_counts()
                self.rep_count += reps
            
                lower_shift = deepcopy(x)
                lower_shift[i] -= s
            
            
                circuit_lower = self.circuit.to_qiskit(params=lower_shift)
                result_lower = execute(circuit_lower, self.circuit.backend, shots=reps).result().get_counts()
                self.rep_count += reps
        
                deriv[i] = r * (H.eval_dict(result_upper) - H.eval_dict(result_lower)) * self.lr
        
        return deriv
    
    
    def get_first_fubini_term(self, params, draw=False):
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
        first_term = np.zeros((num_params, num_params))
        
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
                            
                        if self.rot_circuit:
                            deriv_circuit = self.init_circuit + deriv_circuit

                        result = execute(
                                deriv_circuit, 
                                self.circuit.backend, 
                                shots=reps
                                    ).result().get_counts()
                        self.rep_count += reps
                        
                        try:
                            result = result['0'*(self.circuit.n+1)] / reps
                        except:
                            result = 0.
                    
                        val += 2 * result - 1
                        
                first_term[i, j+i] = val
                first_term[j+i, i] = val

        return first_term
                

        
    def get_second_fubini_term(self, params, draw=False):
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
        dot_products = np.zeros(num_params)
        
        for i, (con_one, a_one, c_one) in enumerate(zip(config_assign, gate_assign, counts)):
                
            idx_one = con_one
            val = 0.
                
            """specify array position of inserted gates"""
            if param_config[con_one].find('layer') > -1 or param_config[con_one] == 'coll':
                shift_one = self.circuit.n
            else:
                shift_one = 1
                
            """sums arise for parameters applied to sums of generators"""
            for sums_one in range(c_one):
                
                deriv_gates = deepcopy(gates)
                deriv_config = deepcopy(param_config)
                
                """insert additional gate for derivative parameter"""
                insert_gates, insert_config = get_derivative_insertion(
                                                    deriv_gates[a_one+sums_one], self.circuit.n)
        
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
                
                if self.rot_circuit:
                    deriv_circuit = self.init_circuit + deriv_circuit
                
                if draw:
                    print(deriv_circuit)

                result = execute(
                        deriv_circuit, 
                        self.circuit.backend, 
                        shots=reps
                            ).result().get_counts()
                self.rep_count += reps
                        
                try:
                    result = result['0'*(self.circuit.n+1)] / reps
                except:
                    result = 0.
                    
                val += 2 * result - 1
        
            dot_products[i] = val
        second_fubini_term = np.outer(dot_products, dot_products)

        return second_fubini_term
    
    
    
    
################################################################################


    def simulate_fubini_metric(self, params, draw=False):
        """
        Simulate metric
        
        Input:
        list of current parameter configuration 'params'
        
        output:
        np.array of dim(params) \times dim(params) with fb_metric
        """
        backend = Aer.get_backend("statevector_simulator")
        statevectors = []
        
        gates, param_config = self.circuit.get_param_config()
        gate_assign, config_assign, counts = get_fubini_protocol(self.circuit.n, param_config)
        
        num_params = len(params)
        
        for i, (con_one, a_one, c_one) in enumerate(zip(config_assign, gate_assign, counts)):
                
            idx_one = con_one
            current_state = np.array([0. + 0.j] for _ in range(2**num_params))
                
            """specify array position of inserted gates"""
            if param_config[con_one].find('layer') > -1 or param_config[con_one] == 'coll':
                shift_one = self.circuit.n
            else:
                shift_one = 1
                
            """sums arise for parameters applied to sums of generators"""
            for sums_one in range(c_one):
                
                deriv_gates = deepcopy(gates)
                deriv_config = deepcopy(param_config)
                
                """insert additional gate for derivative parameter"""
                insert_gates, insert_config = get_derivative_insertion(
                                                    deriv_gates[a_one+sums_one], self.circuit.n)
        
                deriv_gates = deriv_gates[:a_one+shift_one] + insert_gates + deriv_gates[a_one+shift_one:]
                deriv_config = deriv_config[:idx_one+1] + insert_config + deriv_config[idx_one+1:]     
                              
                deriv_circuit = self.circuit.to_qiskit(
                                            params=params,
                                            param_config=deriv_config,
                                            gates=deriv_gates
                                            )
                
                if self.rot_circuit:
                    deriv_circuit = self.init_circuit + deriv_circuit
                
                if draw:
                    print(deriv_circuit)

                current_state += execute(
                        deriv_circuit, 
                        backend, 
                            ).result().get_statevector()
        
            statevectors += current_state
        
        """get vanilla circuit state"""
        vanilla_circuit = self.circuit.to_qiskit(params=params)
        if self.rot_circuit:
            vanilla_circuit = self.init_circuit + vanilla_circuit
        vanilla_state += execute(
                        vanilla_circuit, 
                        backend, 
                            ).result().get_statevector()
                                     
        """get first fubini term"""
        fb1 = np.array([[0. + 0.j for _ in range(num_params)] for _ in range(num_params)])
                                     
        
        for i, state1 in enumerate(statevector):
            for j, state2 in enumerate(statevector[:i+1]):
         
                fb1[i,j] = np.inner(np.conj(state1), state2)
                if j < i:
                    fb1[j,i] = np.conj(fb1[i,j])
            
        """get second fubini term"""    
        inner_products = [np.inner(np.conj(statevector[i]), vanilla_state) for i in range(num_params)]
        
        fb2 = np.outer(np.conj(inner_products), inner_products)
        
        """get full fubini-study metric"""
        fb = fb1 - fb2
        fb = (fb + np.conj(fb)) / 2.
                    
        return fb, fb1, fb2








################################################################################
            
             

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

    
        
