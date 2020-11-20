from copy import deepcopy
from qiskit import Aer, execute
from qiskit import QuantumCircuit
import numpy as np
import time

from physics import Hamiltonian
from utils import Circuit, Gate, convert_complex_to_float, get_unitaries, get_init_state, adj, rc, get_rotations
from utils import rcv

class Optimizer:
    def __init__(self, x=None, circuit=None, n=4, grad_reps=100, 
                       fubini_reps=100, max_iter=100, lr=1., rot_circuit=False, exact_gradient=True,
                       init=None, exit=None, sim_reps=0, eigen=None):
        self.x0 = x
        self.n = n
        self.eigen = eigen
        self.circuit = circuit
        self.rep_count = 0
        self.grad_reps = 100
        self.lr = lr
        self.sim_reps = sim_reps
        self.fubini_reps = fubini_reps
        self.rot_circuit = rot_circuit
        self.exact_gradient = exact_gradient
        self.init = init
        self.exit = exit
        
        """define initial rotation"""
        self.init_circuit = QuantumCircuit(self.n+1, self.n+1)    
        for qubit in range(self.n):
            self.init_circuit.h(qubit)

        if self.rot_circuit:
            for qubit in range(self.n):
                self.init_circuit.rz(np.pi/4., qubit)
                self.init_circuit.ry(np.pi/4., qubit)


        
    """x: params, H: computes cost of state, init: initial manipulations, exit: final manipulations"""
    def linear_time_grad(self, H, x, normalize=False):
        
        r = 1.
        s = np.pi / (4 * r)
    
        deriv = [0. for _ in range(len(x))]
    
        t0 = time.time()
        unitaries_circuit = get_unitaries(self.circuit, mode='params', params=x)
        unitaries_upper_shift = get_unitaries(self.circuit, mode='shift', shift=s)
        unitaries_lower_shift = get_unitaries(self.circuit, mode='shift', shift=-s)
        init_state = get_init_state(self.n)

        backend = Aer.get_backend('unitary_simulator')
   
        """we will incrementally add to right what we substract from left"""
        """left and right naming refers to |state> = u_left u_right |init>"""
        c = self.circuit.to_qiskit(params=x, measure=False)
        unitary_left = execute(c, backend).result().get_unitary()
 
        unitary_left = np.identity(2**self.n, dtype=complex)
        for u in unitaries_circuit:
            unitary_left = np.matmul(u, unitary_left)
 
        unitary_left = np.matmul(self.exit, unitary_left)
        unitary_right = self.init
 
        for k in range(len(x)):
          
            u = unitaries_circuit[k]
            u_up = unitaries_upper_shift[k]
            u_down = unitaries_lower_shift[k]

            """obtain shifted states"""
            upper_unitary = np.matmul(unitary_left, u_up)
            lower_unitary = np.matmul(unitary_left, u_down)
            
            upper_unitary = np.matmul(upper_unitary, unitary_right)
            lower_unitary = np.matmul(lower_unitary, unitary_right)
        
            upper_state = np.matmul(upper_unitary, init_state)
            lower_state = np.matmul(lower_unitary, init_state)
            
            val_upper = H.eval_state(upper_state)
            val_lower = H.eval_state(lower_state)
            deriv[k] = r * (val_upper - val_lower)

            """update unitaries"""
            unitary_left = np.matmul(unitary_left, adj(u))
            unitary_right = np.matmul(u, unitary_right)

        if normalize:
            deriv = list(np.array(deriv) / np.linalg.norm(np.array(deriv)))
 
        return deriv


    def eval_params(self, H, x):
        backend = Aer.get_backend("unitary_simulator")
        state = get_init_state(self.n)
        unitaries_circuit = get_unitaries(self.circuit, mode='params', params=x)
        circuit = np.identity(2**self.n, dtype=complex)
        for u in unitaries_circuit:
            circuit = np.matmul(u, circuit)
        circuit = np.matmul(circuit, self.init)
        circuit = np.matmul(self.exit, circuit)
        state = np.matmul(circuit, state)
        score = H.eval_state(state)
        return score
        

    def eval_params_qiskit(self, H, x):
        backend = Aer.get_backend("unitary_simulator")
        state = get_init_state(self.n)
        
        circuit = self.circuit.to_qiskit(params=x, measure=False)
        circuit = execute(circuit, backend).result().get_unitary()
        circuit = np.matmul(circuit, self.init)
        circuit = np.matmul(self.exit, circuit)
        
        state = np.matmul(circuit, state)
        print(f'qiskit state: {state}')
        score = H.eval_state(state)
        return score



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
                
                upper_val = H.multiterm(self.circuit, upper_shift, reps=reps, exact=self.exact_gradient)
                lower_val = H.multiterm(self.circuit, lower_shift, reps=reps, exact=self.exact_gradient)
                
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
        
        #deriv = list(np.array(deriv) / np.linalg.norm(np.array(deriv)))

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
    
    

    def simulate_fubini_metric(self, params, draw=False, angle=0.):
        
        """define initial and final rotation"""
        self.init_circuit = QuantumCircuit(self.n, self.n)
        """starting in y axis"""
        """
        for qubit in range(self.n):
            self.init_circuit.rz(np.pi, qubit)
            self.init_circuit.rx(-np.pi/2, qubit)        
            self.init_circuit.rz(angle, qubit)
            self.init_circuit.rx(angle, qubit)
        """
        """
        Simulate metric exactly. The method computes the derivative states and
        and determines the metric as the matrix of inner products 
        
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
            current_state = np.array([0. + 0.j for _ in range(2**self.n)])
            vanilla_state = np.array([0. + 0.j for _ in range(2**self.n)])
                
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
                                                    deriv_gates[a_one+sums_one], self.circuit.n, simulation=True)
        
                deriv_gates = deriv_gates[:a_one+shift_one] + insert_gates + deriv_gates[a_one+shift_one:]
                deriv_config = deriv_config[:idx_one+1] + insert_config + deriv_config[idx_one+1:]     
                              
                deriv_circuit = self.circuit.to_qiskit(
                                            params=params,
                                            param_config=deriv_config,
                                            gates=deriv_gates,
                                            measure=False
                                            )
                
                deriv_circuit = self.init_circuit + deriv_circuit
                
                if draw:
                    print(deriv_circuit)

                current_state += execute(
                        deriv_circuit, 
                        backend, 
                            ).result().get_statevector()
        
            statevectors += [current_state]
        
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
                                     
        
        for i, state1 in enumerate(statevectors):
            for j, state2 in enumerate(statevectors[:i+1]):
         
                fb1[i,j] = np.inner(np.conj(state1), state2)
                if j < i:
                    fb1[j,i] = np.conj(fb1[i,j])
            
        """get second fubini term"""    
        inner_products = [np.inner(np.conj(statevectors[i]), vanilla_state) for i in range(num_params)]
        
        fb2 = np.outer(np.conj(inner_products), inner_products)
        
        """convert to float"""
        fb1 = convert_complex_to_float(fb1)
        fb2 = convert_complex_to_float(fb2)

        """get full fubini-study metric"""
        fb = fb1 - fb2
        fb = (fb + np.conj(fb)) / 2.
        
        return fb, fb1, fb2





    def linear_time_fubini(self, x, blockwise=False, block_size=None, get_proxis=False, one_block=False,
                          smart=None):
        sim_reps = self.sim_reps
        
        unitaries = get_unitaries(self.circuit, mode='params', params=x)
 
        derivs = []
        for gate in self.circuit.gate_list:
            deriv_gate, _ = get_derivative_insertion(gate, self.n, simulation=True, linear=True)
            derivs += deriv_gate
        
        """transform gates to a list of unitaries"""
        derivs = get_unitaries(self.circuit, mode='deriv', derivs=derivs)
        
        init_state = get_init_state(self.n)
        left = get_rotations(self.n, axis='identity')
        right = get_rotations(self.n, axis='identity')

        for u in unitaries:
            left = np.matmul(u, left)

        """include init and exit"""
        left = np.matmul(self.exit, left)
        right = np.matmul(right, self.init)

        full_unitary = np.matmul(left, right)
        vanilla_state = np.matmul(full_unitary, init_state)
        
        if sim_reps == 0:
            deriv_states = []
           
            for i, conf in enumerate(self.circuit.param_config):
            
                if conf == 'coll':
              
                    deriv_state = get_init_state(self.n, zeros=True)
                    for j in range(self.n):
                        deriv_unitary = np.matmul(derivs[j], right)
                        deriv_unitary = np.matmul(left, deriv_unitary)
                        deriv_state += np.matmul(deriv_unitary, init_state)

                    deriv_states.append(deriv_state)
                    derivs = derivs[self.n:]
                    left = np.matmul(left, adj(unitaries[0]))
                    right = np.matmul(unitaries[0], right)
                    unitaries = unitaries[1:]


                elif conf == 'ind_layer':
                 
                    for j in range(self.n):    
                        deriv_unitary = np.matmul(derivs[0], right)
                        deriv_unitary = np.matmul(left, deriv_unitary)
                        deriv_state = np.matmul(deriv_unitary, init_state)
                        deriv_states.append(deriv_state)
        
                        derivs = derivs[1:]
            
                        left = np.matmul(left, adj(unitaries[0]))
                        right = np.matmul(unitaries[0], right)
                        unitaries = unitaries[1:]
    
        
            num_params = len(x)
            """get first fubini term"""
            fb1 = np.array([[0. + 0.j for _ in range(num_params)] for _ in range(num_params)])
 
            """remove global phase"""
            for i in range(num_params):
                factor = np.conj(deriv_states[i][0]) / np.abs(deriv_states[i][0])
                deriv_states[i] = factor * deriv_states[i]
            factor = np.conj(vanilla_state[0]) / np.abs(vanilla_state[0])
            vanilla_state = factor * vanilla_state
                       
            for i, state1 in enumerate(deriv_states):
                for j, state2 in enumerate(deriv_states[:i+1]):
         
                    fb1[i,j] = np.inner(np.conj(state1), state2)
                    if j < i:
                        fb1[j,i] = np.conj(fb1[i,j])

            """get second fubini term"""    
            scalar_products = [np.vdot(deriv_states[i], vanilla_state) for i in range(num_params)]
            fb2 = np.outer(np.conj(scalar_products), scalar_products)

            """
            simulates measurement based computation of the Fubini-study metric 
            with sim_reps measurements per scalar product
            """

        elif sim_reps > 0.:
            
            deriv_states = []
            """state_summands = []"""

            for i, conf in enumerate(self.circuit.param_config):
            
                if conf == 'coll':
              
                    state_summands = []
                    for j in range(self.n):
                        deriv_unitary = np.matmul(derivs[j], right)
                        deriv_unitary = np.matmul(left, deriv_unitary)
                        state_summands.append(np.matmul(deriv_unitary, init_state))

                    deriv_states.append(state_summands)
                    derivs = derivs[self.n:]
                    left = np.matmul(left, adj(unitaries[0]))
                    right = np.matmul(unitaries[0], right)
                    unitaries = unitaries[1:]

                elif conf == 'ind_layer':
                 
                    for j in range(self.n):    
                        deriv_unitary = np.matmul(derivs[0], right)
                        deriv_unitary = np.matmul(left, deriv_unitary)
                        deriv_state = np.matmul(deriv_unitary, init_state)
                        deriv_states.append([deriv_state])
        
                        derivs = derivs[1:]
            
                        left = np.matmul(left, adj(unitaries[0]))
                        right = np.matmul(unitaries[0], right)
                        unitaries = unitaries[1:]
                        
            num_params = len(x)
            """get first fubini term"""
            fb1 = np.array([[0. for _ in range(num_params)] for _ in range(num_params)])


            """remove global phase"""
            for i in range(num_params):
                factor = np.conj(deriv_states[i][0][0]) / np.abs(deriv_states[i][0][0])
                deriv_states[i] = [summand * factor for summand in deriv_states[i]] 
            factor = np.conj(vanilla_state[0]) / np.abs(vanilla_state[0])
            vanilla_state = factor * vanilla_state


            for i, states1 in enumerate(deriv_states):
                for j, states2 in enumerate(deriv_states):
                    if i > j:
                        fb1[i,j] = fb1[j,i]

                    else:
                        entry = 0.
                        for summand1 in states1:
                            for summand2 in states2:
                                """determine probability to measure 0 or 1 and according std"""
                                inner = round(np.real(np.inner(np.conj(summand1), summand2)), 8)
                                #inner = round(np.vdot(summand1, summand2), 8)
                                #p = 0.5*(inner + 1.)
                                p = abs(inner)
                                #std = np.sqrt(2.*p*(1.-p) / sim_reps)
                                std = np.sqrt(p*(1.-p) / sim_reps)
                                #entry += np.random.normal(loc=inner, scale=std)
                                entry += np.random.normal(loc=np.real(inner), scale=std)
                        fb1[i,j] = entry
                        
            scalar_products = np.array([complex(0., 0.) for _ in range(num_params)])
            for i, states in enumerate(deriv_states):
                
                for idx, summand in enumerate(states):
                    
                    inner = np.vdot(summand, vanilla_state)
                    
                    p = np.real(inner)
                    if p < 0:
                        p = 0.
                        
                    std_real = np.sqrt(p*(1.-p) / sim_reps)
                    p = np.imag(inner)
                    if p < 0:
                        p = 0.
                    std_imag = np.sqrt(p*(1.-p) / sim_reps)
                    
                    result_real = np.random.normal(loc=np.real(inner), scale=std_real)
                    result_imag = np.random.normal(loc=np.imag(inner), scale=std_imag)
                    
                    scalar_products[i] += complex(result_real, result_imag)
            
            fb2 = np.outer(np.conj(scalar_products), scalar_products)
                
         
        """convert to float"""
        #fb1 = convert_complex_to_float(fb1)
        #if sim_reps == 0:
        #fb2 = convert_complex_to_float(fb2)
        
        """get full fubini-study metric"""
        fb1 = np.real(fb1)
        fb2 = np.real(fb2)
        fb = fb1 - fb2

        if blockwise:
            for i in range(num_params):
                for j in range(num_params):
                    #if abs(i-j) > max(i,j)%(1+3*self.n):
                    if abs(i-j) > max(i,j)%(block_size):
                        fb[i,j] = 0.
                        fb1[i,j] = 0.
                        fb2[i,j] = 0.


        if smart is not None:
            not_used = np.diag(fb) < smart
            fb[not_used] = 0.
            fb[:, not_used] = 0.
            fb[not_used, not_used] = 1.


        if one_block:
            fb[:-block_size, :-block_size] = np.identity(num_params - block_size)
            block = fb[num_params - block_size:, num_params - block_size:]
            for i in range(int(num_params/block_size)):
                fb[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = block

        if not get_proxis:
            return fb, fb1, fb2
        else:
            return fb, fb1, fb2, scalar_products




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
                

    
def get_derivative_insertion(gate, n, simulation=False, linear=False):
    target = gate.target
    gate_type = gate.gate_type
    
    """in case of simulation based way to determine fb"""
    if gate_type == 'x' and simulation:
        return [Gate('pauli_x', target, n)], ['none_gate']
    elif gate_type == 'y' and simulation:
        return [Gate('pauli_y', target, n)], ['none_gate']
    elif gate_type == 'z' and simulation:
        return [Gate('pauli_z', target, n)], ['none_gate']
    elif gate_type == 'xx' and simulation:
        return [Gate('cx', target, (target+1)%n)], ['none_gate']
        #return [Gate('cx', target, n), Gate('cx', (target+1)%n, n)], ['none_gate', 'none_gate']
    elif gate_type == 'yy' and simulation:
        return [Gate('cy', target, (target+1)%n)], ['none_gate']
        #return [Gate('cy', target, n), Gate('cy', (target+1)%n, n)], ['none_gate', 'none_gate']
    elif gate_type == 'zz' and simulation and not linear:
        return [Gate('cz', target, (target+1)%n)], ['none_gate']
    elif gate_type == 'zz' and simulation and linear:
        return [Gate('zz', target, (target+1)%n, param=np.pi)], ['none_gate']
        #return [Gate('cz', target, n), Gate('cz', (target+1)%n, n)], ['none_gate', 'none_gate']
    
        """for measurement based approach"""
    elif gate_type == 'x':
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
