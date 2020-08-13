ifrom qiskit import Aer, QuantumCircuit, execute
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../vqe-optimization/')
from utils import Circuit



backend = Aer.get_backend('unitary_simulator')
n = 1

circuit = QuantumCircuit(n, n)
for qubit in range(n):
    circuit.rx(np.pi/2., qubit)

result = execute(circuit, backend).result()

print(result.get_unitary())
print(np.conj(result.get_unitary(circuit, decimals=2)))


"""x: params, H: computes cost of state, init: initial manipulations, exit: final manipulations"""
def linear_grad(x, H, init, exit, normalize=False)
    
    r = 1.
    s = np.pi / (4 * r)

    deriv = [0. for _ in range(len(x))]

    unitaries_circuit = get_circuit_unitaries(mode='params', params=x)
    unitaries_upper_shift = get_circuit_unitaries(mode='upper_shift', shift=s)
    unitaries_lower_shift = get_circuit_unitaries(mode='lower_shift', shift=s)
    init_state = get_complex_identity(2**self.n, init_state=True)

    backend = Aer.get_backend('unitary_simulator')
    
    c = self.circuit.to_qiskit(params=x)

    """we will incrementally add to right what we substract from left"""
    """left and right naming refers to |state> = u_left u_right |init>"""
    unitary_left = execute(c, backend).result().get_unitary()
    unitary_left = np.matmul(exit, unitary_left)
    unitary_right = init

    for k in range(len(k)):
        
        u = unitaries_circuit.pop()
        u_up = unitaries_upper_shift.pop()
        u_down = unitaries_lower_shift.pop()


        """obtain shifted states"""
        upper_unitary = np.matmul(unitary_left, np.conj(u_up))
        lower_unitary = np.matmul(unitary_left, np.conj(u_down))
    
        upper_unitary = np.matmul(upper_unitary, np.conj(unitary_right))
        lower_unitary = np.matmul(lower_unitary, np.conj(unitary_right))

        upper_state = np.matmul(upper_unitary, init_state)
        lower_state = np.matmul(lower_unitary, init_state)

        val_uper = H.eval_state(upper_state)
        val_lower = H.eval_state(lower_state)

        deriv[k] = r * (val_upper - val_lower)

        """update unitaries"""
        unitary_left = np.matmul(unitary_left, np.conj(u))
        unitary_right = np.matmul(u, unitary_right)

    if normalize:
        deriv = list(np.array(deriv) / np.linalg.norm(np.array(deriv)))
     
    return deriv








def get_unitaries(self, params=None, mode=None, shift=None):
    
    unitaries = []

    if mode == 'params':
        params = deepcopy(params)
    elif mode == 'upper_shift':
        params = [shift for i in range(len(self.circuit.params)]
    elif mode == 'lower_shift':
        params = [-shift for i in range(len(self.circuit.params)]


    backend = Aer.get_backend('unitary_simulator')
    gates = deepcopy(self.circuit.gate_list)
    
    for i, conf in enumerate(self.circuit.param_config)):
        
        circuit = QuantumCircuit(self.n, self.n)

        if config == 'coll':
            param = params[0]
            for qubit in range(self.n):
                gates = gates[i]
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

        else:
            raise 'please use layerwise architectures with parameters'

        unitary = execute(circuit, backend).result().get_unitary()
        unitaries.append(unitaries)

    return unitaries






def get_rotations(n, axis='x'):
    backend = Aer.get_backend('unitary_simulator')
    circuit = QuantumCircuit(n, n)
    if axis == 'x':
        for qubit in range(n):
            circuit.h(qubit)
    
    elif axis == 'y':
        for qubit in range(n):
            circuit.rz(np.pi, qubit)
            circuit.rx(np.pi/2, qubit)

    '''refers to the 45 degree Hamiltonian rotation''' 
    elif axis == '45':
        for qubit in range(n):
            circuit.rz(np.pi/4., qubit) 
            circuit.ry(np.pi/4., qubit) 

    unitary = execute(circuit, backend).result().get_unitary()
    return unitary




































                
            




























def get_complex_identity(dim, init_state=False):
    if init_state:
        a = [complex(1., 0.)]
        a += [complex(0., 0.) for _ in range(dim-1)]
        return np.array(a)

    a = np.array([[complex(0., 0.) for _ in range(dim)] for _ in range(dim)])
    for i in range(dim):
        a[i, i] += 1.+0.j
    return a





