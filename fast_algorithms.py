from qiskit import Aer, QuantumCircuit, execute
import numpy as np

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








def get_unitaries(params=None, mode=None, shift=None):
    for param in params




























def get_complex_identity(dim, init_state=False):
    if init_state:
        a = [complex(1., 0.)]
        a += [complex(0., 0.) for _ in range(dim-1)]
        return np.array(a)

    a = np.array([[complex(0., 0.) for _ in range(dim)] for _ in range(dim)])
    for i in range(dim):
        a[i, i] += 1.+0.j
    return a





