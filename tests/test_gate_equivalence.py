import sys
sys.path.append('../')
from utils import Gate, rc
from qiskit import QuantumCircuit, execute, Aer
import numpy as np

all_true = True

angle = np.pi/4.
n = 6
b_s = Aer.get_backend('statevector_simulator')
b_u = Aer.get_backend('unitary_simulator')

for i in range(n):
    gate_x = Gate('x', i, 0, param=angle)
    gate_y = Gate('y', i, 0, param=angle)
    gate_z = Gate('z', i, 0, param=angle)
    gate_zz = Gate('zz', i, (i+1)%n, param=angle)
    gate_x_q = Gate('x', n-1-i, 0, param=angle)
    gate_y_q = Gate('y', n-1-i, 0, param=angle)
    gate_z_q = Gate('z', n-1-i, 0, param=angle)
    gate_zz_q = Gate('zz', n-1-i, (n-i)%n, param=angle)

    gates = [gate_x, gate_y, gate_z, gate_zz]
    gates_q = [gate_x_q, gate_y_q, gate_z_q, gate_zz_q]

    for gate, gate_q in zip(gates, gates_q):
        init_state = np.array([complex(1., 0.)] + [complex(0., 0.) for _ in range(2**n - 1)])
        
        circuit = QuantumCircuit(n, n)
        circuit = gate_q.to_qiskit(circuit, angle)
        u_custom = gate.to_unitary(n)
        u_qiskit = execute(circuit, b_u).result().get_unitary()

        state_sim = execute(circuit, b_s).result().get_statevector()
        state_custom = np.matmul(u_custom, init_state)
        state_qiskit_u = np.matmul(u_qiskit, init_state) 

        custom_qiskit_u = np.allclose(state_custom, state_qiskit_u)
        custom_qiskit_sim = np.allclose(state_custom, state_sim)
        qiskit_u_qiskit_sim = np.allclose(state_sim, state_qiskit_u)
        
        print('check gate {} on qubit {}; custom vs qiskit_u: {}, custom vs qiskit_sim: {}, qiskits: {}'.format(gate.gate_type, i, custom_qiskit_u, custom_qiskit_sim, qiskit_u_qiskit_sim))

        if not custom_qiskit_u or not custom_qiskit_sim:
            all_true = False
            print('custom:')
            print(rc(u_custom))
            print('qiskit:')
            print(rc(u_qiskit))

print('-----------------------')
print('All tests successfull: {}'.format(all_true))
print('-----------------------')
