import sys
sys.path.append('../')
import numpy as np
from physics import Hamiltonian
from utils import Circuit, rcv, rc, get_init_state, get_rotations, get_unitaries
from qiskit import QuantumCircuit, execute, Aer

all_true= True

n = 2
p = 2
H = Hamiltonian(n, hamiltonian_type='transverse_ising', t=1.)


param = np.pi/5.

b_s = Aer.get_backend('statevector_simulator')
b_u = Aer.get_backend('unitary_simulator')

init_state = get_init_state(n)

custom_c = Circuit(n)

for _ in range(p):
    custom_c.add_layer('x', 'ind')
    custom_c.add_layer('z', 'ind')
    custom_c.add_layer('y', 'ind')
    custom_c.add_layer('zz', 'coll')

custom_c.params = [param for _ in range(len(custom_c.params))]

for i, gate in enumerate(custom_c.gate_list):
    print(gate.gate_type, gate.target, gate.control)



unitaries = get_unitaries(custom_c, params = custom_c.params)
u_custom = get_rotations(n, axis='identity')
for u in unitaries:
    u_custom = np.matmul(u, u_custom)

fully_custom_state = np.matmul(u_custom, init_state)

custom_qiskit = custom_c.to_qiskit(measure=False)

qiskit_c = QuantumCircuit(n, n)
for _ in range(p):
    for q in range(n):
        qiskit_c.rx(param, q)
    for q in range(n):
        qiskit_c.rz(param, q)
    for q in range(n):
        qiskit_c.ry(param, q)
    for q in range(n):
        qiskit_c.rzz(param, q, (q+1)%n)

u_qiskit = execute(qiskit_c, b_u).result().get_unitary() 
s_qiskit = execute(qiskit_c, b_s).result().get_statevector() 

u_custom_qiskit = execute(custom_qiskit, b_u).result().get_unitary()
custom_state = np.matmul(u_custom_qiskit, init_state)
qiskit_u_state = np.matmul(u_qiskit, init_state)

val_u_qiskit = H.eval_state(qiskit_u_state)
val_s_qiskit = H.eval_state(s_qiskit)
val_u_custom = H.eval_state(fully_custom_state)
val_q_custom = H.eval_state(custom_state)

print(val_u_qiskit)
print(val_s_qiskit)
print(val_u_custom)
print(val_q_custom)

if not (val_u_qiskit == val_s_qiskit == val_u_custom == val_q_custom):
    all_true = False
    print(rcv(qiskit_u_state))
    print(rcv(s_qiskit))
    print(rcv(fully_custom_state))
    print(rcv(custom_state))
    
    for a,b,c,d in zip(qiskit_u_state, s_qiskit, fully_custom_state, custom_state):
        print(np.abs(a*np.conj(a)), np.abs(b*np.conj(b)), np.abs(c*np.conj(c)), np.abs(d*np.conj(d)))
    print(np.real(np.conj(qiskit_u_state)*qiskit_u_state))
    print(np.real(np.conj(s_qiskit)*s_qiskit))
    print(np.real(np.conj(fully_custom_state)*fully_custom_state))
    print(np.real(np.conj(custom_state)*custom_state))

print('----------------')
print('all states show consistent behavior:: ', all_true)
print('----------------')
