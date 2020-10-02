import sys
sys.path.append('../')
from utils import Circuit, rc, rcv, get_init_state, get_unitaries, get_rotations
from qiskit import QuantumCircuit, execute, Aer
import numpy as np


param = np.pi/2.
all_true = True
n = 1
p = 2

b_s = Aer.get_backend('statevector_simulator')
b_u = Aer.get_backend('unitary_simulator')

init_state = get_init_state(n)

custom_c = Circuit(n)

for _ in range(p):
    #custom_c.add_layer('x', 'ind')
    custom_c.add_layer('z', 'ind')
    custom_c.add_layer('y', 'ind')
    #custom_c.add_layer('zz', 'coll')

custom_c.params = [param for _ in range(len(custom_c.params))]

unitaries = get_unitaries(custom_c, params = custom_c.params)
u_custom = get_rotations(n, axis='identity')
for u in unitaries:
    u_custom = np.matmul(u, u_custom)

fully_custom_state = np.matmul(u_custom, init_state)

custom_qiskit = custom_c.to_qiskit(measure=False)

qiskit_c = QuantumCircuit(n, n)
for _ in range(p):
    for q in range(n):
        #qiskit_c.rx(param, q)
        qiskit_c.rz(param, q)
        qiskit_c.ry(param, q)
        #qiskit_c.rzz(param, q, (q+1)%n)

u_qiskit = execute(qiskit_c, b_u).result().get_unitary() 
s_qiskit = execute(qiskit_c, b_s).result().get_statevector() 

u_custom_qiskit = execute(custom_qiskit, b_u).result().get_unitary()
custom_state = np.matmul(u_custom_qiskit, init_state)
qiskit_u_state = np.matmul(u_qiskit, init_state)

test_1 = np.allclose(fully_custom_state, custom_state)
if not test_1:
    print('--------------')
    print('fully custom:')
    print(rcv(fully_custom_state))
    print('custom state:')
    print(rcv(custom_state))
    print('scalar product: ')
    print(np.abs(np.inner(np.conj(fully_custom_state), custom_state)))
test_2 = np.allclose(fully_custom_state, qiskit_u_state)
if not test_2:
    print('--------------')
    print('fully custom:')
    print(rcv(fully_custom_state))
    print('qiskit u state:')
    print(rcv(qiskit_u_state))
    print(np.abs(np.inner(np.conj(fully_custom_state), qiskit_u_state)))
test_3 = np.allclose(fully_custom_state, s_qiskit)
if not test_3:
    print('--------------')
    print('fully custom:')
    print(rcv(fully_custom_state))
    print('qiskit s state:')
    print(rcv(s_qiskit))
    print(np.abs(np.inner(np.conj(fully_custom_state), s_qiskit)))
test_4 = np.allclose(custom_state, qiskit_u_state)
if not test_4:
    print('--------------')
    print('custom state:')
    print(rcv(custom_state))
    print('qiskit u state:')
    print(rcv(qiskit_u_state))
    print(np.abs(np.inner(np.conj(custom_state), qiskit_u_state)))
test_5 = np.allclose(custom_state, s_qiskit)
if not test_5:
    print('--------------')
    print('custom state:')
    print(rcv(custom_state))
    print('qiskit s state:')
    print(rcv(s_qiskit))
    print(np.abs(np.inner(np.conj(custom_state), s_qiskit)))
test_6 = np.allclose(qiskit_u_state, s_qiskit)
if not test_6:
    print('--------------')
    print('qiskit u state:')
    print(rcv(qiskit_u_state))
    print('qiskit s state:')
    print(rcv(s_qiskit))
    print(np.abs(np.inner(np.conj(qiskit_u_state), s_qiskit)))

print('custom to unitary vs custom to qiskit: ', test_1)
print('custom to unitary vs qiskit via unitary sim: ', test_2)
print('custom to unitary vs qiskit via state sim: ', test_3)
print('custom to qiskit vs qiskit via unitary sim: ', test_4)
print('custom to qiskit vs qiskit via state sim: ', test_5)
print('qiskit via unitary sim vs qiskit via state sim: ', test_6)




if not test_1 or not test_2 or not test_3 or not test_4 or not test_5 or not test_6:
    all_true = False
print('-----------------------')
print('All tests successfull: {}'.format(all_true))
print('-----------------------')
