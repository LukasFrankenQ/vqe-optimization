import sys
sys.path.append('../')
import numpy as np
from physics import Hamiltonian
from optim import Optimizer
from utils import Circuit, rcv, rc, get_init_state, get_rotations, get_unitaries
from qiskit import QuantumCircuit, execute, Aer

all_true= True

n = 2
p = 2
param = 0.
H = Hamiltonian(n, hamiltonian_type='transverse_ising', t=1.)

init = get_rotations(n, axis='identity')
exit = get_rotations(n, axis='identity')

c = Circuit(n)

for _ in range(p):
    c.add_layer('x', 'ind')
    c.add_layer('z', 'ind')
    c.add_layer('y', 'ind')
    c.add_layer('zz', 'coll')

c.params = [param for _ in range(len(c.params))]
opt = Optimizer(circuit=c, n=n, sim_reps=0, init=init, exit=exit)

grad_linear = opt.linear_time_grad(H, c.params)
grad_vanilla = opt.get_gradient(H, c.params)

print('gradient methods equivalence: ', np.allclose(np.array(grad_linear), np.array(grad_vanilla)))

print('----------------')
print('all states show consistent behavior:: ', all_true)
print('----------------')
