import sys
sys.path.append('../')
import numpy as np
from physics import Hamiltonian
from optim import Optimizer
from utils import Circuit, r, rcv, rc, get_init_state, get_rotations, get_unitaries
from qiskit import QuantumCircuit, execute, Aer


all_true= True

n = 2
p = 1
param = np.pi/8.
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

fb_lin, fb1_lin, fb2_lin = opt.linear_time_fubini(c.params, blockwise=False)

print('exact')

print('fb1 lin:')
print(r(fb1_lin))
print('fb2 lin:')
print(r(fb2_lin))
#print('fb lin:')
#print(r(fb_lin))

"""
opt = Optimizer(circuit=c, n=n, sim_reps=100, init=init, exit=exit)

fb_lin, fb1_lin, fb2_lin = opt.linear_time_fubini(c.params, blockwise=False)
print('100 reps')

#print('fb1 lin:')
#print(r(fb1_lin))
#print('fb2 lin:')
#print(r(fb2_lin))
print('fb lin:')
print(r(fb_lin))


opt = Optimizer(circuit=c, n=n, sim_reps=1000, init=init, exit=exit)


fb_lin, fb1_lin, fb2_lin = opt.linear_time_fubini(c.params, blockwise=False)
print('1000 reps')

#print('fb1 lin:')
#print(r(fb1_lin))
#print('fb2 lin:')
#print(r(fb2_lin))
print('fb lin:')
print(r(fb_lin))


opt = Optimizer(circuit=c, n=n, sim_reps=10000, init=init, exit=exit)

fb_lin, fb1_lin, fb2_lin = opt.linear_time_fubini(c.params, blockwise=False)

print('10000 reps')

#print('fb1 lin:')
#print(r(fb1_lin))
#print('fb2 lin:')
#print(r(fb2_lin))
print('fb lin:')
print(r(fb_lin))
"""

opt = Optimizer(circuit=c, n=n, sim_reps=100000, init=init, exit=exit)


fb_lin, fb1_lin, fb2_lin = opt.linear_time_fubini(c.params, blockwise=False)
print('100000 reps')

print('fb1 lin:')
print(r(fb1_lin))
print('fb2 lin:')
print(r(fb2_lin))
#print('fb lin:')
#print(r(fb_lin))
