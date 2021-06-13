from optim import Optimizer, get_derivative_insertion
from physics import Hamiltonian
from utils import Circuit, Gate
import numpy as np
from utils import adj, plot_fubini, get_rotations, rc, get_unitaries
from qiskit import Aer, execute, QuantumCircuit



qubits = [2, 4, 6, 8]
block_sizes = [8, 16, 24, 32]

for n, block_size in zip(qubits, block_sizes):
    for dummy in [True]:
        reps = 10000
        grad_reps = reps
        fb_reps = reps
        alpha = None
        
        iters = 1
        angle = '45'
        rot = angle
        blockwise = True
        diagonal_only = False
        one_block = False
        fubini = True
        
        runs = 10000
        if reps == 0:
            runs = 1
        
        init_noise = 0.

        init = get_rotations(n, axis='y')
        exit = get_rotations(n, axis='identity')
    
        if angle == 'identity':
            pass
        elif angle == '45':
            init = np.matmul(get_rotations(n, "45"), init)
            exit = get_rotations(n, axis='45_inv')

        c = Circuit(n)
        num_layers = 4
        for _ in range(num_layers):
            c.add_layer('x', "ind")
            c.add_layer('y', "ind")
            c.add_layer('z', "ind")
            c.add_layer('zz', "ind")


        """sim reps > 0 yield simulated shot noise equal to noise after reps number of measurements"""
        opt = Optimizer(n=n, circuit=c, init=init, exit=exit, sim_reps=fb_reps)
        H = Hamiltonian(n, hamiltonian_type='transverse_ising', t=1., sim_reps=grad_reps)

        fb_eigs = []
        p = block_size

        for _ in range(runs):
        
            params = [np.random.rand() * 2 * np.pi for i in range(len(c.params))]
            print('-----------------------------------')
            print('n: ', n)
            print('p: ', p)
            print('fubini: ', fubini)
            print('blockwise: ', blockwise)
            print('smart algorithm cutoff: ', alpha)
            print('block size: ', block_size)
            print('grad reps: ', grad_reps)
            print('fubini reps: ', fb_reps)
            print('Begin run {}/{} for rotation {}'.format(_+1, runs, angle))
            print('-----------------------------------')
         
    
            for i in range(iters):
                
                fb, fb1, fb2, diag = opt.linear_time_fubini(params, blockwise=blockwise, block_size=block_size)
 
                for num in range(num_layers):
                    curr = fb[num*p:(num+1)*p, num*p:(num+1)*p] 
                    eigs = np.linalg.eigvalsh(curr)
                    min_eig = eigs.min()
                    print("In Run {}, Block {}, Min Eigval: {}".format(_, num, min_eig))

                    fb_eigs.append(min_eig)


            np.save("eig_exps/{}_qubits_{}_reps_min_eigs".format(n, reps), np.array(fb_eigs))



