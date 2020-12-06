from optim import Optimizer, get_derivative_insertion
from physics import Hamiltonian
from utils import Circuit, Gate
import numpy as np
from utils import adj, plot_fubini, get_rotations, rc, get_unitaries
from qiskit import Aer, execute, QuantumCircuit



reg_params = [0.05, 0.04, 0.03, 0.02, 0.01]
block_sizes = [25]

for reg_param in reg_params:
    for block_size in block_sizes:

        reps = 100000
        grad_reps = reps
        fb_reps = reps
        alpha = reg_param
        alpha = None
        n = 8
        p = 6
        iters = 200
        angle = '45'
        rot = angle
        blockwise = True
        diagonal_only = False
        one_block = False
        fubini = True
        
        runs = 3
        if reps == 0:
            runs = 1
        
        lr = 0.1
        init_noise = 0.

        init = get_rotations(n, axis='y')
        exit = get_rotations(n, axis='identity')
    
        if angle == 'identity':
            pass
        elif angle == '45':
            init = np.matmul(get_rotations(n, "45"), init)
            exit = get_rotations(n, axis='45_inv')

        c = Circuit(n)
        for _ in range(p):
            c.add_layer('x', "ind")
            c.add_layer('y', "ind")
            c.add_layer('z', "ind")
            c.add_layer('zz', "coll")


        """sim reps > 0 yield simulated shot noise equal to noise after reps number of measurements"""
        opt = Optimizer(n=n, circuit=c, init=init, exit=exit, sim_reps=fb_reps)
        H = Hamiltonian(n, hamiltonian_type='transverse_ising', t=1., sim_reps=grad_reps)

        scores = []
        norms_pre = []
        norms_post = []
        fb_diag = []

        for _ in range(runs):
        
            params = [np.random.normal(scale=init_noise) for i in range(len(c.params))]
            print('-----------------------------------')
            print('n: ', n)
            print('p: ', p)
            print('fubini: ', fubini)
            print('blockwise: ', blockwise)
            print('smart algorithm cutoff: ', alpha)
            print('block size: ', block_size)
            print('grad reps: ', grad_reps)
            print('fubini reps: ', fb_reps)
            print('reg_param: ', reg_param)
            print('Begin run {}/{} for rotation {}'.format(_+1, runs, angle))
            print('-----------------------------------')
            score_data = []
            norm_pre_data = []
            norm_post_data = []
            fb_diagonal_data = []
         
            score = opt.eval_params(H, params)
            print("Initial Score {}".format(score))
    
            for i in range(iters):
                grad = opt.linear_time_grad(H, params)
                grad_norm = np.linalg.norm(np.array(grad))   
                norm_pre_data.append(grad_norm)
                
                fb, fb1, fb2, diag = opt.linear_time_fubini(params, blockwise=blockwise, block_size=block_size, smart=alpha, no_commuters=True)
                fb = fb + np.identity(len(grad)) * reg_param
                fb = np.linalg.inv(fb)
                grad = np.matmul(fb, np.array(grad))
                params = list(np.array(params) - lr * grad)
                grad_norm = np.linalg.norm(grad)   
 
                score = opt.eval_params(H, params)
                print("Iteration {}/{}; Score {}; Grad Norm pre vs post {} vs {}".format(i+1, iters, score, norm_pre_data[-1], grad_norm))
                

                score_data.append(score)
                fb_diagonal_data.append(fb)
                norm_post_data.append(grad_norm)
 

            scores += [np.array(score_data)]
            norms_pre += [np.array(norm_pre_data)]
            norms_post += [np.array(norm_post_data)]
            fb_diag += [np.array(fb_diagonal_data)] 

            path = 'commuting_algorithm_saves/'
            filename = 'tfi_'+str(n)+'_qubits_'+str(p)+"_layers_reps_"+str(reps)+'_saved_full_metric_reg_param_'+str(reg_param)+'_blocksize_'+str(block_size)+'_'

            np.save(path+filename+'scores', np.array(scores))
            np.save(path+filename+'grad_pre_norm', np.array(norms_pre))
            np.save(path+filename+'grad_post_norm', np.array(norms_post))
            #np.save(path+filename+'fb', np.array(fb_diag))




