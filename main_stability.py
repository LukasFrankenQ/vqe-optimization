from optim import Optimizer, get_derivative_insertion
from physics import Hamiltonian
from utils import Circuit, Gate
import numpy as np
from utils import adj, plot_fubini, get_rotations, rc, get_unitaries
from qiskit import Aer, execute, QuantumCircuit


def smallest(liste, n=3):
    mins = []
    liste.sort(reverse=False)
    for i in range(n):
        mins.append(round(liste[i], 4))
    return mins


n = 8
p = 6

angle = 'identity'

block_sizes = [25]
rep_list = [0, 10000, 1000, 100]

for reps in rep_list:
    for block_size in block_sizes:

        grad_reps = 0
        fb_reps = reps

        n = 8
        p = 6
        iters = 300
        rot = angle
        blockwise = True
        diagonal_only = False
        one_block = True
        fubini = True
        get_proxis = True
        runs = 3
        if reps == 0:
            runs = 1
        reg_param = 0.05
        lr = 0.1
        init_noise = 0.0

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
        proxis_data = []

        for _ in range(runs):
        
            params = [np.random.normal(scale=init_noise) for i in range(len(c.params))]
            print('-----------------------------------')
            print('n: ', n)
            print('p: ', p)
            print('fubini: ', fubini)
            print('blockwise: ', blockwise)
            print('one block: ', one_block)
            print('rot: ', rot)
            print('block size: ', block_size)
            print('grad reps: ', grad_reps)
            print('fubini reps: ', fb_reps)
            print('reg_param: ', reg_param)
            print('getting proxis: ', get_proxis)
            print('Begin run {}/{} for rotation {}'.format(_+1, runs, angle))
            print('-----------------------------------')
            score_data = []
            norm_pre_data = []
            norm_post_data = []
         
            score = opt.eval_params(H, params)
            print("Initial Score {}".format(score))
    
            for i in range(iters):
                grad = opt.linear_time_grad(H, params)
                grad_norm = np.linalg.norm(np.array(grad))   
                norm_pre_data.append(grad_norm)
                if fubini:
                    fb, fb1, fb2, proxis = opt.linear_time_fubini(params, blockwise=blockwise, block_size=block_size, get_proxis=get_proxis, one_block=one_block)
                    fb = fb + np.identity(len(grad)) * reg_param
                    fb = np.linalg.inv(fb)
                    grad = np.matmul(fb, np.array(grad))
                if not fubini:
                    grad = np.array(grad)
                params = list(np.array(params) - lr * grad)
                grad_norm = np.linalg.norm(grad)   
 
                score = opt.eval_params(H, params)
                print("Iteration {}/{}; Score {}; Grad Norm pre vs post {} vs {}".format(i+1, iters, score, norm_pre_data[-1], grad_norm))

                score_data.append(score)
                norm_post_data.append(grad_norm)
                proxis_data.append(proxis)

            scores += [np.array(score_data)]
            norms_pre += [np.array(norm_pre_data)]
            norms_post += [np.array(norm_post_data)]

            path = 'saves/stability/'
            filename = 'tfi_'+str(n)+'_qubits_'+str(p)+'_layers_fubini_'+str(fubini)+"_reps_"+str(reps)+'_reg_param_'+str(reg_param)+'_one_block_for_full_fubini_'+str(one_block)+'_'

            np.save(path+filename+'scores', np.array(scores))
            np.save(path+filename+'grad_pre_norm', np.array(norms_pre))
            np.save(path+filename+'grad_post_norm', np.array(norms_post))
            np.save(path+filename+'proxis', np.array(proxis_data))




