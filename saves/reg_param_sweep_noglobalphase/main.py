from optim import Optimizer, get_derivative_insertion
from physics import Hamiltonian
from utils import Circuit, Gate
import numpy as np
from utils import adj, plot_fubini, get_rotations, rc, get_unitaries
from qiskit import Aer, execute, QuantumCircuit



angle = 'identity'
reg_params = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 1.]
"""recreate conditions of previous experiment"""

reps = 0

for reg_param in reg_params:

    grad_reps = reps
    fb_reps = reps

    n = 8
    p = 6
    iters = 300
    rot = angle
    blockwise = False
    diagonal_only = False
    fubini = True
    runs = 1
    #reg_param = 1e-2
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

    for _ in range(runs):
        
        params = [np.random.normal(scale=init_noise) for i in range(len(c.params))]
        print('-----------------------------------')
        print('n: ', n)
        print('p: ', p)
        print('fubini: ', fubini)
        print('blockwise: ', blockwise)
        print('diagonal only: ', diagonal_only)
        print('rot: ', rot)
        print('grad reps: ', grad_reps)
        print('fubini reps: ', fb_reps)
        print('reg_param: ', reg_param)
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
                fb, fb1, fb2 = opt.linear_time_fubini(params, blockwise=blockwise, diagonal_only=diagonal_only)
                fb = (1.-reg_param) * fb + np.identity(len(grad)) * reg_param
                fb = np.linalg.inv(fb)
                grad = np.matmul(fb, np.array(grad))
            if not fubini:
                grad = np.array(grad)
            params = list(np.array(params) - lr * grad)
            grad_norm = np.linalg.norm(grad)   
 
            score = opt.eval_params(H, params)
            print("Iteration {}/{}; Score {}; Grad Norm {}".format(i+1, iters, score, grad_norm))

            score_data.append(score)
            norm_post_data.append(grad_norm)


        scores += [np.array(score_data)]
        norms_pre += [np.array(norm_pre_data)]
        norms_post += [np.array(norm_post_data)]

        path = 'saves/reg_param_sweep_noglobalphase/'
        filename = 'tfi_'+str(n)+'_qubits_'+str(p)+'_layers_fubini_'+str(fubini)+"_rot_"+str(rot)+'_reg_param_'+str(reg_param)+'_exact_grad_fb_'

        np.save(path+filename+'scores', np.array(scores))
        np.save(path+filename+'grad_pre_norm', np.array(norms_pre))
        np.save(path+filename+'grad_post_norm', np.array(norms_post))




