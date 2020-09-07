from optim import Optimizer, get_derivative_insertion
from physics import Hamiltonian
from utils import Circuit, Gate
import numpy as np
from utils import adj, plot_fubini, get_rotations, rc, get_unitaries
from qiskit import Aer, execute, QuantumCircuit



def get_optimal_params(params, opt, H):
    score = 1.
    reg_param = 1e-2
    lr = 0.1
    i = 0

    while score > 0.006:
        grad = opt.linear_time_grad(H, params)
        
        fb, fb1, fb2 = opt.linear_time_fubini(params, blockwise=False)
        fb = fb + np.identity(len(grad)) * reg_param
        fb = np.linalg.inv(fb)
        grad = np.matmul(fb, np.array(grad))
    
        params = list(np.array(params) - lr * grad)
        score = opt.eval_params(H, params)
        print("Initial optimization: iter {}; Score {}".format(i+1, score))
        i += 1
    
    print("optimal parameters found!")
    return params, score



deviations = [0.02*np.pi*(i+1) for i in range(5)]
opt_config = ['vanilla', 'fubini', 'blockwise', 'diagonal']


reps = 0
grad_reps = reps
fb_reps = reps
n = 8
p = 6
iters = 100
rot = False
blockwise = False
diagonal_only = False
fubini = True
runs = 3
reg_param = 1e-2
lr = 0.1

init = get_rotations(n, axis='y')
exit = get_rotations(n, axis='identity')
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

params = [0.0 for i in range(len(c.params))]
opt_params = [0.0 for i in range(len(c.params))]
opt_params, opt_score = get_optimal_params(params, opt, H)


for deviation in deviations:
    for config in opt_config:

        if config == 'vanilla':
            fubini = False
            blockwise = False   
            diagonal_only = False
        elif config == 'fb':
            fubini = True
            blockwise = False   
            diagonal_only = False
        elif config == 'blockwise':
            fubini = True
            blockwise = True
            diagonal_only = False
        elif config == 'diagonal':
            fubini = True
            blockwise = False
            diagonal_only = True
 
        scores = []
        norms_pre = []
        norms_post = []

        for _ in range(runs):
            
            """perturb_params"""
            params = [opt_params[i] + np.random.normal(scale=deviation) for i in range(len(params))]
    
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
            print('deviation: ', deviation)
            print('Begin run {}/{}'.format(_+1, runs))
            print('-----------------------------------')
            score_data = []
            norm_pre_data = []
            norm_post_data = []
         
            score = opt.eval_params(H, params)
            print("Initial Score {}".format(score-opt_score))
    
            for i in range(iters):
                grad = opt.linear_time_grad(H, params)
                grad_norm = np.linalg.norm(np.array(grad))   
                norm_pre_data.append(grad_norm)
                if fubini:
                    fb, fb1, fb2 = opt.linear_time_fubini(params, blockwise=blockwise, diagonal_only=diagonal_only)
                    fb = fb + np.identity(len(grad)) * reg_param
                    fb = np.linalg.inv(fb)
                    grad = np.matmul(fb, np.array(grad))
                if not fubini:
                    grad = np.array(grad)
                params = list(np.array(params) - lr * grad)
                grad_norm = np.linalg.norm(grad)   
 
                score = opt.eval_params(H, params)-opt_score
                print("Iteration {}/{}; Score {}; Grad Norm {}".format(i+1, iters, score, grad_norm))
    
                score_data.append(score)
                norm_post_data.append(grad_norm)


            scores += [np.array(score_data)]
            norms_pre += [np.array(norm_pre_data)]
            norms_post += [np.array(norm_post_data)]
    
            path = 'saves/local_approx/'
            filename = 'tfi_local_approx_deviation_'+str(round(deviation,3))+str(n)+'_qubits_'+str(p)+'_layers_exact_grad_fb_reg_param_'+str(reg_param)+'_fubini_'+str(fubini)+'_blockwise_'+str(blockwise)+'_diagonal_only_'+str(diagonal_only)+'_'

            np.save(path+filename+'scores', np.array(scores))
            np.save(path+filename+'grad_pre_norm', np.array(norms_pre))
            np.save(path+filename+'grad_post_norm', np.array(norms_post))




