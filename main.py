from optim import Optimizer, get_derivative_insertion
from physics import Hamiltonian
from utils import Circuit, Gate
import numpy as np
from utils import adj, plot_fubini, get_rotations, rc, get_unitaries
from qiskit import Aer, execute, QuantumCircuit



max_iter = 5
angles = [np.pi/4. * i / (max_iter-1) for i in range(max_iter)]


for angle in angles:

    n = 8
    p = 6
    iters = 300
    rot = True
    blockwise = False
    fubini = False
    reps = 1000
    runs = 3
    reg_param = 1e-3
    lr = 0.1
    init_noise = 0.0

    init = get_rotations(n, axis='y')
    exit = get_rotations(n, axis='identity')
    if rot:
        init = np.matmul(get_rotations(n, "gradual_rot", angle=angle), init)
        exit = get_rotations(n, axis='gradual_rot_inv', angle=angle)

    c = Circuit(n)
    for _ in range(p):
        c.add_layer('x', "ind")
        c.add_layer('y', "ind")
        c.add_layer('z', "ind")
        c.add_layer('zz', "coll")


    """sim reps > 0 yield simulated shot noise equal to noise after reps number of measurements"""
    opt = Optimizer(n=n, circuit=c, init=init, exit=exit)
    H = Hamiltonian(n, hamiltonian_type='transverse_ising', t=1., sim_reps=reps)

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
        print('rot: ', rot)
        print('simulated reps: ', reps)
        print('Begin run {}/{} for angle {} pi'.format(_+1, runs, angle/np.pi))
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
                fb, fb1, fb2 = opt.linear_time_fubini(params, blockwise=blockwise)
                fb = fb + np.identity(len(grad)) * reg_param
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

        np.save('saves/gradual_rotation/tfi_'+str(n)+'_qubits_'+str(p)+'_layers_fubini_'+str(fubini)+"_rot_"+str(rot)+'_angle_'+str(round(angle,3))+'_sim_reps_'+str(reps)+'_blockwise_'+str(blockwise)+'_scores', np.array(score_data))
        np.save('saves/gradual_rotation/tfi_'+str(n)+'_qubits_'+str(p)+'_layers_fubini_'+str(fubini)+"_rot_"+str(rot)+'_angle_'+str(round(angle,3))+'_sim_reps_'+str(reps)+'_blockwise_'+str(blockwise)+'_grad_pre_norm', np.array(norm_pre_data))
        np.save('saves/gradual_rotation/tfi_'+str(n)+'_qubits_'+str(p)+'_layers_fubini_'+str(fubini)+"_rot_"+str(rot)+'_angle_'+str(round(angle,3))+'_sim_reps_'+str(reps)+'_blockwise_'+str(blockwise)+'_grad_post_norm', np.array(norm_post_data))




