from qiskit import Aer, execute, QuantumCircuit
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from optim import Optimizer
from physics import Hamiltonian
from utils import Circuit, plot_fubini, plot_score


#######################################################
##  (NON) ROTATED HAMILTONIAN WITH NATURAL GRADIENT
######################################################



def main(max_iter, phi=0., hamiltonian_type=None, p=5, n=4, start_in_x=True, rot_H=False, random_matrix=False):

    hamiltonian_type = "transverse_ising"
    n = n
    lr = 0.01
    max_iter = max_iter
    grad_reps = 10000
    fubini_frequency = 20
    fubini_reps = 100
    #external field
    t = 0.0
    random_matrix = random_matrix

    #Tikhonov regularization parameter
    reg_param = 1e-2

    circuit = Circuit(n)

    #QAOA layers, i.e. Trotter-Suzuki steps  
    p = p
    

    if rot_H:
        start_in_x = False
        init_circuit = QuantumCircuit(n, n)
        append_circuit = QuantumCircuit(n, n)
        for qubit in range(n):
            init_circuit.h(qubit)
            init_circuit.rz(np.pi/4., qubit)
            init_circuit.ry(np.pi/4., qubit)
            append_circuit.rz(-np.pi/4., qubit)
            append_circuit.ry(-np.pi/4., qubit)
        append_circuit.measure([i for i in range(n)], [i for i in range(n)])
        rot_circuit = True
    else:
        rot_circuit = False
        init_circuit = None
        append_circuit = None


    if start_in_x:
        circuit.add_layer('h', 'none')

    for _ in range(p):
        circuit.add_layer('zz', 'coll')
        circuit.add_layer('x', 'ind')
        circuit.add_layer('y', 'ind')
        circuit.add_layer('z', 'ind')



    params = [0. for _ in range(len(circuit.params))]
    circuit.params = params
    #params = circuit.params
    #print(circuit.to_qiskit())

    #available: sk, single_qubit_z, rot_single_qubit_z, transverse_ising, spin_chain
    H = Hamiltonian(n, hamiltonian_type=hamiltonian_type, t=t, 
                    init_circuit=init_circuit,
                    append_circuit=append_circuit)
    opt = Optimizer(circuit=circuit, fubini_reps=fubini_reps, grad_reps=grad_reps, rot_circuit=rot_circuit)

    score_data = []
    grad_norm_data = []
    #parallel and orthogonal norm data
    old_grad_norm_data = []
    new_grad_norm_data = []
    p_norm_data = []
    o_norm_data = []

    increase_reps = 0

    for i in range(1,max_iter+1):


        grad = opt.get_gradient(H, params)
        
        if not random_matrix:
            # get simulated fb metric
            fb, fb1, fb2 = opt.simulate_fubini_metric(params)
        elif random_matrix:
            mu = 0.03
            var = 0.25
            fb = np.random.normal(mu, var, (len(params), len(params)))
        
        
        # regularize
        fb = fb + np.identity(len(grad)) * reg_param


        #invert and apply
        fb = np.linalg.inv(fb)
        old_gradient = np.array(grad)
        new_gradient = np.matmul(fb, old_gradient)

        #determine length of component parallel and orthogonal gradient
        old = old_gradient
        new = new_gradient
        parallel = old * np.dot(old, new) / np.linalg.norm(old)
        orthogonal = new - parallel

        old_grad_norm_data.append(np.linalg.norm(old))
        new_grad_norm_data.append(np.linalg.norm(new))
        p_norm_data.append(np.linalg.norm(parallel))
        o_norm_data.append(np.linalg.norm(orthogonal))


        #update gradient
        params = list(np.array(params) - lr * new_gradient)
        #params = list(np.array(params) - lr * np.array(grad))
        #grad_norm_data.append(np.linalg.norm(np.array(grad)))

        #print('/n ------- current gradient -------------')
        #print(old_gradient)
        #print(new_gradient)
        #print('/n ------- current gradient -------------')

        #curr_circuit = circuit.to_qiskit(params=deepcopy(params))
        #result = execute(curr_circuit, circuit.backend, shots=grad_reps).result().get_counts()
        #score = H.eval_dict(result)

        score = H.multiterm(circuit, params, reps=10000)

        if (i-1)%1 == 0:
            print('Iteration {}, Score: {}'.format(i, score))

        score_data.append(score)

        # increase algorithm precision at certain performance
        if score < 0.25 and increase_reps == 0:
            opt.fubini_reps = opt.fubini_reps * 2
            opt.grad_reps = opt.grad_reps * 2
            lr = lr * 0.5
            increase_reps = 1
        elif score < 0.15 and increase_reps == 1:
            opt.fubini_reps = opt.fubini_reps * 2
            opt.grad_reps = opt.grad_reps * 2
            lr = lr * 0.5
            increase_reps = 2
        elif score < 0.05 and increase_reps == 2:
            opt.fubini_reps = opt.fubini_reps * 2
            opt.grad_reps = opt.grad_reps * 2
            lr = lr * 0.5
            increase_reps = 3

    data = [
        np.array(score_data),
        #np.array(grad_norm_data)
        np.array(old_grad_norm_data),
        np.array(new_grad_norm_data),
        np.array(p_norm_data),
        np.array(o_norm_data)
    ]

    return data, params




max_iter = 500
iters = 5
phi = 0.
n = 4
p = 4
rot_H = True
random_matrix = True


names = []

scores = []
#grad_norm_data = []
old_grad_norm_data = []
new_grad_norm_data = []
p_norm_data = []
o_norm_data = []

for l in range(iters):

    names += ['run '+str(l+1)]

    data, params = main(max_iter, p=p, n=n, rot_H=rot_H, random_matrix=random_matrix)

    scores += [data[0]]
    #grad_norm_data += [data[1]]
    old_grad_norm_data += [data[1]]
    new_grad_norm_data += [data[2]]
    p_norm_data += [data[3]]
    o_norm_data += [data[4]]

    np.save('saves/SIM_TFI_with_rot_ng_with_random_matrix_scores', np.array(scores))
    np.save('saves/SIM_TFI_with_rot_ng_with_random_matrix_old_grad_norm', np.array(old_grad_norm_data))
    np.save('saves/SIM_TFI_with_rot_ng_with_random_matrix_new_grad_norm', np.array(new_grad_norm_data))
    np.save('saves/SIM_TFI_with_rot_ng_with_random_matrix_parallel_norm', np.array(p_norm_data))
    np.save('saves/SIM_TFI_with_rot_ng_with_random_matrix_orthogonal_norm', np.array(o_norm_data))

df_scores = pd.DataFrame.from_dict(dict(zip(names, scores)))
#df_grad_norm = pd.DataFrame.from_dict(dict(zip(names, grad_norm_data)))
df_old_grad_norm = pd.DataFrame.from_dict(dict(zip(names, old_grad_norm_data)))
df_new_grad_norm = pd.DataFrame.from_dict(dict(zip(names, new_grad_norm_data)))
df_p_norm = pd.DataFrame.from_dict(dict(zip(names, p_norm_data)))
df_o_norm = pd.DataFrame.from_dict(dict(zip(names, o_norm_data)))

f, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
sns.despine(left=True)
sns.set_style('darkgrid')

sns.lineplot(data=df_scores, ax=axs[0]).set_title('Scores')
sns.lineplot(data=df_old_grad_norm, ax=axs[1], legend=False).set_title('Old Gradient Norm')
sns.lineplot(data=df_new_grad_norm, ax=axs[2], legend=False).set_title('New Gradient Norm')
sns.lineplot(data=df_p_norm, ax=axs[3], legend=False).set_title('New Parallel Norm')
sns.lineplot(data=df_o_norm, ax=axs[4], legend=False).set_title('New Orthogonal Norm')

#axs[2].set_ylim([0., 5.])
#axs[3].set_ylim([0., 5.])
#axs[4].set_ylim([0., 5.])

plt.tight_layout()
plt.savefig('saves/tfi_t_0_rot_True_4_qubits_4_layers_ng_with_random_matrix.png', dpi=400)



