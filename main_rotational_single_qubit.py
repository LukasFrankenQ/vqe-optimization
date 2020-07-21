from qiskit import Aer, execute
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from optim import Optimizer
from physics import Hamiltonian
from utils import Circuit, plot_fubini, plot_score

"""
Code to run experiments on the Natural Gradient - optimized
Gradient for the trivial problem of product state Qubits
"""


def main(max_iter, phi=0., hamiltonian_type='rot_single_qubit_z'):

    n = 4
    lr = 0.1
    max_iter = max_iter
    grad_reps = 100
    fubini_frequency = 20
    fubini_reps = 5000
    """hamiltonian rotation"""
    phi = phi
    
    """Tikhonov regularization parameter"""
    reg_param = 1e-4

    circuit = Circuit(n)

    """QAOA layers, i.e. Trotter-Suzuki steps"""  
    p = 1
    
    circuit.add_layer('h', 'none')
    for _ in range(p):
        circuit.add_layer('y', 'ind')
        circuit.add_layer('z', 'ind')
    
    params = [0. for _ in range(len(circuit.params))]
    circuit.params = params
    print(circuit.to_qiskit())

    """available: sk, single_qubit_z, rot_single_qubit_z, transverse_ising, spin_chain"""
    H = Hamiltonian(n, hamiltonian_type=hamiltonian_type, phi=phi)
    opt = Optimizer(circuit=circuit, fubini_reps=fubini_reps, grad_reps=grad_reps)

    score_data = []
    """parallel and orthogonal norm data"""
    old_grad_norm_data = []
    new_grad_norm_data = []
    p_norm_data = []
    o_norm_data = []

    for i in range(1,max_iter+1):
        
        grad = opt.get_gradient(H, params)
        
        fb1 = opt.get_first_fubini_term(params, draw=False)
        fb2 = opt.get_second_fubini_term(params, draw=False)
        fb = (fb1 - fb2) + np.identity(len(grad)) * reg_param
        
        """invert and apply"""
        fb = np.linalg.inv(fb)
        old_gradient = np.array(grad)
        new_gradient = np.matmul(fb, old_gradient)
        
        """determine length of component parallel and orthogonal gradient"""
        old = old_gradient
        new = new_gradient
        parallel = old * np.dot(old, new) / np.linalg.norm(old)
        orthogonal = new - parallel
        
        old_grad_norm_data.append(np.linalg.norm(old))
        new_grad_norm_data.append(np.linalg.norm(new))
        p_norm_data.append(np.linalg.norm(parallel))
        o_norm_data.append(np.linalg.norm(orthogonal))
        
        
        """update gradient"""
        params = list(np.array(params) - lr * new_gradient)
        
        """
        curr_circuit = circuit.to_qiskit(params=deepcopy(params))
        result = execute(curr_circuit, circuit.backend, shots=grad_reps).result().get_counts()
        score = H.eval_dict(result)
        """
        
        score = H.multiterm(circuit, params)
        
        if (i-1)%5 == 0:
            print('Iteration {}, Score: {}'.format(i, score))
        
        score_data.append(score)
        
    data = [
        np.array(score_data),
        np.array(old_grad_norm_data),
        np.array(new_grad_norm_data),
        np.array(p_norm_data),
        np.array(o_norm_data)
    ]

    return data, params
    

"""
NON ROTATED HAMILTONIAN
"""   


max_iter = 250
iters = 3
phi = 0.

names = []

scores = []
old_grad_norm_data = []
new_grad_norm_data = []
p_norm_data = []
o_norm_data = []

for l in range(iters):
    
    names += ['run '+str(l+1)]
    
    data, params = main(max_iter, phi=0.)
    
    scores += [data[0]]
    old_grad_norm_data += [data[1]]
    new_grad_norm_data += [data[2]]
    p_norm_data += [data[3]]
    o_norm_data += [data[4]]
    

df_scores = pd.DataFrame.from_dict(dict(zip(names, scores)))

df_old_grad_norm = pd.DataFrame.from_dict(dict(zip(names, old_grad_norm_data)))
df_new_grad_norm = pd.DataFrame.from_dict(dict(zip(names, new_grad_norm_data)))
df_p_norm = pd.DataFrame.from_dict(dict(zip(names, p_norm_data)))
df_o_norm = pd.DataFrame.from_dict(dict(zip(names, o_norm_data)))


f, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
sns.despine(left=True)
sns.set_style('darkgrid')

sns.lineplot(data=df_scores, ax=axs[0]).set_title('scores')
sns.lineplot(data=df_old_grad_norm, ax=axs[1], legend=False).set_title('old norm')
sns.lineplot(data=df_new_grad_norm, ax=axs[2], legend=False).set_title('new norm')
sns.lineplot(data=df_p_norm, ax=axs[3], legend=False).set_title('new parallel norm')
sns.lineplot(data=df_o_norm, ax=axs[4], legend=False).set_title('new orthogonal norm')


#axs[2].set_ylim([0., 5.])
#axs[3].set_ylim([0., 5.])
#axs[4].set_ylim([0., 5.])

plt.tight_layout()
plt.savefig('saves/single_qubit_orientational_zero_init_no_rotation.png', dpi=400)
plt.show()


"""
ROTATED HAMILTONIAN
"""

max_iter = 250
iters = 3
phi = np.pi/4.

names = []

scores = []
old_grad_norm_data = []
new_grad_norm_data = []
p_norm_data = []
o_norm_data = []

for l in range(iters):
    
    names += ['run '+str(l+1)]
    
    data, params = main(max_iter, phi=phi)
    
    scores += [data[0]]
    old_grad_norm_data += [data[1]]
    new_grad_norm_data += [data[2]]
    p_norm_data += [data[3]]
    o_norm_data += [data[4]]


df_scores = pd.DataFrame.from_dict(dict(zip(names, scores)))

df_old_grad_norm = pd.DataFrame.from_dict(dict(zip(names, old_grad_norm_data)))
df_new_grad_norm = pd.DataFrame.from_dict(dict(zip(names, new_grad_norm_data)))
df_p_norm = pd.DataFrame.from_dict(dict(zip(names, p_norm_data)))
df_o_norm = pd.DataFrame.from_dict(dict(zip(names, o_norm_data)))

f, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
sns.despine(left=True)
sns.set_style('darkgrid')

sns.lineplot(data=df_scores, ax=axs[0]).set_title('scores')
sns.lineplot(data=df_old_grad_norm, ax=axs[1], legend=False).set_title('old norm')
sns.lineplot(data=df_new_grad_norm, ax=axs[2], legend=False).set_title('new norm')
sns.lineplot(data=df_p_norm, ax=axs[3], legend=False).set_title('new parallel norm')
sns.lineplot(data=df_o_norm, ax=axs[4], legend=False).set_title('new orthogonal norm')

#axs[2].set_ylim([0., 5.])
#axs[3].set_ylim([0., 5.])
#axs[4].set_ylim([0., 5.])

plt.tight_layout()
plt.savefig('saves/single_qubit_orientational_zero_init_45_degrees_rotation.png', dpi=400)
plt.show()
