from qiskit import Aer, execute
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from optim import Optimizer
from physics import Hamiltonian
from utils import Circuit, plot_fubini, plot_score




def main(max_iter, phi=0., hamiltonian_type=None, p=5, n=4, start_in_x=True):

    hamiltonian_type = "transverse_ising"
    n = n
    lr = 0.1
    max_iter = max_iter
    grad_reps = 100
    fubini_frequency = 20
    fubini_reps = 5000
    #external field
    t = 0.0
    
    #Tikhonov regularization parameter
    reg_param = 1e-4

    circuit = Circuit(n)

    #QAOA layers, i.e. Trotter-Suzuki steps  
    p = p
    
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
    H = Hamiltonian(n, hamiltonian_type=hamiltonian_type, t=t)
    opt = Optimizer(circuit=circuit, fubini_reps=fubini_reps, grad_reps=grad_reps)

    score_data = []
    grad_norm_data = []
    #parallel and orthogonal norm data
    old_grad_norm_data = []
    new_grad_norm_data = []
    p_norm_data = []
    o_norm_data = []

    for i in range(1,max_iter+1):
        
        grad = opt.get_gradient(H, params)
        
        #fb1 = opt.get_first_fubini_term(params, draw=False)
        #fb2 = opt.get_second_fubini_term(params, draw=False)
        #fb = (fb1 - fb2) + np.identity(len(grad)) * reg_param
        
        #invert and apply
        #fb = np.linalg.inv(fb)
        #old_gradient = np.array(grad)
        #new_gradient = np.matmul(fb, old_gradient)
        
        #determine length of component parallel and orthogonal gradient
        #old = old_gradient
        #new = new_gradient
        #parallel = old * np.dot(old, new) / np.linalg.norm(old)
        #orthogonal = new - parallel
        
        #old_grad_norm_data.append(np.linalg.norm(old))
        #new_grad_norm_data.append(np.linalg.norm(new))
        #p_norm_data.append(np.linalg.norm(parallel))
        #o_norm_data.append(np.linalg.norm(orthogonal))
        
        
        #update gradient
        #params = list(np.array(params) - lr * new_gradient)
        params = list(np.array(params) - lr * np.array(grad))
        grad_norm_data.append(np.linalg.norm(np.array(grad)))
        
        
        #curr_circuit = circuit.to_qiskit(params=deepcopy(params))
        #result = execute(curr_circuit, circuit.backend, shots=grad_reps).result().get_counts()
        #score = H.eval_dict(result)
    
        
        score = H.multiterm(circuit, params)
        
        if (i-1)%5 == 0:
            print('Iteration {}, Score: {}'.format(i, score))
        
        score_data.append(score)
        
    data = [
        np.array(score_data),
        np.array(grad_norm_data)
        #np.array(old_grad_norm_data),
        #np.array(new_grad_norm_data),
        #np.array(p_norm_data),
        #np.array(o_norm_data)
    ]

    return data, params

##############################
#NO ROTATION - VANILLA GRADIENT
##########################333

max_iter = 500
iters = 5
phi = 0.
n = 6
p = 3

names = []

scores = []
grad_norm_data = []

#old_grad_norm_data = []
#new_grad_norm_data = []
#p_norm_data = []
#o_norm_data = []

for l in range(iters):
    
    names += ['run '+str(l+1)]
    
    data, params = main(max_iter, p=p, n=n)
    
    scores += [data[0]]
    grad_norm_data += [data[1]]
    #old_grad_norm_data += [data[1]]
    #new_grad_norm_data += [data[2]]
    #p_norm_data += [data[3]]
    #o_norm_data += [data[4]]


df_scores = pd.DataFrame.from_dict(dict(zip(names, scores)))
df_grad_norm = pd.DataFrame.from_dict(dict(zip(names, grad_norm_data)))

#df_old_grad_norm = pd.DataFrame.from_dict(dict(zip(names, old_grad_norm_data)))
#df_new_grad_norm = pd.DataFrame.from_dict(dict(zip(names, new_grad_norm_data)))
#df_p_norm = pd.DataFrame.from_dict(dict(zip(names, p_norm_data)))
#df_o_norm = pd.DataFrame.from_dict(dict(zip(names, o_norm_data)))

f, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
sns.despine(left=True)
sns.set_style('darkgrid')

sns.lineplot(data=df_scores, ax=axs[0]).set_title('Scores')
sns.lineplot(data=df_grad_norm, ax=axs[1], legend=False).set_title('Gradient Norm')
#sns.lineplot(data=df_new_grad_norm, ax=axs[2], legend=False).set_title('new norm')
#sns.lineplot(data=df_p_norm, ax=axs[3], legend=False).set_title('new parallel norm')
#sns.lineplot(data=df_o_norm, ax=axs[4], legend=False).set_title('new orthogonal norm')

#axs[2].set_ylim([0., 5.])
#axs[3].set_ylim([0., 5.])
#axs[4].set_ylim([0., 5.])

plt.tight_layout()
plt.savefig('saves/transverse_ising_t_0_0_init_0_degrees_rotation_no_ng.png', dpi=400)
plt.show()

"""



###############################################################################
# ROTATION - VANILLA GRADIENT
###############################################################################3

from qiskit import Aer, execute, QuantumCircuit
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from optim import Optimizer
from physics import Hamiltonian
from utils import Circuit, plot_fubini, plot_score

def main(max_iter, phi=0., hamiltonian_type='rot_single_qubit_z', p=5, n=4, start_in_x=True):

    hamiltonian_type = "transverse_ising"
    n = n
    lr = 0.1
    max_iter = max_iter
    grad_reps = 100
    fubini_frequency = 20
    fubini_reps = 5000
    #external field
    t = 0.0
    
    #define circuit rotation
    init_circuit = QuantumCircuit(n, n)
    append_circuit = QuantumCircuit(n, n)
    for qubit in range(n):
        init_circuit.h(qubit)
        init_circuit.rz(np.pi/4., qubit)
        init_circuit.ry(np.pi/4., qubit)
        append_circuit.rz(-np.pi/4., qubit)
        append_circuit.ry(-np.pi/4., qubit)
    append_circuit.measure([i for i in range(n)], [i for i in range(n)])
    
    #Tikhonov regularization parameter
    reg_param = 1e-4

    circuit = Circuit(n)

    #QAOA layers, i.e. Trotter-Suzuki steps  
    p = p
    
    #if start_in_x:
    #    circuit.add_layer('h', 'none')
    
    for _ in range(p):
        circuit.add_layer('zz', 'coll') 
        circuit.add_layer('x', 'ind')
        circuit.add_layer('y', 'ind')
        circuit.add_layer('z', 'ind')


    params = [0. for _ in range(len(circuit.params))]
    circuit.params = params
    #params = circuit.params
    print(circuit.to_qiskit())

    #available: sk, single_qubit_z, rot_single_qubit_z, transverse_ising, spin_chain
    H = Hamiltonian(n, 
                    hamiltonian_type=hamiltonian_type, 
                    t=t, 
                    init_circuit=init_circuit,
                    append_circuit=append_circuit)
    
    opt = Optimizer(circuit=circuit, 
                    fubini_reps=fubini_reps, 
                    grad_reps=grad_reps, 
                    rot_circuit=True, 
                    n=n)

    score_data = []
    grad_norm_data = []
    #parallel and orthogonal norm data
    #old_grad_norm_data = []
    #new_grad_norm_data = []
    #p_norm_data = []
    #o_norm_data = []

    for i in range(1,max_iter+1):
                     
        if i%30 == 0:
            factor = 5
            grad_reps = grad_reps * factor
            lr = lr / factor
            print("Increase grad reps to ", grad_reps)
            print("Reduce learning rate to ", lr)
        
        grad = opt.get_gradient(H, params, reps=grad_reps)
        
        #fb1 = opt.get_first_fubini_term(params, draw=False)
        #fb2 = opt.get_second_fubini_term(params, draw=False)
        
        #fb = (fb1 - fb2) + np.identity(len(grad)) * reg_param
        
        
        #invert and apply
        #fb = np.linalg.inv(fb)
        #old_gradient = np.array(grad)
        #new_gradient = old_gradient
        #new_gradient = np.matmul(fb, old_gradient)
        
        #determine length of component parallel and orthogonal gradient
        #old = old_gradient
        #new = new_gradient
        #parallel = old * np.dot(old, new) / np.linalg.norm(old)
        #orthogonal = new - parallel
        
        #old_grad_norm_data.append(np.linalg.norm(old))
        #new_grad_norm_data.append(np.linalg.norm(new))
        #p_norm_data.append(np.linalg.norm(parallel))
        #o_norm_data.append(np.linalg.norm(orthogonal))
        
        
        #update gradient
        params = list(np.array(params) - lr * np.array(grad))
        #new_grad_norm_data.append(np.linalg.norm(new_gradient))
        #old_grad_norm_data.append(np.linalg.norm(old_gradient))
        
        
        #curr_circuit = circuit.to_qiskit(params=deepcopy(params))
        #result = execute(curr_circuit, circuit.backend, shots=grad_reps).result().get_counts()
        #score = H.eval_dict(result)
        
        
        score = H.multiterm(circuit, params)
        
        if (i-1)%5 == 0:
            print('Iteration {}, Score: {}'.format(i, score))
        
        score_data.append(score)
        grad_norm_data.append(np.linalg.norm(np.array(norm)))

    data = [
        np.array(score_data),
        np.array(grad_norm_data)
        #np.array(old_grad_norm_data),
        #np.array(new_grad_norm_data),
        #np.array(p_norm_data),
        #np.array(o_norm_data)
    ]

    return data, params
    
    
max_iter = 100
iters = 3
phi = 0.
n = 6
p = n

names = []

scores = []
grad_norm = []
#old_grad_norm_data = []
#new_grad_norm_data = []
#p_norm_data = []
#o_norm_data = []


for l in range(iters):
    
    names += ['run '+str(l+1)]
    
    data, params = main(max_iter, p=p, n=n)
    
    scores += [data[0]]
    grad_norm_data += [data[1]]
    #old_grad_norm_data += [data[1]]
    #new_grad_norm_data += [data[2]]
    #p_norm_data += [data[3]]
    #o_norm_data += [data[4]]

    np.save('TFI_rot_no_ng_scores', np.array(scores))
    np.save('TFI_rot_no_ng_grad_norm', np.array(grad_norm_data))


df_scores = pd.DataFrame.from_dict(dict(zip(names, scores)))
df_grad_norm = pd.DataFrame.from_dict(dict(zip(names, grad_norm_data)))

#df_old_grad_norm = pd.DataFrame.from_dict(dict(zip(names, old_grad_norm_data)))
#df_new_grad_norm = pd.DataFrame.from_dict(dict(zip(names, new_grad_norm_data)))
#df_p_norm = pd.DataFrame.from_dict(dict(zip(names, p_norm_data)))
#df_o_norm = pd.DataFrame.from_dict(dict(zip(names, o_norm_data)))

f, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
sns.despine(left=True)
sns.set_style('darkgrid')

sns.lineplot(data=df_scores, ax=axs[0]).set_title('Scores')
sns.lineplot(data=df_grad_norm, ax=axs[1]).set_title('Gradient Norm')
#sns.lineplot(data=df_old_grad_norm, ax=axs[1], legend=False).set_title('Vanilla Gradient Norm')
#sns.lineplot(data=df_new_grad_norm, ax=axs[2], legend=False).set_title('NG Gradient Norm')
#sns.lineplot(data=df_p_norm, ax=axs[3], legend=False).set_title('NG Parallel Gradient Norm')
#sns.lineplot(data=df_o_norm, ax=axs[4], legend=False).set_title('NG Orthogonal Gradient Norm')

#axs[1].set_ylim([0., 10.])
#axs[2].set_ylim([0., 10.])
#axs[3].set_ylim([0., 10.])
#axs[4].set_ylim([0., 10.])

plt.tight_layout()
plt.savefig('saves/transverse_ising_t_0_0_init_45_degrees_rotation_without_ng.png', dpi=400)
plt.show()


"""

"""
############################################################
#ROTATED HAMILTONIAN - VANILLA GRADIENT
############################################################

max_iter = 300
iters = 3
phi = 0.
n = 6
p = n

names = []

scores = []
old_grad_norm_data = []
new_grad_norm_data = []
p_norm_data = []
o_norm_data = []


for l in range(iters):
    
    names += ['run '+str(l+1)]
    
    data, params = main(max_iter, p=p, n=n)
    
    scores += [data[0]]
    #grad_norm_data += [data[1]]
    old_grad_norm_data += [data[1]]
    #new_grad_norm_data += [data[2]]
    #p_norm_data += [data[3]]
    #o_norm_data += [data[4]]


df_scores = pd.DataFrame.from_dict(dict(zip(names, scores)))
#df_grad_norm = pd.DataFrame.from_dict(dict(zip(names, grad_norm_data)))

df_old_grad_norm = pd.DataFrame.from_dict(dict(zip(names, old_grad_norm_data)))
#df_new_grad_norm = pd.DataFrame.from_dict(dict(zip(names, new_grad_norm_data)))
#df_p_norm = pd.DataFrame.from_dict(dict(zip(names, p_norm_data)))
#df_o_norm = pd.DataFrame.from_dict(dict(zip(names, o_norm_data)))

f, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
sns.despine(left=True)
sns.set_style('darkgrid')

sns.lineplot(data=df_scores, ax=axs[0]).set_title('Scores')
sns.lineplot(data=df_old_grad_norm, ax=axs[1], legend=False).set_title('Gradient Norm')
#sns.lineplot(data=df_new_grad_norm, ax=axs[2], legend=False).set_title('new norm')
#sns.lineplot(data=df_p_norm, ax=axs[3], legend=False).set_title('new parallel norm')
#sns.lineplot(data=df_o_norm, ax=axs[4], legend=False).set_title('new orthogonal norm')

#axs[2].set_ylim([0., 5.])
#axs[3].set_ylim([0., 5.])
#axs[4].set_ylim([0., 5.])

plt.tight_layout()
plt.savefig('saves/transverse_ising_t_0_0_init_45_degrees_rotation_no_ng.png', dpi=400)
plt.show()

"""
"""
############################################################
#ROTATED HAMILTONIAN - NATURAL GRADIENT
############################################################


max_iter = 100
iters = 3
phi = 0.
n = 6
p = n

names = []

scores = []
old_grad_norm_data = []
new_grad_norm_data = []
p_norm_data = []
o_norm_data = []


for l in range(iters):
    
    names += ['run '+str(l+1)]
    
    data, params = main(max_iter, p=p, n=n)
    
    scores += [data[0]]
    #grad_norm_data += [data[1]]
    old_grad_norm_data += [data[1]]
    new_grad_norm_data += [data[2]]
    p_norm_data += [data[3]]
    o_norm_data += [data[4]]


df_scores = pd.DataFrame.from_dict(dict(zip(names, scores)))
#df_grad_norm = pd.DataFrame.from_dict(dict(zip(names, grad_norm_data)))

df_old_grad_norm = pd.DataFrame.from_dict(dict(zip(names, old_grad_norm_data)))
df_new_grad_norm = pd.DataFrame.from_dict(dict(zip(names, new_grad_norm_data)))
df_p_norm = pd.DataFrame.from_dict(dict(zip(names, p_norm_data)))
df_o_norm = pd.DataFrame.from_dict(dict(zip(names, o_norm_data)))

"""












    
