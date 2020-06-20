from qiskit import Aer, execute
from copy import deepcopy
import numpy as np

#from optim import Optimizer
#from physics import Hamiltonian
from utils import Circuit, plot_fubini

def main():
    """
    This proof on concept program creates a simplistic VQE circuit architecture and evaluates 
    the fubini-study metric during the parameter optimization. 
    As expected the optimization rotates the qubits into an eigenstate of the Hamiltonian terms,
    such that the part of the fb-metric circuit's generators, which are featured in this Hamiltonian 
    are converging to a rank 1 matrix. 
    
    More sophistaced examples follow soon. 
    
    The optimization method is vanilla gradient descent
    https://arxiv.org/abs/1905.13311
    To evaluate the metric we use the method suggested in (so far only the first term is evaluated)
    Phys. Rev. X, 7(2):021050
    """

    n = 4
    lr = 0.4
    max_iter = 100
    grad_reps = 100
    fubini_frequency = 20

    circuit = Circuit(n)

    circuit.add_layer('h', 'none')
    circuit.add_layer('y', 'ind')
    circuit.add_layer('z', 'ind')
    
    print(circuit.to_qiskit())
    
    params = circuit.params

    H = Hamiltonian(n, hamiltonian_type='single_qubit_z')
    opt = Optimizer(circuit=circuit)

    score_data = []
    fubini_matrices = []
    fubini_iterations = []

    for i in range(1,max_iter+1):
        grad = opt.get_gradient(H.eval_dict, params)
        params = list(np.array(params) - np.array(grad))
        
        curr_circuit = circuit.to_qiskit(params=deepcopy(params))
        result = execute(curr_circuit, circuit.backend, shots=grad_reps).result().get_counts()
        score = H.eval_dict(result)
    
        if i%5 == 0:
            print('Iteration {}, Score: {}'.format(i, score))
        if (i-1)%fubini_frequency == 0:
            fb = opt.get_first_fubini_term(params, draw=False) - opt.get_second_fubini_term(params, draw=False)
            fubini_matrices.append(fb)
            fubini_iterations.append(i)

        score_data += [score]

    plot_fubini(fubini_matrices, fubini_iterations)
    
    
if __name__ == "__main__":
    main()