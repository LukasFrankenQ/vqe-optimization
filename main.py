from qiskit import Aer, execute
from copy import deepcopy
import numpy as np

from optim import Optimizer
from physics import Hamiltonian
from utils import Circuit, plot_fubini

def main():
    save_id = str(rd.randint(0, 1e6))
    """
    This proof of concept program creates a simplistic VQE circuit architecture and evaluates 
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

    n = 8
    lr = 0.4
    max_iter = 100
    grad_reps = 100
    fubini_frequency = 5

    circuit = Circuit(n)

    """QAOA layers"""  
    p = 3  
    
    circuit.add_layer('h', 'none')
    
    for _ in range(p):
        circuit.add_layer('x', 'coll')
        circuit.add_layer('zz', 'ind')
    
    print(circuit.to_qiskit())
    
    params = circuit.params

    """available: sk, single_qubit_z, transverse_ising"""
    H = Hamiltonian(n, hamiltonian_type='transverse_ising')
    opt = Optimizer(circuit=circuit)

    score_data = []
    fubini_terms_one = []
    fubini_terms_two = []
    fubini_matrices = []
    fubini_iterations = []

    for i in range(1,max_iter+1):
        grad = opt.get_gradient(H, params)
        params = list(np.array(params) - np.array(grad))
        
        """
        curr_circuit = circuit.to_qiskit(params=deepcopy(params))
        result = execute(curr_circuit, circuit.backend, shots=grad_reps).result().get_counts()
        score = H.eval_dict(result)
        """
        
        score = H.multiterm(circuit, params)
    
        if i%5 == 0:
            print('Iteration {}, Score: {}'.format(i, score))
        if (i-1)%fubini_frequency == 0:
            fb1 = opt.get_first_fubini_term(params, draw=False)
            fb2 = opt.get_second_fubini_term(params, draw=False)
            fubini_terms_one.append(fb1)
            fubini_terms_two.append(fb2)
            fubini_matrices.append(fb2 - fb1)
            fubini_iterations.append(i)

        score_data += [score]

    print('\n term one:')
    plot_fubini(fubini_terms_one, fubini_iterations, savename='term_one'+save_id)
    print('term two:')
    plot_fubini(fubini_terms_two, fubini_iterations, savename='term_two'+save_id)
    print('full fubini:')
    plot_fubini(fubini_matrices, fubini_iterations, savename='both_terms'+save_id)
    
    
if __name__ == "__main__":
    main()