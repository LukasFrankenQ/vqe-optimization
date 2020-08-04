import numpy as np
import random as rd
from qiskit import execute, QuantumCircuit, Aer
from copy import deepcopy


class Hamiltonian:
    def __init__(self, 
                 n, 
                 hamiltonian_type='sk', 
                 phi=None, 
                 t=1.,
		 exact=True, 
                 rotate=False, 
                 init_circuit=None, 
                 append_circuit=None):
        self.n = n
        self.t = t
        self.J = 1.
        self.exact = exact
        self.hamiltonian_type = hamiltonian_type
        if exact:
            self.backend = Aer.get_backend('statevector_simulator')
            if hamiltonian_type == 'transverse_ising' or hamiltonian_type == 'spin_chain':
                self.val_vector_zz, self.val_vector_x = self.get_val_vectors()

        else:
            self.backend = Aer.get_backend('qasm_simulator')
        self.matrix = self.get_matrix()
        self.max_value, self.min_value = self.get_boundaries()
        self.phi = phi
        self.rotate = rotate
        self.init_circuit = init_circuit
        self.append_circuit = append_circuit
        
        
    def get_matrix(self):
        if self.hamiltonian_type == 'sk':
            vals = [-1., 1.]
            h = np.array([[0. for _ in range(k+1)]+[rd.choice(vals) for _ in range(self.n-k-1)] 
                          for k in range(self.n)])
            
        elif self.hamiltonian_type == 'single_qubit_z' or self.hamiltonian_type == 'rot_single_qubit_z':
            return None
        
        elif self.hamiltonian_type == 'transverse_ising' or self.hamiltonian_type == 'spin_chain':
            n = self.n
            unity = np.identity(2)
            z = np.array([[1., 0.], [0., -1.]])

            h = np.zeros((2**n, 2**n))

            for i in range(n):
                matrix = 1.
                if i == n-1:
                    matrix = np.kron(matrix, z)
                    for _ in range(i-1):
                        matrix = np.kron(matrix, unity)
                else:
                    for _ in range(i):
                        matrix = np.kron(matrix, unity)
        
                matrix = np.kron(matrix, z)
                if i < n-1:
                    matrix = np.kron(matrix, z)
                for _ in range(n-2-i):
                    matrix = np.kron(matrix, unity)
                h += matrix
                 
        return h

    
    def get_val_vectors(self):
        """get binary vectors"""
        binaries = []
        for i in range(2**self.n):
            a = format(i, 'b')
            b = self.n - np.log2(i+1)
            binary = int(b)*'0' + a 
            if len(binary) == self.n+1:
                binary = self.n * '0' 

            # order reversed due to the way qiskit orders qubits
            binary = [float(binary[j]) for j in range(self.n)][::-1]
            binaries.append(np.array(binary))

        """compute value of configurations"""
        val_vector_zz = []
        val_vector_x = []
        for config in binaries:
            config = 2. * config - 1.
            shift = np.array(list(config[1:]) + list([config[0]]))
             
            val_vector_zz.append(np.dot(config, shift) * (-self.J))
            val_vector_x.append(-self.t * np.dot(np.ones(self.n), config))
        
        return np.array(val_vector_zz), np.array(val_vector_x)


    def get_boundaries(self):
        if self.hamiltonian_type == 'sk':
            zero = '0'
            minimum = np.inf
            maximum = -np.inf
   
            for i in range(2**self.n):
                conf = bin(i)[2:]
                conf = conf + zero*(self.n-len(conf))
                conf = list(conf)
                conf = 2*np.array([float(conf[i]) for i in range(len(conf))]) - 1.
                energy = np.dot(conf, np.matmul(self.matrix, conf))
                if energy < minimum:
                    minimum = energy
                if energy > maximum:
                    maximum = energy

        elif self.hamiltonian_type == 'transverse_ising':

            if self.n%2==0:
                alphas = [np.pi / self.n*(2*i+1) for i in range(self.n//2)]
                E_gs = 0.
            else:
                alphas = [np.pi / self.n*(2*i+2) for i in range(self.n//2)]
                E_gs = - 1. - self.t 

            E_gs -= 2*sum([np.sqrt(1+2*np.cos(alphas[i])*self.t+self.t**2) for i in range(self.n//2)])
            
            """normalization is commented out"""
            """E_gs /= N"""

            maximum = (-1.)*E_gs
            minimum = E_gs
                    
        elif self.hamiltonian_type == 'single_qubit_z' or self.hamiltonian_type == 'rot_single_qubit_z':
            minimum = -1. * self.n 
            maximum = 1. * self.n 
            
        elif self.hamiltonian_type == 'spin_chain':
            minimum = -2. * self.n 
            maximum = 2. * self.n
                    
        return maximum, minimum

    
    """takes measured dictionary and return value between 0 and 1"""
    def eval_dict(self, results, hamiltonian_type=None):
        hamiltonian_type = hamiltonian_type or self.hamiltonian_type
        
        avg = 0.
        total = 0.
        for meas, count in results.items():
            meas = 2*np.array([float(meas[i]) for i in range(len(meas))]) - 1.
            
            if hamiltonian_type == 'sk' or hamiltonian_type == "transverse_ising":
                energy = np.dot(meas, np.matmul(self.matrix, meas))
            elif hamiltonian_type == 'single_qubit_z' or hamiltonian_type == 'rot_single_qubit_z':
                energy = np.dot(np.ones(self.n), meas)
                
            avg += float(count) * energy
            total += float(count)
        avg /= total
        return (self.max_value - avg) / (self.max_value - self.min_value)
    
    
    """Evaluating cost for a parameter configuration for a Hamiltonian with multiple terms"""
    def multiterm(self, circuit, params, reps=10, J=1., exact=True):

        if self.hamiltonian_type == "transverse_ising":

            energy = 0.           
            
            """compute configuration score on first term"""
            curr_circuit = circuit.to_qiskit(params=deepcopy(params), measure=False)

            if self.init_circuit is not None:
                curr_circuit = self.init_circuit + curr_circuit
            
            if not exact:

                result = execute(curr_circuit+self.append_circuit, self.backend, shots=reps).result().get_counts()

                for meas, count in result.items():
                    meas = 2*np.array([float(meas[i]) for i in range(len(meas))]) - 1.
                    shift = np.array(list(meas)[1:] + [list(meas)[0]])
                    energy += np.dot(meas, shift) * (-J) * float(count) / float(reps)

            else:
                 state = execute(curr_circuit+self.append_circuit, self.backend).result().get_statevector()
                 distribution = np.conj(np.array(state)) * np.array(state)
                 hold = np.array([np.real(entry) for entry in distribution])
                 energy += np.inner(self.val_vector_zz, hold)

            """compute configuration score on second term"""
            if not self.t == 0.:
                for i in range(self.n):
                    curr_circuit.h(i)

                if not exact:
                    result = execute(curr_circuit+self.append_circuit, self.backend, shots=reps).result().get_counts()
                    for meas, count in result.items():
                        meas = 2*np.array([float(meas[i]) for i in range(len(meas))]) - 1.
                        energy += np.dot(np.ones(self.n), meas) * (-self.t) * float(count) / float(reps)

                else:
                    state = execute(curr_circuit+self.append_circuit, self.backend).result().get_statevector()
                    distribution = np.conj(np.array(state)) * np.array(state)
                    hold = np.array([np.real(entry) for entry in distribution])

                    energy += np.dot(self.val_vector_x, hold)

                
        elif self.hamiltonian_type == 'rot_single_qubit_z':
            energy = 0.
            
            """determine coefficient by rotation angle"""
            Jz = np.cos(self.phi)**2
            Jy = np.sin(self.phi)**2
            
            """eval performance on z axis"""
            z_circuit = circuit.to_qiskit(params=deepcopy(params))
            result = execute(z_circuit, circuit.backend, shots=reps).result().get_counts()
            energy += Jz * self.eval_dict(result)
            
            """eval performance on y axis"""
            y_circuit = circuit.to_qiskit(params=deepcopy(params))
            for i in range(self.n):
                y_circuit.rz(np.pi, i)
                y_circuit.rx(np.pi/2, i)
            result = execute(y_circuit, circuit.backend, shots=reps).result().get_counts()
            energy += Jy * self.eval_dict(result)
            
            return energy
                
        
        elif self.hamiltonian_type == 'spin_chain':
            
            energy = 0.           
            
            """compute configuration score on zz term"""
            curr_circuit = circuit.to_qiskit(params=deepcopy(params), measure=False)

            if self.init_circuit is not None:
                curr_circuit = self.init_circuit + curr_circuit
            
            state = execute(curr_circuit+self.append_circuit, self.backend).result().get_statevector()
            distribution = np.conj(np.array(state)) * np.array(state)
            hold = np.array([np.real(entry) for entry in distribution])
            energy += np.inner(self.val_vector_zz, hold)

            """compute configuration score on xx term"""
            to_basis = QuantumCircuit(self.n, self.n)
            for qubit in range(self.n):
                 to_basis.h(qubit)

            state = execute(curr_circuit+self.append_circuit+to_basis, self.backend).result().get_statevector()
            distribution = np.conj(np.array(state)) * np.array(state)
            hold = np.array([np.real(entry) for entry in distribution])
            energy += np.inner(self.val_vector_zz, hold)


            """compute configuration score on yy term"""
            to_basis = QuantumCircuit(self.n, self.n)
            for qubit in range(self.n):
                to_basis.rz(np.pi, qubit)
                to_basis.rx(np.pi/2., qubit)

            state = execute(curr_circuit+self.append_circuit+to_basis, self.backend).result().get_statevector()
            distribution = np.conj(np.array(state)) * np.array(state)
            hold = np.array([np.real(entry) for entry in distribution])
            energy += np.inner(self.val_vector_zz, hold)

            
            if not self.t == 0.:
                """compute configuration score transverse field term"""
                to_basis = QuantumCircuit(self.n, self.n)
                for i in range(self.n):
                    to_basis.h(i)

                else:
                    state = execute(curr_circuit+self.append_circuit+to_basis, self.backend).result().get_statevector()
                    distribution = np.conj(np.array(state)) * np.array(state)
                    hold = np.array([np.real(entry) for entry in distribution])

                    energy += np.dot(self.val_vector_x, hold)
            
            """spin chain energy at the moment not normalized"""
            return -(energy + 2 * float(self.n))
        
        return (self.max_value - energy) / (self.max_value - self.min_value)
            
            
            
            
