# Explorations into the Quantum Natural Gradient

This repository contains the source code for experimentation concerning the Quantum Natural Gradient. Its main purpose is to facilitate collaboration with colleagues.

### Requirements

- **qiskit 0.19.3**
- **qiskit-aer 0.5.2**
- **qiskit-terra 0.14.1**
- **qiskit-aqua 0.7.1**
- **numpy 1.18.5**
- **matplotlib 3.1.2**


### Instructions

The so far available toy-example is in 'main.py'.

For the experiments we require objects to (i) provide circuit architecture and translation to simulator backend methods (see utils.py) (ii) a Hamiltonian object capable of evaluating measurements on the ground of chosen physical principles (see physics.py) and (iii) an optimizer responsible for determining optimization steps and other helpful quantities such as the fubini-study metric (see optim.py).

Please find the respective methods in the according directories.