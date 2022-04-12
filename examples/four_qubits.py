import os

import numpy as np
from  math import sqrt
import matplotlib.pyplot as plt

from qiskit import IBMQ
#IBMQ.save_account('d5c3abdfb2d464260eeda4fc0aecd1ed1e3e64c270703dd5a0a34a1546f57c7d7a1e0d19d695c0adc43f509b7219cff4f86da11e818edcf095dfab7403649c1f')
IBMQ.load_account()

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, ClassicalRegister, BasicAer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.test.mock import FakeMelbourne


from source.utils import get_logger

provider = IBMQ.get_provider(hub='ibm-q')
for backend in provider.backends():
    print(backend)

logger = get_logger(__name__)


def get_noise(p_meas, p_gate):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_model

def construct_circuit():
    r0 = [1, 0, 0, 0]
    r1 = [sqrt(5 / 9), sqrt(4 / 9), 0, 0]
    r2 = [sqrt(5 / 9), -sqrt(1 / 9), 0. + 1j * sqrt(1 / 3), 0]
    r3 = [sqrt(5 / 9), -sqrt(1 / 9), 0. - 1j * sqrt(1 / 3), 0]

    initial_state_list = [r0, r1, r2, r3]

    qr = QuantumRegister(8, 'q')
    anc = QuantumRegister(5, 'ancilla')
    cr = ClassicalRegister(5, 'c')
    circuit = QuantumCircuit(anc, qr, cr)

    for i, r in enumerate(initial_state_list):
        circuit.initialize(r, [qr[2 * i], qr[2 * i + 1]])

    for i in range(5):
        circuit.h(anc[i])

    for i in range(2):
        circuit.cswap(anc[4], qr[i], qr[2 + i])
        circuit.cswap(anc[4], qr[4 + i], qr[6 + i])
        circuit.cswap(anc[3], qr[i], qr[4 + i])
        circuit.cswap(anc[2], qr[2 + i], qr[6 + i])
        circuit.cswap(anc[1], qr[2 + i], qr[4 + i])

        circuit.cswap(anc[0], qr[i], qr[2 + i])

    circuit.h(anc[0])

    circuit.measure(anc, cr)

    return circuit


def construct_circuit_smaller():
    r0 = [1, 0, 0, 0]
    r1 = [sqrt(5 / 9), sqrt(4 / 9), 0, 0]
    r2 = [sqrt(5 / 9), -sqrt(1 / 9), 0. + 1j * sqrt(1 / 3), 0]
    r3 = [sqrt(5 / 9), -sqrt(1 / 9), 0. - 1j * sqrt(1 / 3), 0]

    initial_state_list = [r0, r1, r2, r3]

    qr = QuantumRegister(8, 'q')
    anc = QuantumRegister(4, 'ancilla')
    cr = ClassicalRegister(4, 'c')
    circuit = QuantumCircuit(anc, qr, cr)

    for i, r in enumerate(initial_state_list):
        circuit.initialize(r, [qr[2 * i], qr[2 * i + 1]])

    for i in range(4):
        circuit.h(anc[i])

    for i in range(2):
        circuit.cswap(anc[3], qr[i], qr[6 + i])
        circuit.cswap(anc[2], qr[i], qr[2 + i])
        circuit.cswap(anc[1], qr[i + 4], qr[6 + i])

        circuit.cswap(anc[0], qr[i], qr[4 + i])

    circuit.h(anc[0])

    circuit.measure(anc, cr)

    return circuit

if __name__ == '__main__':

    n_noise = 5
    niter = 20
    nanc = 4
    circuit_type = 'small'

    data = np.zeros((n_noise, niter))
    noise_level = 0

    results_folder = '../results/four_qubit'
    os.makedirs(results_folder, exist_ok=True)
    results_file = f'{results_folder}/quasm_simulator_four_qubit_smaller.npy'

    for noise in np.linspace(0, 0.05, num=n_noise):
        for i in range(niter):
            noise_model = get_noise(noise, noise)
            circuit = construct_circuit_smaller()

            simulator =  Aer.get_backend('qasm_simulator')
            job = execute(circuit, simulator, shots=8112, noise_model=noise_model)
            result = job.result()

            # Returns counts
            counts = result.get_counts(circuit)
            total = sum(counts.values())
            counts_final = {state[:-1]: c for state, c in counts.items() if state[-1] == '0'}

            counts = {state: c / total for state, c in counts_final.items()}

            if circuit_type == 'long':
                r12 = counts['0000'] * 2 ** nanc - 1
                r13 = counts['0001'] * 2 ** nanc - 1
                r14 = counts['0010'] * 2 ** nanc - 1
                r23 = counts['1010'] * 2 ** nanc - 1
                r24 = counts['1001'] * 2 ** nanc - 1
                r34 = counts['0110'] * 2 ** nanc - 1

            elif circuit_type == 'small':
                r12 = counts['111'] * 2 ** nanc - 1
                r13 = counts['000'] * 2 ** nanc - 1
                r14 = counts['001'] * 2 ** nanc - 1
                r23 = counts['110'] * 2 ** nanc - 1
                r24 = counts['011'] * 2 ** nanc - 1
                r34 = counts['100'] * 2 ** nanc - 1

            logger.info(f'LHS of face is {r12 + r13 + r14 - r23 - r24 - r34}')
            data[noise_level, i] = r12 + r13 + r14 - r23 - r24 - r34

        noise_level += 1

    np.save(results_file, data)



