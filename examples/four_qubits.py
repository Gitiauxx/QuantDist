import numpy as np
from  math import sqrt
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, ClassicalRegister, BasicAer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

from source.utils import get_logger

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

noise_model = get_noise(0.02,0.02)

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
circuit.draw(output='mpl')
plt.show()

# Execute the circuit on the qasm simulator
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=20000, noise_model=noise_model)
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
total = sum(counts.values())
counts_final = {state[:-1]: c for state, c in counts.items() if state[-1] == '0'}

counts = {state: c / total for state, c in counts_final.items()}

r12 = counts['0000'] * 2 ** 5 - 1
r13 = counts['0001'] * 2 ** 5 - 1
r14 = counts['0010'] * 2 ** 5 - 1
r23 = counts['1010'] * 2 ** 5 - 1
r24 = counts['1001'] * 2 ** 5 - 1
r34 = counts['0110'] * 2 ** 5 - 1


logger.info(f'LHS of face is {r12 + r13 + r14 - r23 - r24 - r34}')



