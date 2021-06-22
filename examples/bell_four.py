import os
import numpy as np
from  math import sqrt
import matplotlib.pyplot as plt

from qiskit import IBMQ
IBMQ.save_account('daf121f55fe15cc9307edc659ca1fd235174637bf2008a5a302fdfc19b389b842faf054c0dda5c6d43586cc8705cecfd62b07538e026fcb2e3accdb37096261f',
                  overwrite=True)
IBMQ.load_account()

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, ClassicalRegister, BasicAer
from qiskit.compiler import transpile, assemble
from qiskit.test.mock import FakeAthens, FakeSantiago, FakeMelbourne
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


def construct_bell_four(theta):
    """"
    Construct four states circuit
    :param: theta angle parameter
    """
    r0 = [1, 0]
    r1 = [np.cos(3 * theta / 2 - np.pi), np.sin(3 * theta / 2 - np.pi)]
    r2 = [np.cos(np.pi - theta), np.sin(np.pi - theta)]
    r3 = [np.cos(theta / 2), np.sin(theta / 2)]

    initial_state_list = [r0, r1, r2, r3]

    qr = QuantumRegister(4, 'q')
    anc = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c_anc')
    cr_q = ClassicalRegister(4, 'c_q')
    circuit = QuantumCircuit(anc, qr, cr, cr_q)

    for i, r in enumerate(initial_state_list):
        circuit.initialize(r, qr[i])

    circuit.h(anc[0])

    #control swap between qr[0] and qr[2]
    circuit.cnot(qr[2], qr[0])

    circuit.h(qr[2])
    circuit.cnot(qr[0], qr[2])
    circuit.tdg(qr[2])
    circuit.cnot(anc[0], qr[2])
    circuit.t(qr[2])
    circuit.cnot(qr[0], qr[2])
    circuit.tdg(qr[2])
    circuit.cnot(anc[0], qr[2])
    circuit.t(qr[0])
    circuit.t(qr[2])
    circuit.cnot(anc[0], qr[0])
    circuit.h(qr[2])
    circuit.t(anc[0])
    circuit.tdg(qr[0])
    circuit.cnot(anc[0], qr[0])

    circuit.cnot(qr[2], qr[0])

    # bell measurement
    circuit.cnot(qr[0], qr[1])
    circuit.cnot(qr[2], qr[3])
    circuit.h(qr[0])
    circuit.h(qr[2])

    circuit.measure(qr, cr_q)
    circuit.measure(anc, cr)

    return circuit, qr, anc

def construct_bell_four_swap(theta):
    r0 = [1, 0]
    r1 = [np.cos(3 * theta / 2 - np.pi), np.sin(3 * theta / 2 - np.pi)]
    r2 = [np.cos(np.pi - theta), np.sin(np.pi - theta)]
    r3 = [np.cos(theta / 2), np.sin(theta / 2)]

    initial_state_list = [r0, r1, r2, r3]

    qr = QuantumRegister(4, 'q')
    anc = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c_anc')
    cr_q = ClassicalRegister(4, 'c_q')
    circuit = QuantumCircuit(anc, qr, cr, cr_q)

    for i, r in enumerate(initial_state_list):
        circuit.initialize(r, qr[i])

    circuit.h(anc[0])

    circuit.cnot(qr[2], qr[0])

    circuit.h(qr[2])
    circuit.cnot(qr[0], qr[2])
    circuit.tdg(qr[2])

    circuit.cnot(qr[0], qr[2])
    circuit.cnot(anc[0], qr[0])
    circuit.cnot(qr[0], qr[2])
    circuit.cnot(anc[0], qr[0])

    circuit.t(qr[2])
    circuit.cnot(qr[0], qr[2])
    circuit.tdg(qr[2])

    circuit.cnot(qr[0], qr[2])
    circuit.cnot(anc[0], qr[0])
    circuit.cnot(qr[0], qr[2])
    circuit.cnot(anc[0], qr[0])

    circuit.t(qr[0])
    circuit.t(qr[2])
    circuit.cnot(anc[0], qr[0])
    circuit.h(qr[2])
    circuit.t(anc[0])
    circuit.tdg(qr[0])
    circuit.cnot(anc[0], qr[0])

    circuit.cnot(qr[2], qr[0])

    circuit.cnot(anc[0], qr[1])
    circuit.cnot(qr[0], anc[0])
    circuit.cnot(anc[0], qr[1])
    circuit.cnot(qr[0], anc[0])

    circuit.cnot(qr[2], qr[3])
    circuit.h(qr[0])
    circuit.h(qr[2])

    circuit.measure(qr, cr_q)
    circuit.measure(anc, cr)

    return circuit, qr, anc

def construct_bell_four_belem(theta):
    r0 = [1, 0]
    r1 = [np.cos(3 * theta / 2 - np.pi), np.sin(3 * theta / 2 - np.pi)]
    r2 = [np.cos(np.pi - theta), np.sin(np.pi - theta)]
    r3 = [np.cos(theta / 2), np.sin(theta / 2)]

    initial_state_list = [r0, r1, r2, r3]

    qr = QuantumRegister(4, 'q')
    anc = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c_anc')
    cr_q = ClassicalRegister(4, 'c_q')
    circuit = QuantumCircuit(anc, qr, cr, cr_q)

    for i, r in enumerate(initial_state_list):
        circuit.initialize(r, qr[i])

    circuit.h(anc[0])

    circuit.cnot(qr[2], qr[0])

    circuit.h(qr[2])
    circuit.cnot(qr[0], qr[2])
    circuit.tdg(qr[2])

    circuit.cnot(qr[0], qr[2])
    circuit.cnot(anc[0], qr[0])
    circuit.cnot(qr[0], qr[2])
    circuit.cnot(anc[0], qr[0])

    circuit.t(qr[2])
    circuit.cnot(qr[0], qr[2])
    circuit.tdg(qr[2])

    circuit.cnot(qr[0], qr[2])
    circuit.cnot(anc[0], qr[0])
    circuit.cnot(qr[0], qr[2])
    circuit.cnot(anc[0], qr[0])

    circuit.t(qr[0])
    circuit.t(qr[2])
    circuit.cnot(anc[0], qr[0])
    circuit.h(qr[2])
    circuit.t(anc[0])
    circuit.tdg(qr[0])
    circuit.cnot(anc[0], qr[0])

    circuit.cnot(qr[2], qr[0])

    circuit.cnot(qr[0], qr[1])

    circuit.cnot(qr[2], qr[3])
    circuit.h(qr[0])
    circuit.h(qr[2])

    circuit.measure(qr, cr_q)
    circuit.measure(anc, cr)

    return circuit, qr, anc

def run_job(circuit, backend, layout=None, real=False, noise_model=None):

    if real:
        mapped_circuit = transpile(circuit, backend=backend, optimization_level=1, initial_layout=layout)

        logger.info(f'Depth: {mapped_circuit.depth()}')
        logger.info(f'Gates count: {mapped_circuit.count_ops()}')


        qobj = assemble(mapped_circuit, backend=backend, shots=8192)
        job = backend.run(qobj)
        result = job.result()

    else:
        mapped_circuit = None
        job = execute(circuit, backend, shots=8192, noise_model=noise_model)
        result = job.result()

    return result, mapped_circuit

def bitwise_and(x, y):

    assert len(x) == len(y)

    total = 0
    for i in range(len(x)):
        total += int(x[i]) * int(y[i])

    return total


# Returns counts
def count(result):
    counts = result.get_counts(circuit)
    total = sum(counts.values())

    counts = {state: c / total for state, c in counts.items()}

    r12_state = 0
    for state, c in counts.items():
        if (state[-1] == '0') & (state[2:4] == '00'):
            r12_state += c
    r12 = r12_state * 2 ** 2

    r23_state = 0
    for state, c in counts.items():
        if (state[-1] == '1') & (state[2:4] == '00'):
            r23_state += c
    r23 = r23_state * 2 ** 2

    r34_state = 0
    for state, c in counts.items():
        if (state[-1] == '0') & (state[:2] == '00'):
            r34_state += c
    r34 = r34_state * 2 ** 2

    r14_state = 0
    for state, c in counts.items():
        if (state[-1] == '1') &  (state[:2] == '00'):
            r14_state += c
    r14 = r14_state * 2 ** 2

    logger.info(f'LHS of face is {r12 + r23 + r34 - r14}')

    return r12 + r23 + r34 - r14

# backend =  Aer.get_backend('qasm_simulator')
# backend = FakeSantiago()

provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_lima')

iterations = 10
npoints = 20
results = np.empty((npoints, 1 + iterations))
p_meas = 0.015
p_gate = 0.01

pointer = 0
for theta in np.linspace(0, np.pi / 4, npoints, endpoint=True):
    logger.info(f' Angle {np.pi - theta}')

    for i in range(iterations):
        noise_model = get_noise(p_meas, p_gate)
        circuit, qr, anc = construct_bell_four_belem(np.pi - theta)
        layout = {qr[0]: 2, qr[1]: 0, qr[2]: 3, qr[3]: 4, anc[0]: 1}
        layout_belem = {qr[0]: 1, qr[1]: 0, qr[2]: 3, qr[3]: 4, anc[0]: 2}
        logger.info(f'Iteration {i}')
        result, mapped_circuit = run_job(circuit, backend, layout=layout_belem, real=True)
        lhs = count(result)

    # results[i, 0] = mapped_circuit.depth()
    #
    # gate_counts = mapped_circuit.count_ops()
    # results[i, 1] = gate_counts['rz']
    # results[i, 2] = gate_counts['cx']
    # results[i, 3] = gate_counts['sx']

        results[pointer, 1 + i] = lhs

    results[pointer, 0] = theta
    pointer += 1

# accuracy = (results[:, 4] >= 2).astype('int32').mean()
# logger.info(f'Accuracy of quantum-classic classifier is {accuracy}')

results_folder = '../results/four_qubit'
os.makedirs(results_folder, exist_ok=True)
np.save(f'{results_folder}/accuracy_classifier_real_lima.npy', results)



