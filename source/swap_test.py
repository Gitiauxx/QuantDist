import numpy as np

from source.swap import cswap
from source.universal_gates import *
from source.utils import *
from source.operations import measure_register

def swap_test_2Q(psi, phi):
    """
    Implement a 2 Qubit swap test
    :param psi:
    :param phi:
    :return:
    """
    n = 1 + 2 * int(np.log2(psi.shape[0]))

    state = np.kron(ZERO, np.kron(psi, phi))

    H = hadamard(0, n)
    CS = cswap(0, 1, 2, n)

    out = np.dot(H, np.dot(CS, np.dot(H, state)))
    measurement_register_0 = measure_register(out, 0, n)

    return 2 * measurement_register_0 - 1

def swap_test_3Q(psi, phi, xi):
    """
    Implement a non-optimized 3 Qubit swap test
    :param psi:
    :param phi:
    :param xi:
    :return: <psi|phi>, <psi|xi>, <phi|xi>
    """
    n = 3 + 3 * int(np.log2(psi.shape[0]))

    state = np.kron(ZERO, np.kron(ZERO, np.kron(ZERO, np.kron(psi, np.kron(phi, xi)))))
    H1 = hadamard(0, n)
    H2 = hadamard(1, n)
    H3 = hadamard(2, n)

    CS1 = cswap(0, 3, 4, n)
    CS2 = cswap(1, 4, 5, n)
    CS3 = cswap(2, 3, 4, n)

    out = np.dot(H3, np.dot(H2, np.dot(H1, state)))
    out = np.dot(CS3, np.dot(CS2, np.dot(CS1, out)))
    out = np.dot(H3, np.dot(H2, np.dot(H1, out)))

    measurement_0 = 2 * measure_register(out, 0, n) - 1
    measurement_1 = 2 * measure_register(out, 1, n) - 1
    measurement_2 = 2 * measure_register(out, 2, n) - 1

    return measurement_0, measurement_1, measurement_2
