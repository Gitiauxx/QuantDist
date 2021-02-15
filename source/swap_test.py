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


def swap_test_2Q_recycle(psi, phi):
    """
    Implement a 2 Qubit swap test with psi recycled on last register and phi recycled on the
    third register
    :param psi:
    :param phi:
    :return:
    """
    n = 2 + 2 * int(np.log2(psi.shape[0]))

    state = np.kron(ZERO, np.kron(psi, np.kron(phi, ZERO)))

    H = hadamard(0, n)
    CS = cswap(0, 1, 2, n)
    CNOT1 = cnot(1, 3, n)
    CNOT2 = cnot(2, 3, n)
    CCNOT = ccnot((0, 3), 2, n)

    out = np.dot(CS, np.dot(H, state))
    out = np.dot(CNOT2, np.dot(CNOT1, out))
    out = np.dot(CNOT2, np.dot(CCNOT, out))
    out = np.dot(H, out)

    measurement_register_0 = measure_register(out, 0, n)
    measurement_register_2 = measure_register(out, 2, n)
    measurement_register_3 = measure_register(out, 3, n)

    return 2 * measurement_register_0 - 1, measurement_register_2, measurement_register_3


def swap_test_8Q(phi):
    """
    Implement 8 Qubit swap test
    :param phi:

    :return: <phi_i|phi_j>
    """
    n = 8 + phi.shape[1] * int(np.log2(phi.shape[0]))

    state = np.kron(phi[-2, :], phi[-1, :])
    for i in range(2, phi.shape[0]):
        state = np.kron(phi[-i -1, :], state)

    H_list = []
    for i in range(8):
        state = np.kron(ZERO, state)
        H_list.append(hadamard(i, n))

    CS_list = []

    CS_list.append(cswap(7, 8, 9, n))
    CS_list.append(cswap(7, 10, 11, n))
    CS_list.append(cswap(6, 8, 10, n))
    CS_list.append(cswap(5, 9, 11, n))
    CS_list.append(cswap(4, 9, 10, n))

    CS_list.append(cswap(7, 12, 13, n))
    CS_list.append(cswap(7, 14, 15, n))
    CS_list.append(cswap(6, 12, 14, n))
    CS_list.append(cswap(5, 13, 15, n))
    CS_list.append(cswap(4, 13, 15, n))

    CS_list.append(cswap(3, 8, 12, n))
    CS_list.append(cswap(2, 9, 13, n))
    CS_list.append(cswap(1, 9, 12, n))

    CS_list.append(cswap(0, 1, 2, n))

    out = np.dot(H_list[0], state)
    for h in H_list[1:]:
        out = np.dot(h, out)

    for c in CS_list:
        out = np.dot(c, out)

    #for h in H_list[1:]:
    out = np.dot(H_list[0], out)

    measurement_list = []
    for i in range(0, 8):
        measurement_list.append(measure_register(out, i, n))

    return measurement_list

if __name__ == '__main__':

    for i in range(8):
        for j in range(i):
            phi_list = [ZERO] * (j - 1)
            phi_list.append(ONE)
            phi_list = phi_list + [ZERO] * (i - j -1)
            phi_list.append(ONE)
            phi_list = phi_list + [ZERO] * (8 - i - 1)

            phi = np.row_stack(phi_list)

            meas = swap_test_8Q(phi)
            print(meas)
