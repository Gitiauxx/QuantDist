import numpy as np
from source.utils import apply_one_qbit_gate


def prepare(a):
    """
    Prepare a vector a into a quantum state
    :param a:
    :return:
    """
    return a / np.sqrt((a ** 2).sum())


def measure_register(psi, x, n):
    """
    Measure the probability that the xth register
    is in a state |0> r for a system of size n.

    For example, if n=2 and x=1, the probability is
    |<phi|phi>|^2 with phi = P |psi> and P = kron(I, M), M=[[1, 0], [0, ]]
    :param state:
    :param x:
    :return:
     """
    M = np.array([[1, 0], [0, 0]])
    P = apply_one_qbit_gate(x, n, M)

    subspace = np.dot(P, psi)

    return np.linalg.norm(subspace) ** 2

