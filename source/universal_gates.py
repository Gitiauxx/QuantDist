import numpy as np
from source.utils import apply_one_qbit_gate

def hadamard(x, n):
    """
    implement a hadamard gate as a 2 by 2 matrix on the xth qubit in a system of size n
    :param x:
    :param n: system size
    :return:
    """
    H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    return apply_one_qbit_gate(x, n, H)

def t(x, n):
    """
    Implement a phase shift T gate T = [[1, 0], [0, exp(i pi/4)]] on the xth qbit of
    a system of size n
    :param x: position to apply tha gate
    :param n:
    :return:
    """
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    return apply_one_qbit_gate(x, n, T)


def cnot(c, x, n):
    """
    CNOT operation with c as control and x as target, using projection and tensor product.
    For example if n=2, c=0, x=1:
    P1 = [[0, 0], [0, 1]] and P0=[[1, 0], [0, 0]] and CNOT = kron(P0, I) + kron(P1, X).

    For example, if n=3, c=2, x=1:
    CNOT = kron(I, I, P0) + kron(I, X, P1).

    :param c: position of control
    :param x: position of X operation
    :param n: size of system
    :return: (2^n, 2^n) matrix
    """

    X = np.array([[0, 1], [1, 0]])
    P1 = np.array([[0, 0], [0, 1]])
    P0 = np.array([[1, 0], [0, 0]])

    C0 = np.ones(1)
    C1 = np.ones(1)
    for i in range(n):
        if i == c:
            C0 = np.kron(C0, P0)
            C1 = np.kron(C1, P1)
        else:
            C0 = np.kron(C0, np.eye(2))
            if i == x:
                C1 = np.kron(C1, X)
            else:
                C1 = np.kron(C1, np.eye(2))

    return C0 + C1


def ccnot(controls, x, n):
    """
    Implement a Toffoli gate controlled by controls[0] and controls[1]

    :param controls:
    :param x:
    :param n:
    :return:
    """
    c1, c2 = controls
    CNOT2 = cnot(c2, x, n)
    CNOT1 = cnot(c1, x, n)
    CNOT = cnot(c1, c2, n)
    TX = t(x, n)
    T1 = t(c1, n)
    T2 = t(c2, n)
    H = hadamard(x, n)

    TCT = np.dot(TX, np.dot(CNOT1, TX.conj().T))

    return np.round(np.real(np.dot(CNOT,
                  np.dot(T2.conj().T,
                         np.dot(T1,
                                np.dot(H,
                                       np.dot(CNOT,
                                              np.dot(T2,
                                                     np.dot(TCT,
                                                            np.dot(CNOT2,
                                                                   np.dot(TCT,
                                                                          np.dot(CNOT2, H))))))))))))