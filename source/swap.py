import numpy as np
from source.universal_gates import *


def cswap(c, x, y, n):
    """
    Control swap with c as control, swapping x and y.
    The gate is implemented using projection and tensor product as in
    https://pdfs.semanticscholar.org/4774/7792a7b13028e47c9daa2259f77d264a27a8.pdf

    CSWAP(c, x, y, n) = kron(I, CNOT(y, x, n-1))CCNOT((c, x), y, n)kron(I, CNOT(y, x, n-1))

    For example, if n=3, c=1, x=(0, 2), U=SWAP:
    CU = kron(I, P0, I) + kron(SWAP, P1).

    :param c: position of control
    :param x: position of X operation
    :param n: size of system
    :param Unitary operation to be applied on the
    :return: (2^n, 2^n) matrix
    """

    CNOT2 = cnot(y, x, n)
    CCNOT = ccnot((c, x), y, n)
    return np.dot(CNOT2, np.dot(CCNOT, CNOT2))


def swap(x, y, n):
    """
    implement a swap from  3 CNOTs
    :param x: position of first qubit to swap
    :param y: position of second qubit to swap
    :param n: size of system
    :return:
    """

    C1 = cnot(x, y, n)
    C2 = cnot(y, x, n)

    return np.dot(C1, np.dot(C2, C1))

if __name__ == '__main__':
    Y = np.array([0, 1])
    X = np.array([1, 1])

    Z = ccnot((0, 1), 2, 3)
    print(Z)
    ZZ = cswap(0, 1, 2, 3)
    print(ZZ)