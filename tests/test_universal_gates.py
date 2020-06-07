import numpy as np

from source.universal_gates import *

P0 = np.array([[1, 0], [0, 0]])
P1 = np.array([[0, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])

def test_cnot():
    C1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.array_equal(cnot(0, 1, 2), C1)

    C2 = np.kron(np.eye(2), P0) + np.kron(X, P1)
    assert np.array_equal(cnot(1, 0, 2), C2)

    C3 = np.kron(np.eye(2), np.kron(P0, np.eye(2))) + np.kron(X, np.kron(P1, np.eye(2)))
    assert np.array_equal(cnot(1, 0, 3), C3)


def test_ccnot():

    C1 = np.eye(8)
    C1[7, 7] = 0
    C1[7, 6] = 1
    C1[6, 6] = 0
    C1[6, 7] = 1
    assert np.array_equal(ccnot((0, 1), 2, 3), C1)

    C2 = np.kron(np.eye(2), np.kron(P0, P0)) + np.kron(X, np.kron(P1, P1)) \
         + np.kron(np.eye(2), np.kron(P0, P1)) + np.kron(np.eye(2), np.kron(P1, P0))
    assert np.array_equal(ccnot((1, 2), 0, 3), C2)

    C3 = np.kron(P0, np.kron(np.eye(2), P0)) + np.kron(P1, np.kron(X, P1)) \
         + np.kron(P0, np.kron(np.eye(2), P1)) + np.kron(P1, np.kron(np.eye(2), P0))
    assert np.array_equal(ccnot((0, 2), 1, 3), C3)
