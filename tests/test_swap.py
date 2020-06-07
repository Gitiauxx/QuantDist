import numpy as np

from source.swap import *
from source.utils import *

def test_cswap():
    S = np.eye(8)
    S[5, 5] = 0
    S[5, 6] = 1
    S[6, 6] = 0
    S[6, 5] = 1

    assert np.array_equal(cswap(0, 1, 2, 3), S)

    phi = np.array([0, 1])
    psi = np.array([1, 0])
    xi = np.array([1, 1])

    state = np.kron(ZERO, np.kron(phi, psi))
    expected_swap = np.kron(ZERO, np.kron(phi, psi)).astype(float)
    obtained_swap = np.dot(cswap(0, 1, 2, 3), state)

    assert np.array_equal(expected_swap, obtained_swap)

    state = np.kron(ONE, np.kron(phi, psi))
    expected_swap = np.kron(ONE, np.kron(psi, phi)).astype(float)
    obtained_swap = np.dot(cswap(0, 1, 2, 3), state)

    assert np.array_equal(expected_swap, obtained_swap)

    state = np.kron(phi, np.kron(ONE, psi))
    expected_swap = np.kron(psi, np.kron(ONE, phi)).astype(float)
    obtained_swap = np.dot(cswap(1, 0, 2, 3), state)

    assert np.array_equal(expected_swap, obtained_swap)

    state = np.kron(ONE, np.kron(np.kron(phi, psi), xi))
    expected_swap = np.kron(ONE, np.kron(np.kron(psi, phi), xi)).astype(float)
    obtained_swap = np.dot(cswap(0, 1, 2, 4), state)

    assert np.array_equal(expected_swap, obtained_swap)

    state = np.kron(phi, np.kron(np.kron(ONE, psi), xi))
    expected_swap = np.kron(phi, np.kron(np.kron(ONE, xi), psi)).astype(float)
    obtained_swap = np.dot(cswap(1, 2, 3, 4), state)

    assert np.array_equal(expected_swap, obtained_swap)


