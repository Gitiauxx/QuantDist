import numpy as np
from pytest import approx

from source.operations import measure_register

def test_measure_register():

    zero = np.array([1, 0])
    one = np.array([0, 1])
    psi1 = 1 / np.sqrt(2) * (np.kron(zero, zero) + np.kron(one, zero))

    assert measure_register(psi1, 0, 2) == approx(1 / 2)

    psi2 = 1 / np.sqrt(2) * (1j * np.kron(zero, zero) + np.kron(one, zero))
    assert measure_register(psi2, 0, 2) == approx(1 / 2)

    psi3 = 1 / np.sqrt(3) * (np.kron(zero, np.kron(zero, zero)) + np.kron(one, np.kron(zero, one))
                             + np.kron(one, np.kron(one, one)))
    assert measure_register(psi3, 1, 3) == approx(2 / 3)