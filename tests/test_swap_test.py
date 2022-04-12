import numpy as np

from pytest import approx

from source.swap_test import *
from source.utils import *


def test_swap_test_2q():
    phi = np.array([0, 1])
    assert swap_test_2Q(phi, phi) == approx(1)

    psi = np.array([1, 0])
    assert swap_test_2Q(phi, psi) == approx(0)
    assert swap_test_2Q(psi, phi) == approx(0)

    psi = np.array([1, 1 / 2])
    assert swap_test_2Q(phi, psi) == approx(1 / 2)

    psi = 1 / np.sqrt(2) * (ZERO + ONE)
    phi = 1 / np.sqrt(2) * (ZERO - ONE)
    assert swap_test_2Q(phi, psi) == approx(0)

    psi = 1 / np.sqrt(2) * (ZERO + 1j * ONE)
    phi = 1 / np.sqrt(2) * (ZERO - ONE)
    assert swap_test_2Q(psi, phi) == approx(np.abs(np.dot(psi.conj().T, phi)) ** 2)

    psi = 1 / np.sqrt(5) * (2 * ZERO + 1j * ONE)
    phi = 1 / np.sqrt(2) * (ZERO - 1j * ONE)
    assert swap_test_2Q(psi, phi) == approx(np.abs(np.dot(psi.conj().T, phi)) ** 2)


def test_swap_test_2q_recycle():
    phi = ZERO
    assert swap_test_2Q_recycle(phi, phi) == (approx(1), approx(1), approx(1))

    psi = ONE
    assert swap_test_2Q_recycle(phi, psi) == (approx(0), approx(0), approx(1))

    psi = 1 / np.sqrt(2) * (ZERO + ONE)
    phi = ZERO
    assert swap_test_2Q_recycle(psi, psi) == (approx(1 / 2), approx(1 / 2), approx(1 / 2))
    assert swap_test_2Q_recycle(psi, phi) == (approx(1 / 2), approx(1), approx(1 / 2))

    phi = -1 / np.sqrt(2) * (ZERO + ONE)
    assert swap_test_2Q_recycle(psi, phi) == (approx(-1 / 2), approx(1 / 2), approx(1 / 2))