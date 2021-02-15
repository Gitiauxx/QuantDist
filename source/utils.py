import logging
import sys

import numpy as np

ZERO = np.array([1, 0])
ONE = np.array([0, 1])

def get_logger(name):
    """
    Return a logger for current module
    Returns
    -------

    logger : logger instance

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                                  datefmt="%Y-%m-%d - %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger

def apply_one_qbit_gate(x, n, gate):
    """
    A utility to apply a one-qbit gate on the xth qbit of a system of size n
    using tensor product:

    for example if gate = hadamard, n=3 and x=2, returns
    kron(I, I , hadamard)

    :param x: position of the gate
    :param n: size of system
    :param gate: gate to apply as a 2 by 2 matrix
    :return: (2^n, 2^n) matrix
    """

    OUT = np.ones(1)

    for i in range(n):
        if i == x:
            OUT = np.kron(OUT, gate)
        else:
            OUT = np.kron(OUT, np.eye(2))

    return OUT