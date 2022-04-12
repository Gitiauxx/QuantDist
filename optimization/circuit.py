import numpy as np

from source.universal_gates import x, rz, cnot

ATHENS = {'A': x(0, 5), 'B': x(1, 5), 'C': x(2, 5), 'D': x(3, 5), 'E': x(4, 5),
          'F': cnot(0, 1, 5), 'G': cnot(1, 2, 5), 'H': cnot(2, 3, 5), 'I': cnot(3, 4, 5),
          'J': lambda theta: rz(0, 5, theta), 'K': lambda theta: rz(1, 5, theta),
          'L': lambda theta: rz(2, 5, theta), 'M': lambda theta: rz(3, 5, theta),
          'N': lambda theta: rz(4, 5, theta), 'size': 5}

class Circuit(object):
    """
    Quantum circuits generates from specification using matrix representation
    Specifications is a string of letters. Letters are ordered as in a circuit
    so mtrix representation is reversed
    """

    def __init__(self, specifications, machine=ATHENS):

        self.spec = specifications
        self.machine = machine
        self.n = self.machine['size']

    def construct(self):

        self.theta = []
        out = np.eyes(self.n)

        for letter in self.spec:
            gate = self.machine[letter]
            if letter in ['J', 'K', 'L', 'M', 'N']:
                theta = np.random.uniform(0, 2 * np.pi)
                gate = gate(theta)
            out = np.dot(gate, out)


