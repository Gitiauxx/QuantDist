import numpy as np
import argparse

from source.universal_gates import cnot, t, tg, id, hadamard
from ga.generic_ga import GA


def construct_matrix_from_list(circuit_list, n):
    """
    Construct unitary corresponding to a circuit of logic gates with n inputs
    :param circuit_list:
    :param n:
    :return: Unitary
    """
    unitary = np.eye(2 ** n)

    for gate in circuit_list[::-1]:
        name = gate[0]

        if name == 'h':
            x = gate[1]
            H = hadamard(x, n)
            unitary = np.dot(H, unitary)

        elif name == 't':
            x = gate[1]
            T = t(x, n)
            unitary = np.dot(T, unitary)

        elif name == 'tg':
            x = gate[1]
            Tg = tg(x, n)
            unitary = np.dot(Tg, unitary)

        elif name == 'id':
            continue

        elif name == 'cnot':
            c = gate[1]
            x = gate[2]
            C = cnot(c, x, n)
            unitary = np.dot(C, unitary)

        else:
            raise ValueError(f"Universal gates are: h, tg, cnot, t, id; we get {name}")

    return 1 / np.sqrt(2 ** n) * unitary


def count_cnot(circuit_list):
    """
    Count the number of cnots in a circuit
    :param circuit_list:
    :return:
    """

    counter = 0

    for gate in circuit_list:
        name = gate[0]
        if name == 'cnot':
            counter += 1

    return counter


def count_single_qubit_gate(circuit_list):
    """
    Count the number of single qubit gates in a circuit
    :param circuit_list:
    :return:
    """

    counter = 0

    for gate in circuit_list:
        name = gate[0]
        if name != 'cnot':
            counter += 1

    return counter

class CircuitGA(GA):

    def __init__(self, depth, target_circuit,
                 ninputs=5, pop_size=500, n_genes=2, mutation_rate=0.1, max_gen=1000, stop=0, fidelity_weight=10, n_mutation=10):
        super().__init__(pop_size=pop_size, n_genes=n_genes, mutation_rate=mutation_rate, max_gen=max_gen, stop=stop, n_mutation=n_mutation)

        self.depth = depth
        self.ninputs = ninputs
        self.target = construct_matrix_from_list(target_circuit, ninputs)
        self.target_list = target_circuit
        self.fidelity_weight = fidelity_weight

        self.gates = ['cnot', 't', 'tg', 'id', 'h']

    def fitness_function(self, individual):

        unitary = construct_matrix_from_list(individual, self.ninputs)
        fidelity = 1 - np.abs(np.trace(np.dot(unitary.conj().T, self.target)))
        cnot_count = count_cnot(individual)
        single_gate = count_single_qubit_gate(individual)

        return fidelity * self.fidelity_weight + 0.0 * cnot_count + 0.0 * single_gate

    def select_parents(self, fitness):

        min1 = -1
        min2 = -1

        min_value1 = np.inf
        min_value2 = np.inf

        for i, value in enumerate(fitness):
            if (value < min_value1):

                min_value2 = min_value1
                min2 = min1

                min_value1 = value
                min1 = i

            elif value < min_value2:
                min_value2 = value
                min2 = i

        return min1, min2

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            circuit = []
            counter = 0

            while counter < self.depth:
                idx = np.random.choice(len(self.gates))
                x = np.random.randint(0, self.ninputs)

                if self.gates[idx] == 'cnot':
                    r = np.arange(self.ninputs)
                    p = np.ones_like(r)
                    p[x] = 0
                    c = np.random.choice(r, 1, p=p/p.sum())[0]
                    circuit.append(('cnot', c, x))

                else:
                    circuit.append((self.gates[idx], x))

                counter += 1

            population.append(circuit)


        return population

    def mutate(self, individual):

        id = np.random.randint(0, len(individual))

        new_id = np.random.choice(len(self.gates))
        x = np.random.randint(self.ninputs)

        if self.gates[new_id] == 'cnot':
            r = np.arange(self.ninputs)
            p = np.ones_like(r)
            p[x] = 0
            c = np.random.choice(r, 1, p=p / p.sum())[0]

            individual[id] = ('cnot', c, x)

        else:
            individual[id] = (self.gates[new_id], x)

        return individual

    def process_mutation(self, population, ma, pa):
        counter = 0
        while counter < self.n_mutation:
            id = np.random.randint(0, self.pop_size)

            if id != ma:
                u = np.random.rand()
                if u > 0.5:
                    population[id] = self.mutate(population[id])
                else:
                    circuit = population[id]
                    gate_id = np.random.randint(0, len(circuit))
                    x = np.random.randint(0, self.ninputs)

                    if circuit[gate_id][0] != 'cnot':
                        circuit[gate_id] = (circuit[gate_id][0], x)

                    else:
                        v = np.random.rand()

                        if v > 0.5:
                            y = circuit[gate_id][2]
                            while x == y:
                                x = np.random.randint(0, self.ninputs)
                            circuit[gate_id] = (circuit[gate_id][0], x, y)

                        else:
                            y = circuit[gate_id][1]
                            while x == y:
                                x = np.random.randint(0, self.ninputs)
                            circuit[gate_id] = (circuit[gate_id][0], y, x)

                    population[id] = circuit

                counter += 1

            else:
                continue

        return population

    def crossover(self, first_parent, sec_parent):
        crossover_pt = np.random.randint(1, len(first_parent))
        offspring1 = first_parent[:crossover_pt] + sec_parent[crossover_pt:]
        offspring2 = sec_parent[:crossover_pt] + first_parent[crossover_pt:]

        return offspring1, offspring2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ninputs', type=int, default=5)
    parser.add_argument('--depth', type=int, default=32)
    parser.add_argument('--mutation_rate', type=int, default=0.25)
    parser.add_argument('--num_iterations', type=int, default=100000)
    parser.add_argument('--n_mutation', type=int, default=100)
    parser.add_argument('--pop_size', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fidelity_weight', type=float, default=200.)
    parser.add_argument('--run', default='test')

    args = parser.parse_args()


    depth = args.depth
    ninputs = args.ninputs
    mutation_rate = args.mutation_rate
    n_genes = depth
    fidelity_weight = args.fidelity_weight
    pop_size = args.pop_size
    n_mutation = args.n_mutation

    np.random.seed(args.seed)

    target_list = [('h', 0), ('cnot', 3, 1), ('h', 3), ('cnot', 1, 3), ('tg', 3), ('cnot', 0, 3), ('t', 3),
                   ('cnot', 1, 3), ('tg', 3), ('cnot', 0, 3), ('t', 1), ('t', 3), ('cnot', 0, 1), ('h', 3),
                   ('t', 0), ('tg', 1), ('cnot', 0, 1), ('cnot', 3, 1), ('cnot', 1, 2), ('cnot', 3, 4),
                   ('h', 1), ('h', 3)]
    for i in range(depth - len(target_list)):
        target_list.append(('id', np.random.randint(0, ninputs)))

    target_circuit = construct_matrix_from_list(target_list, ninputs)

    ca = CircuitGA(depth, target_list,
                   ninputs=ninputs,
                   pop_size=args.pop_size,
                   n_genes=n_genes,
                   mutation_rate=mutation_rate,
                   max_gen=args.num_iterations,
                   stop=0,
                   fidelity_weight=fidelity_weight,
                   n_mutation=n_mutation)
    circuit, fitness = ca.run()

    c = construct_matrix_from_list(circuit, ninputs)
    fidelity  = np.abs(np.trace(np.dot(c.conj().T, target_circuit)))

    print(circuit)
    print(f' Fidelity: {fidelity}')
    print(f'Number of cnot: {count_cnot(circuit)}')

