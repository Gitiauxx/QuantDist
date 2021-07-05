# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:34:36 2020

@author: Ian
"""
#import modules
import argparse
import os
import numpy as np
import sympy as sp
import random as r
import datetime
import json


def count_cswap_gates(circuit):
    """
    Count the number of control swap gates in circuit
    :param circuit:
    :return:
    """
    count = 0
    for gate in circuit:
        if gate[0] == 'cswap':
            count += 1
    return count


def genetic_search(depth, numAncillas, numInputs, genSize=250, mutationChance=20, numIteration=500, seed=0):

    numQubits = numAncillas + numInputs
    r.seed(seed)

    genSize = 250      #gensize is the size of the population created - it's usually between 150 and 250

    a,b,c,d,e,f,g,h,z,y,u,t, l, m, n, o, p, s = sp.symbols('a,b,c,d,e,f,g,h,z,y,u,t, l, m, n, o, p, s') #these variables correspond to inputs like phi1, phi2 etc.
    varList = [a,b,c,d,e,f,g,h,z,y,u,t, l, m, n, o, p, s]
    initialQuantumState = [[0]*numAncillas + varList[:numInputs]]
    initialQuantumState[0].append(1)

    print(initialQuantumState)

    terminationCriteria = np.math.factorial(numInputs)/(2*np.math.factorial(numInputs-2)) #termination criteria is the total number of overlaps we're looking for
    print("Termination Fitness: "+str(terminationCriteria))


    #***UTILITY METHODS***#
#   These methods are useful tools that make the overall program work more efficiently

    #simplify - takes in a quantum state and returns a condensed version that makes sure
#           any kets in the same state are combined 
    def simplify(quantumState):
        finalState = []
        condensed = []
    
        for i in range(len(quantumState)):
            currKet = []
            for j in range(len(quantumState[i])-1):
                currKet.append(quantumState[i][j])
            if currKet in condensed:
                continue
            else:
                coeff = quantumState[i][numAncillas+numInputs]
                for j in range(i+1, len(quantumState)):
                    activate = True
                    for k in range(len(quantumState[j])-1):
                        if currKet[k] != quantumState[j][k]:
                            activate = False
                            break
                    if activate:
                        coeff += quantumState[j][numAncillas+numInputs]
                copy = currKet.copy()
                condensed.append(copy)
                currKet.append(coeff)
                finalState.append(currKet)
        return finalState

#***GATE ELEMENTS***#
#These are all the gate methods to be called when creating a circuit; they return what a certain
#gate will look like in a list of gates that then comprise the circuit 

#HAD - returns a hadamard gate: list with first element as 'h' to identify it as a H gate
#      and the second element is the qubit upon which it is acting
    def HAD(qubit):
        h = ['h']
        h.append(qubit)
        return h

#CSWAP - returns a CSWAP gate: list with first element as 'cswap' to identify it as a CSWAP gate,
#        the second element is the qubit number which is the control, and third element is a list
#        of the two qubit numbers which are being swapped
    def CSWAP(control, targets):
        c = ['cswap']
        c.append(control)
        c.append(targets)
        return c



#***GATE APPLIERS***#
#These are the methods which when called transform a quantum state based on what gate is being 
#applied. For example, calling hadamard applies the hadamard gate to a quantum state and returns
#an updated quantum state 

#hadamard - returns updated quantum state after applying hadamard gate to it 
    def hadamard(qubit, quantumState):
        newQuantumState = quantumState.copy()
        for i in range(len(quantumState)): #Goes through all the states in the quantum state
            if quantumState[i][qubit] == 0:  #if the qubit being acted upon is zero, it will create a new state with 1 as well and add that to the list
                newState = quantumState[i].copy()
                newState[qubit] = 1
                newQuantumState.append(newState)
            else:              #if the qubit acted upon is one, then it replaces the current state with a negative version, and creates a new state with 0 replacing the one
                newQuantumState[i][numQubits] = -1*quantumState[i][numQubits]
            
                newState = quantumState[i].copy()
                newState[qubit] = 0
                newState[numQubits] = 1
                newQuantumState.append(newState)
        return newQuantumState

#cswap - returns updated quantum state after applying cswap gate to it 
    def cswap(control, targets, quantumState):
        newQuantumState = quantumState.copy()
        for i in range(len(quantumState)):
            if newQuantumState[i][control] == 0:
                continue
            else:
                firstTarget = newQuantumState[i][targets[0]]
                secondTarget = newQuantumState[i][targets[1]]
                newQuantumState[i][targets[0]] = secondTarget
                newQuantumState[i][targets[1]] = firstTarget
        return newQuantumState

#swap - returns updated quantum state after swapping two target qubits 
    def swap(targets,quantumState):
        copyState = quantumState.copy()
        for i in range(len(quantumState)):
            temp = copyState[i][targets[0]]
            copyState[i][targets[0]] = quantumState[i][targets[1]]
            copyState[i][targets[1]] = temp
        return copyState

#swapSinglet - returns updated quantum state for single ket after swapping
    def swapSinglet(targets, quantumState):
        copyState = quantumState.copy()
        temp = copyState[targets[0]]
        copyState[targets[0]] = quantumState[targets[1]]
        copyState[targets[1]] = temp
        return copyState

#run_circuit - takes in a circuit and a quantum state upon which the circuit will act and 
#              returns the quantum state the circuit produces 
    def run_circuit(circuit):
        quantumState = initialQuantumState.copy()
        for i in range(len(circuit)):
            if circuit[i][0] == 'h':
                quantumState = hadamard(circuit[i][1],quantumState)
            elif circuit[i][0] == 'cswap':
                quantumState = cswap(circuit[i][1], circuit[i][2], quantumState)
    
        quantumState = simplify(quantumState)
        return quantumState


#***CIRCUIT GENERATION***#
#These methods involve the creation of random circuits, the creation of broad populations 
#of circuits etc. Circuits are represented as lists of gates from the gate elements section
    
#randomCircuit - returns a circuit with random gates 
    def randomCircuit():
        circuit = []
    
        #hadamard zone
        for i in range(numAncillas):
            circuit.append(HAD(i))
    
        #cswap zone
        for i in range(int(depth)):
            control = r.randint(0,numAncillas-1)
            targets = [r.randint(numAncillas,numAncillas+numInputs-1)]
            t2 = r.randint(numAncillas,numAncillas+numInputs-1)
            while t2 == targets[0]:
                t2 = r.randint(numAncillas,numAncillas+numInputs-1)
            targets.append(t2)
            circuit.append(CSWAP(control,targets))
    
    #Last hadamard zone
    #circuit.append(HAD(control))
    
        return circuit


#initialPopulation - returns a list of random circuits of size genSize 
    def initialPopulation():
        initPopulation = []
        for i in range(genSize):
            initPopulation.append(randomCircuit())
        return initPopulation

#randomGate - returns a random gate, either identity gate, hadamard, cswap, or mcmt
    def randomGate():#THIS CAN BE MOVED SOMEWHERE ELSE
        randum = r.randint(0,10)
        if randum<3:
            return ['id']
        else:
            return CSWAP(r.randint(0,numAncillas-1),[r.randint(numAncillas,numAncillas+numInputs-1), r.randint(numAncillas,numAncillas+numInputs-1)])


#***FITNESS CALCULATION***#
#These methods are geared towards calculating the fitness values of circuits 
#based on certain fitness criteria that can be changed based on the users preference

#fitnessCalculator - given a circuit and the initial quantum state it is meant to act on
#                    it returns the fitness value of that circuit 
    def fitnessCalculator(circuit):
        fitness = 0
        #First it will go through the circuit and apply the gates onto the initial quantum state
        #this will create the final quantum state
        quantumState = run_circuit(circuit)
        #Second, it will check to see if the final quantum state contains what we want
        #using the swap test check method - that method will return a value, which is ultimately
        #what will be returned
        fitness = count_pair_two_register(quantumState)
        #Third, it can apply any other fitness criteria methods
    
        return fitness
    
#fitnessList - given a list of circuits and the initial quantum state they are meant to act on
#              it returns a list of the fitness values of each of those circuits
    def fitnessList(circuitList):
        fitnessList = []
        for i in range(len(circuitList)):
            fitnessList.append(fitnessCalculator(circuitList[i]))
        return fitnessList

#swap_test_check - given a quantum state, it will check to see if any pairs have been swapped
#                  to determine the fitness of the state prepared by the circuit
    def swap_test_check(quantumState):
        ancList = []
        LoL = []
        pairs = []
        fitness = 0
    
        #getting ancilla categories
        for i in range(len(quantumState)):
            currBase = []
            currList = []
            for j in range(numAncillas):
                currBase.append(quantumState[i][j])
            if currBase in ancList:
                continue
            else:
                currIn = []
                for j in range(numAncillas, numAncillas+numInputs):
                    currIn.append(quantumState[i][j])
                currList.append(currIn)
                for j in range(i+1, len(quantumState)):
                    activate = True
                    for k in range(numAncillas):
                        if currBase[k] != quantumState[j][k]:
                            activate = False
                            break
                    if activate:
                        currIn = []
                        for k in range(numAncillas, numAncillas+numInputs):
                            currIn.append(quantumState[j][k])
                        currList.append(currIn)
                LoL.append(currList)
                ancList.append(currBase)
    
    #going through and checking all the different ancilla bases
        for i in range(len(LoL)):
            if len(LoL[i]) == 2:
                for j in range(len(LoL[i][0])):
                    for k in range(j+1, len(LoL[i][0])):
                        if swapSinglet([j,k], LoL[i][0]) == LoL[i][1] and pairCheck(LoL[i][0][j], LoL[i][0][k], pairs):
                            fitness += 1
                            pairs.append([LoL[i][0][j], LoL[i][0][k]])
        return fitness

#pairCheck - takes in two variables and a list of pairs of variables, it returns true if 
#            the list of pair values does not contain a pair in which both var1 and var2 are
#            the pair components - it basically checks to see if the two variable pair has 
#            been checked in that list 
    def pairCheck(var1, var2, pairs):
        notChecked = True
        for i in range(len(pairs)):
            if var1 in pairs[i] and var2 in pairs[i]:
                notChecked = False
        return notChecked

    def count_pair_two_registers_i_j(quantum_state, i, j):
        """
        Count number of pairs that appear on input i and j in quantun_state. Order does not matter
        :param quantum_state:
        :param i:
        :param j:
        :return:
        """
        register_i_j = [{q[i], q[j]} for q in quantum_state]
        return len({frozenset(el) for el in register_i_j})


    def count_pair_two_register(quantum_state):
        """
        Count number of pairs that appear on input 0 and 1 in quantun_state . Order does not matter
        :param quantum_state:
        :return:
        """
        return count_pair_two_registers_i_j(quantum_state, numAncillas, numAncillas + 1)


    #***NATURAL SELECTION / GROWTH METHODS***#
    #selection - this method selects which circuits out of the current population will be used
    #            in the next generation of circuits
    def selection(fitnessList,circuitList):
        step = 5
        nextGen = []
        for i in range(0,len(fitnessList),step):
            maxFit=fitnessList[i]
            maxCirc = circuitList[i]
            for j in range(1,step):
                if fitnessList[i+j] > maxFit:
                    maxFit=fitnessList[i+j]
                    maxCirc = circuitList[i+j]
            nextGen.append(maxCirc)
        return nextGen

#crossover - takes in a circuit list called selected to denote the circuits 
#            chosen through natural selection, and returns a new circuit population
#            having applied crossover techniques to make new circuits
    def crossover(selected):
        newGen = selected.copy()
        babySize = genSize-len(selected)
        count = 0
        for i in range(babySize):
            newCircuit = []
            point = r.randint(1,numQubits-1)
            for j in range(len(selected[count])):
                if j<=point:
                    newCircuit.append(selected[count][j])
                else:
                    newCircuit.append(selected[count+r.randint(0,len(selected)-count-1)][j])
            newGen.append(newCircuit)
            if count == len(selected)-1:
                count = 0
            else:
                count+= 1
        return newGen

#mutate - takes in a circuit list called nextGen to denote the next generation of circuits 
#         returns a circuit list with modified circuits having mutation applied to them
#         at low probabilitiy rates
    def mutate(nextGen):
        for i in range(len(nextGen)):
            mutationNum = r.randint(0,100)
            if mutationNum<=mutationChance:
                swappedGate = r.randint(numAncillas, len(nextGen[i])-2)
                randGate = randomGate()
                nextGen[i][swappedGate] = randGate
        return nextGen

#criteriaCheck - takes in a list of fitness values and returns the index which has 
#                the greatest value 
    def criteriaCheck(fitnessList):
        maxFit=fitnessList[0]
        maxIndex = 0
        for i in range(len(fitnessList)):
            if fitnessList[i] > maxFit:
                maxFit=fitnessList[i]
                maxIndex = i
        if maxFit>= terminationCriteria:
            return True, maxIndex, maxFit
        else:
            return False, maxIndex, maxFit
    
    #This returns a working circuit given the input parameters defined at the top
    def main_method():
        #***MAIN METHOD***#
        #Initial Iteration
        # 0. Track number of iterations
        iterationNum = 1
        # 1. Create initial population of circuits
        initialCircuitPop = initialPopulation()
        # 2. Find the fitness for each of the circuit's in the initial population
        fitList = fitnessList(initialCircuitPop)
        #CHECK - to see if max fitness meets termination criteria, if it does, end the program, if not, continue
        finished = criteriaCheck(fitList)
        maxCircuit = finished[1]
        done = finished[0]
        print(finished[2])
        #Algorithm Loop
        while done == False and iterationNum< numIteration:
            #print('iteration: '+str(iterationNum))
            iterationNum+=1
            #THIS WILL CONTINUE RUNNING UNTIL CONVERGENCE IS ACHIEVED
            #FIRST, it will create the next generation of circuits
            #3. Select the part and parcels of the next generation through a selection method
            selectedList = selection(fitList,initialCircuitPop)
            #4. Apply crossover technique to generate next generation of circuits
            nextGen = crossover(selectedList)
            #5. Apply mutation to the nextgeneration
            newGenMutated = mutate(nextGen)
            #SECOND, it will test to see whether the next generation contains the circuit we want
            fitList = fitnessList(newGenMutated)
            finished = criteriaCheck(fitList)
            maxCircuit = finished[1]
            done = finished[0]
            print(finished[2])
            if done:
                print(newGenMutated[maxCircuit])
                print(f'Number of cswap is {count_cswap_gates(newGenMutated[maxCircuit])}')
                return newGenMutated[maxCircuit], done
            initialCircuitPop = newGenMutated
        return initialCircuitPop[maxCircuit], done

    return main_method()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inputs', type=int, default=8)
    parser.add_argument('--num_ancillas', type=int, default=6)
    parser.add_argument('--depth', type=int, default=9)
    parser.add_argument('--mutation_rate', type=int, default=20)
    parser.add_argument('--num_iterations', type=int, default=500)
    parser.add_argument('--gen_size', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', default='test')

    args = parser.parse_args()

    circuit, found = genetic_search(args.depth,
                   args.num_ancillas,
                   args.num_inputs,
                   genSize=args.gen_size,
                   mutationChance=args.mutation_rate,
                   numIteration=args.num_iterations,
                   seed=args.seed)

    print(f'Search returns: {found}')
    print(f'Circuit is of depth {count_cswap_gates(circuit)}')

    tstamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = '/scratch/xgitiaux/quantdist/genetic_search'
    save_dir = f'{save_dir}/run_{args.run}_{args.depth}_{args.num_inputs}_{args.num_ancillas}'
    os.makedirs(save_dir, exist_ok=True)

    results = {'succes': found, 'circuit': circuit}
    with open(f'{save_dir}/results_{tstamp}.json', 'w') as file:
        json.dump(results, file)

