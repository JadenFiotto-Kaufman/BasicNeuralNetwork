from csv import *
from random import shuffle
from functools import reduce
from NeuralNetwork import NeuralNetwork

Data = None
with open('seeddata.csv', newline='') as csvfile:
    rows = reader(csvfile, delimiter=' ')
    Data = [row[0].split(',') for row in rows]

NN = NeuralNetwork()
Network = NN.Initialize(Data)

for i in range(1, 10000):
    shuffle(Data)
    sum_error = 0
    for data in Data:
        NN.InitializeInputs(data, Network)
        NN.FowardPropagate(Network)
        expected = NN.Expected(data[-1])

        ####################################################################
        outputs = [node.Activation for node in Network[-1].Neurons]
        outputLayer = Network[-1].Neurons
        maxindex = reduce(lambda x, y: x if outputLayer[x].Activation > outputLayer[y].Activation else y,range(len(outputLayer)))
        if maxindex == NN.OutcomeToIndex[data[-1]]:
            sum_error += 1
        # sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        #############################################################################
        NN.BackwardPropagate(expected, Network)
        NN.Train(Network)

    print(sum_error / len(Data) * 100)
