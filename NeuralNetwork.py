from Node import *
from Layer import *
from math import *

class NeuralNetwork:
    LearningRate = 0.5
    OutcomeToIndex = {}
    IndexToOutcome = {}
    Data = None


    def Initialize(self, Data):

        self.__ToFloat(Data)
        self.__Normalize(Data)

        num_input = len(Data[0][:-1])
        outcomes = set([row[-1] for row in Data])
        num_output = len(outcomes)

        self.__SetOutcomes(list(outcomes))
        
        Network = []
        Network.append(Layer(num_input))
        Network.append(Layer(ceil((len(Network[-1]) + num_output) / 2.0), initialization=len(Network[-1])))
        Network.append(Layer(num_output, initialization=len(Network[-1])))

        return Network

    def __ToFloat(self, Data):
        for row in Data:
            for index in range(len(row[:-1])):
                row[index] = float(row[index])
    def __Normalize(self, Data):
        minmax = [[min(column), max(column)] for column in zip(*Data)]
        for row in Data:
            for index in range(len(row) - 1):
                row[index] = (row[index] - minmax[index][0]) / (minmax[index][1] - minmax[index][0])
    def __SetOutcomes(self, outcomes):

        for index in range(len(outcomes)):
            self.IndexToOutcome[index] = outcomes[index]
            self.OutcomeToIndex[outcomes[index]] = index
    def __TransferDerivitive(self, act):
        return act * (1.0 - act)

    def Expected(self, goal):
        return [1.0 if self.OutcomeToIndex[goal] == index else 0.0 for index in range(len(self.OutcomeToIndex))]
    def BackwardPropagate(self, expected, network):

        outputLayer = network[-1].Neurons
        for index in range(len(outputLayer)):
            node = outputLayer[index]
            node.Error = self.__TransferDerivitive(node.Activation) * (expected[index] - node.Activation)
        for index in reversed(range(1,len(network) - 1)):
            layer = network[index].Neurons
            for iindex in range(len(layer)):
                node = layer[iindex]
                error = 0.0
                for pnode in network[index + 1].Neurons:
                    error += pnode.Weights[iindex] * pnode.Error
                node.Error = self.__TransferDerivitive(node.Activation) * error
    def Train(self, network):

        for index in range(1,len(network)):
            layer = network[index]
            neurons = layer.Neurons
            inputs = [node.Activation for node in network[index - 1].Neurons]
            for node in neurons:
                for iindex in range(len(inputs)):
                    node.Weights[iindex] += self.LearningRate * node.Error * inputs[iindex]
                layer.Bias += self.LearningRate * node.Error

    def FowardPropagate(self, network):
        for index in range(len(network)-1):
            layer = network[index + 1]
            neurons = layer.Neurons
            for node in neurons:
                node.Activate([Input.Activation for Input in network[index].Neurons], layer.Bias)
    def InitializeInputs(self, data, network):
        inputLayer = network[0].Neurons
        for i in range(len(inputLayer)):
            inputLayer[i].Activation = data[i]
