from random import random
from math import exp

class Node:
    Activation = None
    def Initialize(self, num):
        self.__InitializeWeights(num)
        self.Error = None
        return self
    def __InitializeWeights(self, num_weights):
        self.Weights = [random() for index in range(num_weights)]
    def Activate(self, Inputs, Bias):
        self.Activation = self.__Sigmoid(sum(map(lambda x,y : x * y, Inputs,self.Weights)) + Bias)
    def __Sigmoid(self, Sum):
        return 1.0 / (1.0 + exp(-Sum))