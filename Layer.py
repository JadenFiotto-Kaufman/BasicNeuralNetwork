from random import random
from Node import *

class Layer:

    def __init__(self, Length, **kwargs):
        self.Bias = random()
        if kwargs is not None and "initialization" in kwargs:
            self.Neurons = [Node().Initialize(kwargs["initialization"]) for index in range(Length)]
        else:
            self.Neurons = [Node() for index in range(Length)]


    def __len__(self):
        return len(self.Neurons)