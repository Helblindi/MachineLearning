import numpy as np
import copy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, has_bias=True, bias_value=-1):
        self.weights = copy.copy(weights)
        self.has_bias = has_bias
        self.bias_value = bias_value

    def simulate(self, inputs):
        sigma = 0
        if np.shape(self.weights)[0] != \
           (np.shape(inputs)[0] + int(self.has_bias)):
            raise IndexError("Input size not equal to weights size")
        if self.has_bias:
            inputs = np.insert(inputs, 0, self.bias_value)
        self.last_inputs = inputs
        for weight, value in zip(self.weights, inputs):
            sigma += weight * value
        self.last_result = sigmoid(sigma)
        return self.last_result

    def new_weights(self, weights):
        # return a shallow copy of the weights and put it into an array
        self.new_weights = np.array(copy.copy(weights))

    def replace_weights(self):
        if not hasattr(self, "new_weights"):
            raise RuntimeError("no new weights to replace with")
        self.weights = np.array(copy.copy(self.new_weights))
        self.new_weights = None
        del self.new_weights

    def __repr__(self):
        i = 0 + (not self.has_bias)
        ret_val = ""
        for weight in self.weights:
            ret_val += "Weight {}: {}\n".format(i, weight)
            i += 1
        return ret_val
