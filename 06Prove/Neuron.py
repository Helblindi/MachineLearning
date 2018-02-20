import random


# definition for our Neuron Class
class Neuron:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.set_weights([random.uniform(-0.5, 0.5) for x in range(0, n_inputs + 1)])  # +1 for bias weight

    def sum(self, inputs):
        # Does not include the bias
        return sum(val * self.weights[i] for i, val in enumerate(inputs))

    def set_weights(self, weights):
        self.weights = weights

    def __str__(self):
        return 'Weights: %s, Bias: %s' % (str(self.weights[:-1]), str(self.weights[-1]))


# end of Neuron Class
