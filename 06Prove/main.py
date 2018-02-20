from NeuronLayer import *
from Neuron import *
from NeuralNetwork import *
from Datasets.data_getter import *

"""
This Weeks Assignment
5) Be able to take input from a dataset instance (with an arbitrary number of attributes)
   and have each node produce an output (i.e., 0 or 1) according to its weights.

6) Be able to load and process at least the following two datasets:

    - Iris (You didn't think we could leave this out did you!)

    - Pima Indian Diabetes

7) You should appropriately normalize each data set.
"""


# main driver function for the neural network program
def main():
    data, targets = get_data_pima_indians_diabetes()
    n_n = NeuralNetwork(len(data[0]), 1)
    print(n_n.get_weights())
    for datapoint in data:
        print(n_n.update(datapoint))

    print(n_n)
    return 0


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()


# end of main
