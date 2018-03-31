import random
import sys
from NeuralNet.Neuron import *
from sklearn.metrics import accuracy_score


class Perceptron:
    # Default init settings create a single layer perceptron
    def __init__(self,
                 input_size,
                 hidden_layers=0,
                 hidden_nodes=0,
                 output_nodes=1,
                 use_bias=True,
                 bias_value=-1):
        self.layers = []
        self.input_size = input_size
        node_num_is_list = False
        if type(hidden_nodes) is not int:
            node_num_is_list = True
            node_num_list = hidden_nodes
        for _i in range(hidden_layers):

            if node_num_is_list:
                hidden_nodes = node_num_list[_i]
                if _i != 0:
                    input_size = node_num_list[_i - 1]

            self.layers.append(NeuralNetLayer(hidden_nodes,
                                               input_size,
                                               use_bias,
                                               bias_value))
            input_size = hidden_nodes

        self.output_layer = NeuralNetLayer(output_nodes,
                                            input_size,
                                            use_bias,
                                            bias_value)

    def __repr__(self):
        index = 1
        ret_val = ""

        for layer in self.layers:
            ret_val += "Layer {}:\n{}".format(index, layer)
            index += 1
        ret_val += "Output Layer:\n{}".format(self.output_layer)

        return ret_val

    def simulate(self, input_values):
        if np.shape(input_values)[0] != self.input_size:
            raise ValueError("input size does not match perceptron")

        for layer in self.layers:
            input_values = layer.simulate(input_values)
        return self.output_layer.simulate(input_values)


class NeuralNetLayer:
    def __init__(self,
                 number_of_nodes,
                 input_size,
                 use_bias=True,
                 bias_value=-1):
        self.usingBias = use_bias
        weights = np.zeros(input_size + int(use_bias))
        nodes = []
        for _i in range(number_of_nodes):
            for j in range(input_size + int(use_bias)):
                weights[j] = random.uniform(-1, 1)
            nodes.append(Neuron(weights, use_bias, bias_value))
        self.nodes = np.array(nodes)

    def __repr__(self):
        ret_val = "Using bias node: {}\n".format(self.usingBias)
        i = 0
        for node in self.nodes:
            ret_val += "Node {}:\n{}\n".format(i, node)
            i += 1
        return ret_val

    def simulate(self, inputs):
        results = []
        for node in self.nodes:
            results.append(node.simulate(inputs))
        return np.array(results)


class NeuralNetModel:
    def fit(self,
            training_data,
            training_targets,
            hidden_layers=0,
            hidden_nodes=0,
            output_nodes=1,
            use_bias=True,
            bias_value=-1,
            learning_rate=0.1,
            patience=250,
            p_threshold=0.1,
            max_iterations=1000,
            stop_training=95):
        self.learning_rate = learning_rate
        self.perceptron = Perceptron(np.shape(training_data)[1],
                                     hidden_layers,
                                     hidden_nodes,
                                     output_nodes,
                                     use_bias,
                                     bias_value)
        target_array = np.zeros(output_nodes)
        accuracy_history = []
        for _i in range(max_iterations):
            score = accuracy_score(training_targets, self.predict(training_data))
            accuracy_history.append(score*100)
            sys.stdout.write("\rTraining iteration {} - Accuracy: {:.1f}%".format(_i, score*100))
            sys.stdout.flush()
            best_score = 0
            backup = copy.deepcopy(self.perceptron)
            for row, target in zip(training_data, training_targets):
                self.perceptron.simulate(row)
                target_array[int(target)] = 1
                self.update_weights(target_array)
                target_array[int(target)] = 0

            if score > best_score:
                best_score = score
            if np.std(accuracy_history[-patience:]) < p_threshold and _i > patience:
                print("\nTerminating due to no learning progress")
                break
            if _i > 2:
                if best_score > accuracy_history[-1]:
                    backup = copy.deepcopy(self.perceptron)
            if accuracy_history[-1] >= stop_training:
                print("\nTarget accuracy reached")
                break

        if accuracy_history[-1] < best_score:
            print("Backup model was better than last perceptron, restoring...")
            self.perceptron = backup
        return accuracy_history

    def __repr__(self):
        return self.perceptron

    def update_weights(self, targets):
        perceptron = self.perceptron

        # Update output layer first
        for node, target in \
                zip(perceptron.output_layer.nodes, targets):
            node.error = node.last_result * \
                         (1 - node.last_result) * \
                         (node.last_result - target)
            new_weights = []
            for weight, value in zip(node.weights, node.last_inputs):
                new_weights.append(weight - (self.learning_rate * node.error * value))
            node.new_weights(new_weights)

        # Update remaining layers
        next_layer = perceptron.output_layer
        for layer in perceptron.layers:
            index = 0
            for node in layer.nodes:
                sigma = 0
                for outerNode in next_layer.nodes:
                    sigma += outerNode.weights[index] * outerNode.error

                node.error = node.last_result * (1 - node.last_result) * sigma
                new_weights = []
                for weight, value in zip(node.weights, node.last_inputs):
                    new_weights.append(weight - (self.learning_rate * node.error * value))
                node.new_weights(new_weights)
                index += 1
            next_layer = layer

        for node in perceptron.output_layer.nodes:
            node.replace_weights()

        for layer in perceptron.layers:
            for node in layer.nodes:
                node.replace_weights()

    def predict(self, data):
        if not hasattr(self, "perceptron"):
            raise RuntimeError("model has not been trained")
        results = []
        for row in data:
            row_result = self.perceptron.simulate(row)
            results.append(np.argmax(row_result))
        return np.array(results)
