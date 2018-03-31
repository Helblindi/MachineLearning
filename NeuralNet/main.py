import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Datasets.data_getter import get_data_pima_indians_diabetes
from NeuralNet.Perceptron import *
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def main():
    # Constants to be passed in to model.fit
    iris_stop = 80
    iris_hidden = 1
    iris_nodes = 3
    iris_rate = 0.15
    iris_patience = 250
    iris_p_threshold = 0.1

    pima_stop = 71
    pima_hidden = 1
    pima_nodes = 2
    pima_rate = 0.2
    pima_patience = 150
    pima_p_threshold = 0.5

    # load iris data set
    iris = load_iris()

    # normalize data
    norm_data = preprocessing.normalize(iris.data)

    # split data using train_test_split
    data_train, data_test, target_train, target_test = \
        train_test_split(norm_data,
                         iris.target,
                         test_size=0.3,
                         train_size=0.7,
                         shuffle=True)

    # Fit the model
    model = PerceptronModel()
    iris_history_h1 = model.fit(data_train,
                                target_train,
                                hidden_layers=iris_hidden,
                                hidden_nodes=iris_nodes,
                                learning_rate=iris_rate,
                                output_nodes=3,
                                patience=iris_patience,
                                p_threshold=iris_p_threshold,
                                stop_training=iris_stop)

    iris_history_h2 = model.fit(data_train,
                                target_train,
                                hidden_layers=iris_hidden + 1,
                                hidden_nodes=iris_nodes,
                                learning_rate=iris_rate,
                                output_nodes=3,
                                patience=iris_patience,
                                p_threshold=iris_p_threshold,
                                stop_training=iris_stop)

    # get accuracy score and print it out
    score = accuracy_score(target_test, model.predict(data_test))
    print("\nIris accuracy: {0:.1f}%".format(score*100))

    # plot a graph of the iterations that were required crossed with the
    # accuracy at each given iteration
    plt.plot(iris_history_h1, 'b-', label='Hidden Layers: 1')
    plt.plot(iris_history_h2, 'g-', label='Hidden Layers: 2')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Iris Final Accuracy: {:.1f}".format(score*100))
    plt.legend(fontsize='large', loc='upper left')
    plt.savefig("irisTrainingProgress_test.png")

    # clear the current figure
    plt.clf()

    # Compare to SKLearn Nueral Net
    classifier = MLPClassifier(solver='lbfgs',
                               alpha=1e-5,
                               hidden_layer_sizes=(2, 3),
                               random_state=1)
    model = classifier.fit(data_train, target_train)
    neural_net_accuracy = accuracy_score(target_test, model.predict(data_test))
    print("Prebuilt Neural Network accuracy: ", neural_net_accuracy)

    # get data for pima and normalize it
    pima_data, pima_target = get_data_pima_indians_diabetes()
    norm_data = preprocessing.normalize(pima_data)

    # split the data using train_test_split
    data_train, data_test, target_train, target_test = \
        train_test_split(norm_data,
                         pima_target,
                         test_size=0.7,
                         train_size=0.3,
                         shuffle=True)

    # Fit the model
    pima_model = PerceptronModel()
    pima_history_h1 = pima_model.fit(data_train,
                                     target_train,
                                     hidden_layers=pima_hidden,
                                     hidden_nodes=pima_nodes,
                                     output_nodes=2,
                                     learning_rate=pima_rate,
                                     patience=pima_patience,
                                     p_threshold=pima_p_threshold,
                                     stop_training=pima_stop)

    pima_history_h2 = pima_model.fit(data_train,
                                     target_train,
                                     hidden_layers=pima_hidden + 1,
                                     hidden_nodes=pima_nodes,
                                     output_nodes=2,
                                     learning_rate=pima_rate,
                                     patience=pima_patience,
                                     p_threshold=pima_p_threshold,
                                     stop_training=pima_stop)

    # get accuracy score and print it out
    score = accuracy_score(target_test, pima_model.predict(data_test))
    print("\nPima accuracy: {0:.1f}%".format(score*100))

    # plot a graph of the iterations that were required crossed with the
    # accuracy at each given iteration
    plt.plot(pima_history_h1, 'b-', label='Hidden Layers: 1')
    plt.plot(pima_history_h2, 'g-', label='Hidden Layers: 2')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Pima Final Accuracy: {:.1f}".format(score*100))
    plt.legend(fontsize='large', loc='upper left')
    plt.savefig("pimaTrainingProgress_test.png")

    # Compare to SKLearn Nueral Net
    classifier = MLPClassifier(solver='lbfgs',
                               alpha=1e-5,
                               hidden_layer_sizes=(1, 2),
                               random_state=1)
    model = classifier.fit(data_train, target_train)
    neural_net_accuracy = accuracy_score(target_test, model.predict(data_test))
    print("Prebuilt Neural Network accuracy: ", neural_net_accuracy)


if __name__ == '__main__':
    main()
