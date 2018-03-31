from Datasets.data_getter import *
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


"""
Select 3 different data sets of your choice.
For each data set:
- Try at least 3 different "regular" learning algorithms and note the results.
- Use Bagging and note the results. (Play around with a few different options)
- Use AdaBoost and note the results. (Play around with a few different options)
- Use a random forest and note the results. (Play around with a few different options)
"""


def get_accuracy(predicted, expected):
    """
    :param predicted:
    :param expected:
    :return: accuracy of  method
    """
    incorrect = 0
    total = len(expected)
    for i in range(total):
        if predicted[i] != expected[i]:
            incorrect += 1
    return (total - incorrect) / total


def main():
    # Display available options to the user
    print("Available datasets:")
    print("Enter 1 for Iris Data Set")
    print("Enter 2 for Pima Indians Diabetes")
    print("Enter 3 for UCI Car Evaluation")
    print()
    choice = input("Which predictive method would you like to use:")

    # Execute a function dependent on user input
    if choice == "1":
        iris = datasets.load_iris()
        data_set = iris.data
        target_set = iris.target
    elif choice == "2":
        data_set, target_set = get_data_pima_indians_diabetes()
    elif choice == "3":
        data_set, target_set = get_data_uci_car_evaluation()
    else:
        print("Invalid input.")
        exit()

    data_train, data_test, target_train, target_test = \
        train_test_split(data_set,
                         target_set,
                         test_size=0.3,
                         train_size=0.7,
                         shuffle=True)

    # kNN
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(data_train, target_train)
    knn_existing_algorithm_predictions = model.predict(data_test)
    knn_accuracy = get_accuracy(knn_existing_algorithm_predictions, target_test) * 100.0
    print("kNN accuracy: ", knn_accuracy)

    # Neural Network
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5, 2),
                               random_state=1)
    model = classifier.fit(data_train, target_train)
    neural_net_predictions = model.predict(data_test)
    neural_net_accuracy = get_accuracy(neural_net_predictions, target_test) * 100.0
    print("Neural Network accuracy: ", neural_net_accuracy)

    # stochastic gradient descent
    classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    model = classifier.fit(data_train, target_train)
    sgd_predictions = model.predict(data_test)
    sgd_accuracy = get_accuracy(sgd_predictions, target_test) * 100.0
    print("Stochastic Gradient Descent accuracy: ", sgd_accuracy)

    # bagging
    classifier = BaggingClassifier(KNeighborsClassifier(),
                                   max_samples=0.5,
                                   max_features=0.5)
    model = classifier.fit(data_train, target_train)
    bagging_predictions = model.predict(data_test)
    bagging_accuracy = get_accuracy(bagging_predictions, target_test) * 100.0
    print("Bagging accuracy: ", bagging_accuracy)

    # random forest
    classifier = RandomForestClassifier(n_estimators=10)
    model = classifier.fit(data_train, target_train)
    rf_predictions = model.predict(data_test)
    rf_accuracy = get_accuracy(rf_predictions, target_test) * 100.0
    print("Random Forest accuracy: ", rf_accuracy)

    # adaboost
    classifier = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(classifier, data_set, target_set)
    print('Adaboost: ', scores.mean() * 100)

    return


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()
