from sklearn import datasets
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier


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


class KNNModel:
    def __init__(self, data, targets):
        self.k = 5
        self.data = data
        self.targets = targets

    def predict(self, test_data):
        # Predict each element in the data set
        targets = []
        for element in test_data:
            predicted_target = self.predict_one(element)
            targets.append(predicted_target)

        return targets

    def predict_one(self, test_element):
        # Predict each individual element
        test_element_size = len(test_element)
        training_data_size = len(self.data)
        distance_list = []

        # find euclidean distance
        for i in range(training_data_size):
            distance = 0.0
            for j in range(test_element_size):
                distance += (test_element[j] - self.data[i][j]) ** 2
            distance_list.append((distance, self.targets[i]))

        # Sort the distance list
        sorted_distance_list = sorted(distance_list, key=itemgetter(0))

        # Get k nearest neighbors
        nearest_neighbors = sorted_distance_list[:self.k]

        # Find most common neighbor type
        types_of_nearest_neighbors = []
        for neighbor in nearest_neighbors:
            types_of_nearest_neighbors.append(neighbor[1])

        # Find the predicted type
        predicted_type = max(types_of_nearest_neighbors, key=types_of_nearest_neighbors.count)

        return predicted_type


class KNNClassifier:

    def fit(self, data, targets):
        return KNNModel(data, targets)


def main():
    # Load the data set
    iris = datasets.load_iris()
    data_set = iris.data
    target_set = iris.target

    # Get the train data and the test data
    data_train, data_test, targets_train, targets_test = train_test_split(
        data_set,
        target_set,
        train_size=0.7,
        test_size=0.3
        )

    # instantiate our KNNClassifier and get our model
    classifier = KNNClassifier()
    model = classifier.fit(data_train, targets_train)

    # get predicted targets
    knn_targets_predicted = model.predict(data_test)
    custom_algorithm_accuracy = get_accuracy(knn_targets_predicted, targets_test) * 100.0
    print("Custom Algorithm was %.3f percent accurate." % custom_algorithm_accuracy)

    # Compare with an existing algorithm
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(data_train, targets_train)
    knn_existing_algorithm_predictions = model.predict(data_test)
    existing_algorithm_accuracy = get_accuracy(knn_existing_algorithm_predictions, targets_test) * 100.0
    print("Existing Algorithm was %.3f percent accurate." % existing_algorithm_accuracy)

    # Determine the better algorithm
    if custom_algorithm_accuracy > existing_algorithm_accuracy:
        print("Custom algorithm was superior!")
    elif existing_algorithm_accuracy > custom_algorithm_accuracy:
        print("You lose. Good day sir.")
    else:
        print("They are one in the same.")


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()