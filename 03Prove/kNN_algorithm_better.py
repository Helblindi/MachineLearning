from sklearn import datasets
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


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


def get_data_UCI_Car_Evaluation():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    df = pd.read_csv("UCI_Car_Evaluation.csv", header=None, names=headers, index_col=False)

    # replace values in all columns using a dictionary (not currently working)
    #cleanup_nums = {"buying": {"v-high": 4, "high": 3, "med": 2, "low": 1}}
    #df.replace(to_replace=cleanup_nums, inplace=True)

    # replace each column 1 by 1
    df['buying'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4, 3, 2, 1], inplace=True)
    df['maint'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4, 3, 2, 1], inplace=True)
    df['doors'].replace(to_replace=['2', '3', '4', '5more'], value=[1, 2, 3, 4], inplace=True)
    df['persons'].replace(to_replace=['2', '4', 'more'], value=[1, 2, 3], inplace=True)
    df['lug_boot'].replace(to_replace=['small', 'med', 'big'], value=[1, 2, 3], inplace=True)
    df['safety'].replace(to_replace=['low', 'med', 'high'], value=[1, 2, 3], inplace=True)

    return df.values


def main():
    get_data_UCI_Car_Evaluation()


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()