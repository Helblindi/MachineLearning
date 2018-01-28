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


def get_data_uci_car_evaluation():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    df = pd.read_csv("Datasets/UCI_Car_Evaluation.csv",
                     header=None,
                     names=headers,
                     index_col=False)

    # replace values in all columns using a dictionary (not currently working)
    #cleanup_nums = {"buying": {"v-high": 4, "high": 3, "med": 2, "low": 1}}
    #df.replace(to_replace=cleanup_nums, inplace=True)

    # replace each column 1 by 1
    df['buying'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4.0, 3.0, 2.0, 1.0], inplace=True)
    df['maint'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4.0, 3.0, 2.0, 1.0], inplace=True)
    df['doors'].replace(to_replace=['2', '3', '4', '5more'], value=[1.0, 2.0, 3.0, 4.0], inplace=True)
    df['persons'].replace(to_replace=['2', '4', 'more'], value=[1.0, 2.0, 3.0], inplace=True)
    df['lug_boot'].replace(to_replace=['small', 'med', 'big'], value=[1.0, 2.0, 3.0], inplace=True)
    df['safety'].replace(to_replace=['low', 'med', 'high'], value=[1.0, 2.0, 3.0], inplace=True)

    return df.values, np.zeros(1)


def get_data_pima_indians_diabetes():
    headers = ["num_pregnant", "plasma_glucose_con", "diastolic_bp", "tri_thickness",
               "2hr_serum_insulin", "bmi", "diabetes_pedigree_function", "age", "class"]
    df = pd.read_csv("Datasets/Pima_Indians_Diabetes.txt",
                     header=None,
                     names=headers,
                     index_col=False)

    df['plasma_glucose_con'].replace(to_replace=[0], value=[np.nan], inplace=True)
    df['diastolic_bp'].replace(to_replace=[0], value=[np.nan], inplace=True)
    df['tri_thickness'].replace(to_replace=[0], value=[np.nan], inplace=True)
    df['2hr_serum_insulin'].replace(to_replace=[0], value=[np.nan], inplace=True)
    df['bmi'].replace(to_replace=[0], value=[np.nan], inplace=True)
    df['diabetes_pedigree_function'].replace(to_replace=[0], value=[np.nan], inplace=True)
    df['age'].replace(to_replace=[0], value=[np.nan], inplace=True)

    return df.values, np.zeros(1)


def get_data_automobile_mpg():
    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight",
               "acceleration", "model_year", "origin", "car_name"]
    df = pd.read_csv("Datasets/Automobile_MPG.txt",
                     header=None,
                     names=headers,
                     na_values="?",
                     delim_whitespace=True,
                     index_col=False)

    # Need to move mpg to be the last column and drop the car_names
    # column, as it does not provide valuable information
    column_titles = ["cylinders", "displacement", "horsepower", "weight",
                     "acceleration", "model_year", "origin", "mpg"]
    df = df.reindex(columns=column_titles)

    return df.values, np.zeros(1)


def main():
    data_set_car_evaluation, target_set_car_evaluation = get_data_uci_car_evaluation()
    data_set_pima_indians_diabetes, target_set_pima_indians_diabetes = get_data_pima_indians_diabetes()
    data__set_automobile_mpg, target_set_automobile_mpg = get_data_automobile_mpg()


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()