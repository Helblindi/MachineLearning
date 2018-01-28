from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
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
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    df = pd.read_csv("Datasets/UCI_Car_Evaluation.csv",
                     header=None,
                     names=headers,
                     index_col=False)

    # replace each column 1 by 1
    df['buying'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4.0, 3.0, 2.0, 1.0], inplace=True)
    df['maint'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4.0, 3.0, 2.0, 1.0], inplace=True)
    df['doors'].replace(to_replace=['2', '3', '4', '5more'], value=[1.0, 2.0, 3.0, 4.0], inplace=True)
    df['persons'].replace(to_replace=['2', '4', 'more'], value=[1.0, 2.0, 3.0], inplace=True)
    df['lug_boot'].replace(to_replace=['small', 'med', 'big'], value=[1.0, 2.0, 3.0], inplace=True)
    df['safety'].replace(to_replace=['low', 'med', 'high'], value=[1.0, 2.0, 3.0], inplace=True)
    df['class'].replace(to_replace=['unacc', 'acc', 'good', 'vgood'], value=[1.0, 2.0, 3.0, 4.0], inplace=True)

    # convert the dataframe to a numpy array
    array = df.values

    # return the train data and the target data from the array
    # the first returns all columns but the last column
    # the last returns just the last column
    return array[:,:-1], array[:,-1]


def get_data_pima_indians_diabetes():
    headers = ["num_pregnant", "plasma_glucose_con", "diastolic_bp", "tri_thickness",
               "2hr_serum_insulin", "bmi", "diabetes_pedigree_function", "age", "class"]
    df = pd.read_csv("Datasets/Pima_Indians_Diabetes.txt",
                     header=None,
                     names=headers,
                     index_col=False)

    # replace the null values with the mode of each column
    # returns the mode of each column
    modes = df.mode().values[0]

    df['plasma_glucose_con'].replace(to_replace=[0], value=[modes[1]], inplace=True)
    df['diastolic_bp'].replace(to_replace=[0], value=[modes[2]], inplace=True)
    df['tri_thickness'].replace(to_replace=[0], value=[modes[3]], inplace=True)
    df['2hr_serum_insulin'].replace(to_replace=[0], value=[modes[4]], inplace=True)
    df['bmi'].replace(to_replace=[0], value=[modes[5]], inplace=True)
    df['diabetes_pedigree_function'].replace(to_replace=[0], value=[modes[6]], inplace=True)
    df['age'].replace(to_replace=[0], value=[modes[7]], inplace=True)

    # convert the dataframe to a numpy array
    array = df.values

    # return the train data and the target data from the array
    # the first returns all columns but the last column
    # the last returns just the last column
    return array[:, :-1], array[:, -1]


def get_data_automobile_mpg():
    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight",
               "acceleration", "model_year", "origin", "car_name"]
    df = pd.read_csv("Datasets/Automobile_MPG.txt",
                     header=None,
                     names=headers,
                     na_values='?',
                     delim_whitespace=True,
                     index_col=False)

    # drop the car_name column per it does not appear to contain any valuable information
    df = df.drop("car_name", axis=1)

    df = pd.get_dummies(df, columns=['origin'])

    # drop the rows with NA values as it only accounts for 2% of the data
    df.dropna(inplace=True)

    # convert dataframe to numpy arrrays and return
    target = np.array(df["mpg"])
    data = np.array(df.drop("mpg", axis=1))

    # return the train data and the target data from the array
    return data, target


def main():
    # Display available options to the user
    print("Available datasets:")
    print("Enter 1 for Automobile MPG")
    print("Enter 2 for Pima Indians Diabetes")
    print("Enter 3 for UCI Car Evaluation")
    print()
    choice = input("Which predictive method would you like to use:")

    # Execute a function dependent on user input
    if choice == "1":
        data_set, target_set = get_data_automobile_mpg()
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

    # run our own kNN algorithm on the data
    classifier = KNNClassifier()
    model = classifier.fit(data_train, target_train)
    knn_targets_predicted = model.predict(data_test)
    custom_algorithm_accuracy = get_accuracy(knn_targets_predicted, target_test) * 100.0
    print("Custom Algorithm was %.3f percent accurate." % custom_algorithm_accuracy)

    # Compare with an existing algorithm
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(data_train, target_train)
    knn_existing_algorithm_predictions = model.predict(data_test)
    existing_algorithm_accuracy = get_accuracy(knn_existing_algorithm_predictions, target_test) * 100.0
    print("Existing Algorithm was %i percent accurate." % existing_algorithm_accuracy)

    # Determine the better algorithm
    if custom_algorithm_accuracy > existing_algorithm_accuracy:
        print("Custom algorithm was superior!")
    elif existing_algorithm_accuracy > custom_algorithm_accuracy:
        print("You lose. Good day sir.")
    else:
        print("They are one in the same.")

    # apply CV to data with kerenel = "rbf"
    classifier = svm.SVC(kernel="rbf", C=1)
    scores = cross_val_score(classifier, data_set, target_set, cv=10)

    # print normalization
    print("Radial Basis Function Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))

    # apply CV to data with kerenel = "linear"
    classifier = svm.SVC(kernel="linear", C=1)
    scores = cross_val_score(classifier, data_set, target_set, cv=10)

    # print normalization
    print("Linear Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()

# Collaborated with Zac Benning and Josh Backstein
