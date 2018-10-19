import pandas as pd
import scipy as sc
import scipy.stats
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


def calc_info_gain(data, classes, feature):
    # Get data for this specific feature
    feature_data = data[:, feature]

    # Get unique values
    values = list(set(feature_data))

    # Find the entropy for each value
    entropy_list = []
    for value in values:
        occurrences = []
        for i in range(len(classes)):
            if feature_data[i] == value:
                occurrences.append(classes[i])

        entropy_list.append(calc_entropy(occurrences))

    # Find entropy of feature
    feature_entropy = 0.0
    num_classes = len(classes)
    for i in range(len(entropy_list)):
        num_occurrences = list(feature_data).count(values[i])
        feature_entropy += (num_occurrences / num_classes) * entropy_list[i]

    # Calculate the information gain
    classes_entropy = calc_entropy(classes)
    information_gain = classes_entropy - feature_entropy
    return information_gain


# https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
# Input a target vector
def calc_entropy(data):
    # convert 1d array to a pandas series
    data = pd.Series(data)
    p_data = data.value_counts()/len(data)  # calculates the probabilities
    # defaults to natural log for calculating entropy, change base to 2
    entropy = sc.stats.entropy(p_data, base=2)  # input probabilities to get the entropy
    print('entropy: ', entropy)
    return entropy


def make_tree(data, classes, feature_names):
    # initialisations
    n_data = len(data)
    n_features = len(feature_names)

    # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
    (classes_values, classes_counts) = np.unique(classes, return_counts=True)
    ind = np.argmax(classes_counts)
    default = classes_values[ind]

    if n_data == 0 or n_features == 0:
        return default
    elif ind == len(classes):
        # only one class remains
        return default
    else:
        # choose which feature is best
        gain = np.zeros(n_features)
        for feature in range(n_features):
            gain[feature] = calc_info_gain(data, classes, feature)
        best_feature = np.argmax(gain)
        tree = {feature_names[best_feature]:{}}
        # Find the possible feature values
        (best_feature_values, best_feature_counts) = np.unique(best_feature, return_counts=True)
        for value in best_feature_values:
            for data_point in data:
                if data_point[best_feature] == value:
                    if best_feature == 0:
                        data_point = data_point[1:]
                        new_names = feature_names[1:]
                    elif best_feature == n_features:
                        data_point = data_point[:-1]
                        new_names = feature_names[:-1]
                    else:
                        data_point = data_point[:best_feature]
                        data_point.extend(data_point[best_feature+1:])
                        new_names = feature_names[:best_feature]
                        new_names.extend(feature_names[best_feature+1:])
                    newData.append(data_point)
                    newClasses.append(classes[index])
                index += 1
            # now recurse to the next level
            subtree = make_tree(newData, newClasses, new_names)
            # And on returning, add the subtree on to the tree
            tree[feature_names[best_feature]][value] = subtree
        return tree


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

    # return the dataframe
    return df


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

    # return the dataframe
    return df


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

    # return the dataframe
    return df


def get_data_movie_profit():
    headers = ["type", "plot", "stars", "profit"]
    df = pd.read_csv("Datasets/Movie_Profit.csv",
                     header=None,
                     names=headers,
                     index_col=False)

    # return the dataframe
    return df


def main():
    # Need to change the working directory to access the datasets folder
    #df = get_data_movie_profit()
    df = get_data_uci_car_evaluation()
    data = df.values[:, :-1]
    targets = df.values[:, -1]
    # get a list of the headers from the dataframe
    feature_names = list(df)
    feature_names = feature_names[:-1]

    info_gainz = []
    for i in range(len(data[0])):
        gainz = calc_info_gain(data, targets, i)
        info_gainz.append(gainz)

    print(info_gainz)

    make_tree(data, targets, feature_names)

    return 0


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()


# Worked with Josh Backstein and Zach Benning

# Notes
"""
How to decide the root of the tree:
    - find the outcome if you were to split based on each attribute
    - Use the one that split the data the best.
Entropy: Measure of how mixed the set is; impurity.

Computing entropy of a "set" of data:
    - H(S) = - Sum_(of all x in classes) of p(x)log_2(p(c))
    - Case with classes A and B | p(A) = 3/5, p(B) = 2/5
        + -(p(A)log_2(p(A)) + p(B)log_2(p(B)))
        + -p(A)log_2(p(A)) - p(B)log_2(p(B))
        + -(3/5)log_2(3/5) - (2/5)log_2(2/5)

Computing information gain:
    - What is the entropy before split as opposed to after? 
"""