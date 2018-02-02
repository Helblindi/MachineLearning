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


def calc_info_gain_book(data,classes,feature):
    gain = 0
    nData = len(data)
    # List the values that feature can take
    values = []
    for datapoint in data:
        if datapoint[feature] not in values:
            values.append(datapoint[feature])

    featureCounts = np.zeros(len(values))
    entropy = np.zeros(len(values))
    valueIndex = 0

    # Find where those values appear in data[feature] and the corresponding class
    for value in values:
        dataIndex = 0
        newClasses = []
        for datapoint in data:
            if datapoint[feature] == value:
                featureCounts[valueIndex] += 1
                newClasses.append(classes[dataIndex])
            dataIndex += 1

        # Get the values in newClasses
        classValues = []
        for aclass in newClasses:
            if classValues.count(aclass) == 0:
                classValues.append(aclass)

        classCounts = np.zeros(len(classValues))
        classIndex = 0
        for classValue in classValues:
            for aclass in newClasses:
                if aclass == classValue:
                    classCounts[classIndex] += 1
            classIndex += 1

        for classIndex in range(len(classValues)):
            entropy[valueIndex] += calc_entropy(float(classCounts[classIndex]) / sum(classCounts))
            print(classCounts[classIndex])
            #print(calc_entropy(float(classCounts[classIndex]) / sum(classCounts)))
        gain += float(featureCounts[valueIndex]) / nData * entropy[valueIndex]
        valueIndex += 1

    return gain


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
    # defaults to natural log for calculating entropy
    entropy = sc.stats.entropy(p_data, base=2)  # input probabilities to get the entropy
    return entropy


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
    df = get_data_movie_profit()
    data = df.values[:, :-1]
    targets = df.values[:, -1]

    info_gainz = []
    for i in range(len(data[0])):
        gainz = calc_info_gain(data, targets, i)
        print(gainz)
        info_gainz.append(gainz)

    print(info_gainz)

    return 0


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()


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