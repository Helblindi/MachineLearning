from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from HardCodedClassifier import *


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


iris = datasets.load_iris()

# Above and beyond
try:
    with open("FlowerData.txt") as f:
        FlowerData = f.read()
except FileNotFoundError:
    FlowerData = None

print(iris)
# Show the data (the attributes of each instance)
# print(iris.data)

# Show the target values (in numeric format) of each instance
# print(iris.target)

# Show the actual target names that correspond to each number
# print(iris.target_names)

# put the data and the target values together
data_set = np.column_stack((iris.data, iris.target))

# Get the train data and the test data
train_data, test_data = train_test_split(data_set, train_size=0.7, test_size=0.3)

# Part 3: Use an existing algorithm to create a model
classifier = GaussianNB()
model = classifier.fit(train_data[:, :-1], train_data[:, -1])

# Part 4: Use model to make predictions
targets_predicted = model.predict(test_data[:, :-1])

accuracy = get_accuracy(targets_predicted, test_data[:, -1])
print("GaussianNB Accuracy: ", accuracy)

# Part 5: Implement own  new Algorithm
classifier = HardCodedClassifier()
model = classifier.fit(train_data[:, :-1], train_data[:, -1])
hc_targets_predicted = model.predict(test_data[:, :-1])

hc_accuracy = get_accuracy(hc_targets_predicted, test_data[:, -1])
print("Hard Coded Classifier Accuracy: ", hc_accuracy)