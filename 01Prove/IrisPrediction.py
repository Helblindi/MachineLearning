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


def run_gaussian_nb(train, test):
    # Part 3: Use an existing algorithm to create a model
    classifier = GaussianNB()
    model = classifier.fit(train[:, :-1], train[:, -1])

    # Part 4: Use model to make predictions
    targets_predicted = model.predict(test[:, :-1])

    accuracy = get_accuracy(targets_predicted, test[:, -1])
    print("GaussianNB Accuracy: ", accuracy)


def run_hard_coded_classifier(train, test):
    # Part 5: Implement own  new Algorithm
    classifier = HardCodedClassifier()
    model = classifier.fit(train[:, :-1], train[:, -1])
    hc_targets_predicted = model.predict(test[:, :-1])

    hc_accuracy = get_accuracy(hc_targets_predicted, test[:, -1])
    print("Hard Coded Classifier Accuracy: ", hc_accuracy)


def main():
    iris = datasets.load_iris()

    # put the data and the target values together
    data_set = np.column_stack((iris.data, iris.target))

    # Get the train data and the test data
    train_data, test_data = train_test_split(data_set, train_size=0.7, test_size=0.3)

    # Display available options to the user
    print("Available predictive methods:")
    print("Enter 1 for GaussianNB")
    print("Enter 2 for HardCodedClassifier method")
    print()
    choice = input("Which predictive method would you like to use:")

    # Execute a function dependent on user input
    if choice == "1":
        run_gaussian_nb(train_data, test_data)
    elif choice == "2":
        run_hard_coded_classifier(train_data, test_data)
    else:
        print("Invalid input.")


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()
