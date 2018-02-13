class DecisionTreeClassifier:

    def fit(self, data, targets):
        return DecisionTreeModel(data, targets)


class DecisionTreeModel:
    def __init__(self, data, targets):
        self.k = 5
        self.data = data
        self.targets = targets
