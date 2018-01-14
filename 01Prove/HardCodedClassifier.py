import numpy as np


class HardCodedClassifier:
    def fit(self, data, targets):
        return HardCodedModel()


class HardCodedModel:
    def predict(self, test_data):
        targets = np.zeros(len(test_data))
        for target in targets:
            target = 0
        return targets
