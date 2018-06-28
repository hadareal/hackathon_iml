# hi

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import parser
from ex4_tools import DecisionStump, decision_boundaries
import math

TEST_SIZE = 150

def classifier(samples, labels):

    svm = SVC(kernel='sigmoid')
    svm.fit(samples,labels)
    print(svm.score(test_points, test_points_labels))
    return svm

def splitTrainTest(data, labels, trainPer):
    """

    :param data: mxd matrix of m feature samples
    :param labels: list of m labels
    :param trainPer: Percentage of data used for training
    :return: trainData, testData, trainLabels, testLabels
    """

    m, d = data.shape
    indices = np.random.permutation(m)
    training_size = int(m * (trainPer / 100.0))
    training_idx, test_idx = indices[:training_size], indices[training_size:]
    return data[training_idx, :], data[test_idx, :], np.ravel(labels[training_idx]), np.ravel(labels[test_idx])

data,labels,vault = parser.get_normalized_data()

samples, test_points, labels, test_points_labels = splitTrainTest(data, labels, 90.0)
svm = SVC(kernel='sigmoid')
svm.fit(samples, labels)
print(svm.score(test_points, test_points_labels))



