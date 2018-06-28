# hi

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import parser
from ex4_tools import DecisionStump, decision_boundaries
import math

TRAIN_PERCENTAGE = 90.0

def svm(samples, labels):
    """
    Create svm classifier and train it on the samples
    :param samples:
    :param labels:
    :return: svm classifier
    """

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

def adaboost(samples, labels):
    """

    :param samples:
    :param labels:
    :return:
    """
    ada = AdaBoostClassifier()

data,labels = parser.get_normalized_data()
samples, test_points, labels, test_points_labels = parser.split_train_test(data, labels, TRAIN_PERCENTAGE)
svm = SVC(kernel='sigmoid')
svm.fit(samples, labels)
print(svm.score(test_points, test_points_labels))



