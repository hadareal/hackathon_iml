# hi

import numpy as np
from sklearn.svm import SVC
import sklearn.ensemble
import sklearn.tree
import matplotlib.pyplot as plt
import parser
import matplotlib.patches as mpatches
import Clusterer
import kClassifier
from ex4_tools import DecisionStump, decision_boundaries
import math

TRAIN_PERCENTAGE = 90.0
ITERATIONS = 50

def svm(samples, labels):
    """
    Create svm classifier and train it on the samples
    :param samples:
    :param labels:
    :return: svm classifier
    """

    svm = SVC(kernel='sigmoid')
    svm.fit(samples, labels)
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
    Create adaboost classifier and train it on the samples
    :param samples:
    :param labels:
    :return:
    """
    ada = sklearn.ensemble.AdaBoostClassifier()
    ada.fit(samples, labels)
    return ada

def draw_plot(x_values, y1_values, y2_values):

    plt.plot(x_values, y1_values, 'g', x_values, y2_values, 'orange')
    plt.xlabel('iteration')
    plt.ylabel('score')
    plt.title('scores of svm and adaboost')
    green = mpatches.Patch(color='g',label='svm classifier')
    orange = mpatches.Patch(color='orange', label='adaboost classifier')
    plt.legend(handles=[orange, green])
    plt.show()

def bagging(samples, lables):
    """
    Create bagging classifier and train it on the samples
    :param samples:
    :param lables:
    :return:
    """
    bagging = sklearn.ensemble.BaggingClassifier()
    bagging.fit(samples,lables)
    return bagging



def main():
    ada_scores = []
    svm_scores = []
    # bagging_scores = []
    c = Clusterer.clusterer()
    for i in range(ITERATIONS):
        data,labels = parser.get_normalized_data()
        data = c.matrix_feature_extracter(data)
        samples, test_points, labels, test_points_labels = parser.split_train_test(data, labels, TRAIN_PERCENTAGE)
        ada = adaboost(samples,labels)
        s = svm(samples, labels)
        b = bagging(samples,labels)
        ada_scores.append(ada.score(test_points, test_points_labels))
        svm_scores.append(s.score(test_points,test_points_labels))
        # bagging_scores.append(b.score(test_points,test_points_labels))
    draw_plot(np.arange(ITERATIONS),svm_scores,ada_scores)

def main2():
    data,tails = parser.get_normalized_data()
    labels = np.ravel(np.split(tails,len(tails[0]),axis=1)[0])
    test_points = data.T
    print(np.shape(test_points))

main()
# kClassifier.cart()