# Raz is my eternal master
# hi

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def classifier(samples,labels, test_points, test_points_labels):

    clf = SVC(C=1e10, kernel='linear')
    clf.fit(samples,labels)
    print(clf.score(test_points, test_points_labels))

samples = np.array([[1,2,1],[1,3,4],[1,2,3]])
labels = [1,1,0]
test_points = np.array([[3,3,4],[3,2,3]])
test_points_labels = [1,0]
classifier(samples,labels,test_points,test_points_labels)