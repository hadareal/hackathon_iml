from sklearn import tree
import parser
import Clusterer
def cart():
    data, labels = parser.get_normalized_data(label_size=1)
    c = Clusterer.clusterer()
    data = c.matrix_feature_extracter(data)
    train_data, test_data, train_labels, test_labels = parser.k_split_train_test(data, labels, 90.0)
    clf = tree.DecisionTreeClassifier()
    # clf.n_outputs_ = 20
    clf.fit(train_data, train_labels)
    p = clf.predict(test_data)
    print tree.DecisionTreeClassifier.score(clf, test_data, test_labels)

