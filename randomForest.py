from sklearn.ensemble import RandomForestClassifier
import classifier
import matplotlib.pyplot as plt
for k in [1, 3, 5, 7, 10, 20, 30]:
    score = []
    num_of_features = [5,7,15,20,25,30,35,40,45,50,70,100,150]
    for num in num_of_features:
        samples, labels, test_points, test_points_labels = classifier.train_and_test(num)
        s = RandomForestClassifier(max_depth=k)
        s.fit(samples, labels)
        s.predict(test_points)
        score.append(s.score(test_points, test_points_labels))
        print s.score(test_points, test_points_labels)

    plt.plot(num_of_features, score, 'orange')
    plt.xlabel('num of features')
    plt.ylabel('score')
    plt.show()