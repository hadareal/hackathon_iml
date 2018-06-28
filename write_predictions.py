import numpy as np


def calc_error(labels, predicts):
    errors = np.zeros((len(labels, )))
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != predicts[i][j]:
                errors[i] += 1.0 / 2**(i+1)
    err = np.mean(errors)
    print(err)
    print(labels, predicts)


def print(predicts):
    f = open("prediction.txt", "w")
    for j in predicts:
        for i in j:
            f.write(str(i))
        f.write("\n")
    f.close()
