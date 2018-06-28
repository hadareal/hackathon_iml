import parser
import numpy as np

norm = [0.5**n for n in range(1, 21)]


def check_error(label, prediction):
    a = np.abs(label - prediction)
    a[a != 0] = 1
    return np.dot(a, norm).sum()


def get_data():
    """

    :param string_len:
    :param label_size:
    :return: ndarray of size mXd where d=(string_len-label_size)
    """
    data = parser.get_data_from_file()
    data = data[100:]
    for i in range(1027):
        data[i] = data[i][:-20]
    return data

def check_row(j):
    data1 = get_data()
    row = data1[j]
    labels = row[-20:]
    predict = ''
    ones = row.count('10')
    prob1 = ones/float(len(row))
    for i in range(10):
        new = np.random.binomial(1, prob1, 2)
        n = ''
        for i in new:
            n += str(i)
          predict += str(n)
    labels = np.array([int(i) for i in labels])
    predict = np.array([int(i) for i in predict])
    return check_error(labels, predict)

mean = 0
for i in range(1027):
    a = check_row(i)
    mean += a
    print a

print mean / 1027.0