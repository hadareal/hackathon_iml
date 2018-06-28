import numpy as np
import parser

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


def calc_index(seq):
    num = 0
    for i in range(len(seq)):
        num *= 2
        if seq[len(seq) - 1 - i] == '1':
            num += 1
    return num


# v is the sequence, k is the length of the sequence to predict by
# return a vector length k for each sequence represent number (i.e. k=3, 101 - indexed by 5) the probability of getting
# 1 after that
def produce_predictor(v, k):
    probs = np.zeros((2 ** k, 2))
    for i in range(len(v) - k):
        index = calc_index(v[i:i+k])
        probs[index][0] += 1
        if v[i+k] == '1':
            probs[index][1] += 1
    return probs


def predict(v):
    weights = np.zeros((5, ))
    predictions = np.zeros((5, ))
    for i in range(5):
        probs = produce_predictor(v, i+1)
        # print(probs)
        last_seq_data = probs[calc_index(v[len(v)-i-1:len(v)])]
        # print(last_seq_data)
        weights[i] = last_seq_data[0]
        if last_seq_data[1] * 2 > last_seq_data[0]:
            predictions[i] = 1
        else:
            predictions[i] = -1
    # print(predictions, weights)
    p = np.multiply(weights, predictions)
    # print(p)
    h = np.sum(p)
    if h > 0:
        return 1
    else:
        return 0

data = get_data()
summer = 0
for j in range(len(data)):
    d = data[j]
    v1 = d[:-1]
    l = d[-1]
    t = predict(v1)
    print(t, l)
    if int(l) == t:
        summer += 1
# v1 = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0]
print(summer, len(data), summer / len(data))
