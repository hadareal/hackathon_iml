import numpy as np
import math


D=180
LABEL_SIZE = 1


def string_normalize(string, target_length):
    """

    :param string: bit string to normalize
    :param target_length: d
    :return: Normalized string (right length and in list format)
    """

    if len(string) > target_length:
        string = string[len(string) - target_length:]
    elif len(string) < target_length:
        string = string * math.ceil(target_length / len(string))
        return string_normalize(string, target_length)

    return [int(t) for t in string]



def normalize(messy_data, num_of_features):
    """

    :param messy_data: list of strings
    :return: np matrix of size mXd
    """

    for i in range(len(messy_data)):
        messy_data[i] = string_normalize(messy_data[i], num_of_features)
    return np.stack(messy_data, axis=0)


def get_data_from_file():
    """

    :return: list of strings from file
    """
    filename = 'human.txt'
    f = open(filename)
    data = f.readlines() #list of strings of bits
    for i, s in enumerate(data):
        data[i] = s.rstrip()
    return data


def get_normalized_data(num_of_features=D, label_size=LABEL_SIZE):
    """

    :param string_len:
    :param label_size:
    :return: ndarray of size mXd where d=(string_len-label_size)
    """
    data = get_data_from_file()
    data, vault = data[100:], data[:100]
    data = normalize(data, num_of_features)
    data, labels = data[:, :-label_size], data[:, -label_size:]
    print type(data[0][0])

def split_train_test(data, labels, train_per):
    """

    :param data: mxd matrix of m feature samples
    :param labels: list of m labels
    :param train_per: Percentage of data used for training
    :return: trainData, testData, trainLabels, testLabels
    """

    m, d = data.shape
    indices = np.random.permutation(m)
    training_size = int(m * (train_per / 100.0))
    training_idx, test_idx = indices[:training_size], indices[training_size:]
    return data[training_idx, :], data[test_idx, :], np.ravel(labels[training_idx]), np.ravel(labels[test_idx])


def k_split_train_test(data, labels, train_per):
    """

    :param data: mxd matrix of m feature samples
    :param labels: list of m labels
    :param train_per: Percentage of data used for training
    :return: trainData, testData, trainLabels, testLabels
    """

    m, d = data.shape
    indices = np.random.permutation(m)
    training_size = int(m * (train_per / 100.0))
    training_idx, test_idx = indices[:training_size], indices[training_size:]
    return data[training_idx, :], data[test_idx, :], labels[training_idx, :], labels[test_idx, :]

