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

    return [t for t in string]



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
    return data, labels, vault


