import numpy as np
import matplotlib.pyplot as plt
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



def normalize(messy_data):
    """

    :param messy_data: list of strings
    :return: np matrix of size mXd
    """

    for i in range(len(messy_data)):
        messy_data[i] = string_normalize(messy_data[i], D)
    return np.stack(messy_data, axis=0)


def hist(lens):
    plt.hist(lens, bins=100)
    plt.show()


def get_data():
    filename = 'human.txt'
    f = open(filename)
    data = f.readlines() #list of strings of bits
    for i, s in enumerate(data):
        data[i] = s.rstrip()
    return data

def main():

    data = get_data()
    data, vault = data[100:], data[:100]
    data = normalize(data)
    data, labels = data[:, :-LABEL_SIZE], data[:, -LABEL_SIZE:]

if __name__ == "__main__":
    main()
