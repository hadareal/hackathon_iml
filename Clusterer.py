import parser
from random import choice
import numpy as np
import matplotlib.pyplot as plt
import math

ITERS = 10


def bin_derivative(v):
    return [0 if v[i] == v[i + 1] else 1 for i in range(len(v) - 1)]


def num_of_occurences(string, substring):
    return string.count(substring)


def excpected_num_of_occurences(string_len, substring):
    s = 0
    for i in range(ITERS):
        string = ''.join(choice(['0', '1']) for i in xrange(string_len))
        cur = num_of_occurences(string, substring)
        s += cur
    return s / ITERS


def stats(data, str):
    r = []
    for i in range(len(data)):
        n = float(num_of_occurences(data[i], str))
        m = float(excpected_num_of_occurences(len(data[i]), str))
        if n == 0: n += 1
        r.append(m / n)
        # plt.plot(np.arange(iters+1),lossses, 'r-')
        # plt.title('Gradient descent 0-1 loss')
        # plt.ylabel('0-1 loss')
        # plt.xlabel('Iteration')
        #
        # # plt.scatter()
        # plt.show()
        #
        # plt.scatter


def main():
    s = ''.join(choice(['0', '1']) for i in xrange(20))
    print s
    print bin_derivative(s)

    #
    # data = parser.get_data_from_file()
    # stats(data, "001")


if __name__ == "__main__":
    main()
