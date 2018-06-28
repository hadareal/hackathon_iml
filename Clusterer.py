import parser
from random import choice
import numpy as np
import itertools

D = parser.D
ITERS = 100


class clusterer:
    def __init__(self):
        self.substrings = ['0', '1', '00', '01', '10', '11', '000', '001', '010', '011', '100', '101', '110', '111'
                                                                                                              '0000',
                           '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011',
                           '1100', '1101', '1110', '1111']
        self.expected = [self.excpected_num_of_occurences(D, ss) for ss in self.substrings]

    def bin_derivative(self, v):
        return [0 if v[i] == v[i + 1] else 1 for i in range(len(v) - 1)]

    def ftr(self, v):

        n = len(v)
        features = []
        s_v = ''.join([str(c) for c in v])

        for i, substring in enumerate(self.substrings):
            m = float(self.num_of_occurences(s_v, substring))
            if self.expected[i]==0:self.expected[i]=0.0001
            features.append(m / self.expected[i])
        return features

    def matrix_bin_derivative(self, X):
        """

        :param X: mXd ndarray to be differentiated
        :return: mX(d-1) differentiated ndarray
        """
        m, d = X.shape
        R = [0] * m
        for i in range(m):
            R[i] = self.bin_derivative(X[i])
        return np.stack(R)

    def matrix_feature_extracter(self, X):
        """
        :param X: mXd ndarray to be featured
        :return: mX(d-1) featured ndarray
        """
        m, d = X.shape
        R = [0] * m
        for i in range(m):
            R[i] = self.ftr(X[i])
        return np.stack(R)

    def num_of_occurences(self, string, substring):

        return string.count(substring)

    def excpected_num_of_occurences(self, string_len, substring):
        s = 0.0
        for i in range(ITERS):
            string = ''.join(choice(['0', '1']) for i in xrange(string_len))
            cur = self.num_of_occurences(string, substring)
            s += cur
        return s / ITERS

    def stats(self, data, str):
        r = []
        for i in range(len(data)):
            n = float(self.num_of_occurences(data[i], str))
            m = float(self.excpected_num_of_occurences(len(data[i]), str))
            if n == 0: n += 1
            r.append(m / n)


def main():
    # s = ''.join(choice(['0', '1']) for i in xrange(20))
    # n=100000
    data , labels= parser.get_normalized_data()
    print (data.shape)
    print (data.shape)
    d_data = matrix_bin_derivative(data)
    print (d_data.shape)



    # t = data[5]
    # n = float(len(t))
    # e = []
    # for i in range(20):
    #     e.append(t.count('0')/n)
    #     t = bin_derivative(t)
    #
    # s = [str(random.randint(0,1)) for i in range(int(n))]
    # s = ''.join(s)
    # n = float(len(s))
    # d = []
    # for i in range(20):
    #     d.append(s.count('0')/n)
    #     s = bin_derivative(s)
    #
    # plt.plot(e)
    # plt.show()
    # #

    data, labels = parser.get_normalized_data()
    print data.shape
    c = clusterer()
    d_data = c.matrix_feature_extracter(data)
    print d_data.shape



if __name__ == "__main__":
    main()
