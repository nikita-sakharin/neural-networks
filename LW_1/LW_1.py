import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    data = [tuple(line) for line in csv.reader(sys.stdin)]
    in_data = [(float(line[0]), float(line[1])) for line in data]
    out_data = [int(line[2]) for line in data]

    plt.subplot(1, 1, 1)
    color = ['red', 'green']
    for i in range(len(in_data)):
        plt.plot(in_data[i][0], in_data[i][1], 'go', color = color[out_data[i]])
    plt.show()

if __name__ == "__main__":
    main()

class PerceptronRosenblatt:

    """
        w - [Веса]
    """
    def __init__(self):
        w = np.array([1.0, 1.0, 1.0], dtype = np.float64)
#        w = np.array([1.0, 1.0, 1.0], dtype = np.float64)

    def __sum__(self, x) -> float:
        return self.w.dot(np.append(1.0, x));

    def __f__(self, x : float) -> bool:
        return x >= 0;

    """
        f_i_matrix : Матрица признаков - на позиции (i, j) содержится кол-во вхождений в i-ый документ j-го слова словаря
        c_vector : Вектор классов - на i-ой позиции хранится класс i-го документа
    """
    def fit(self, f_i_matrix, c_vector):
        pass

    """
        predict([(1.0, 2.0), (-1.0, 3.0)])
    """
    def predict(self, x):
        return np.array([self.__f__(self.__sum__(x_i)) for x_i in x], dtype = np.bool)
