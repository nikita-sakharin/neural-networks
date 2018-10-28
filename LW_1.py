import numpy as np

class PerceptronRosenblatt:
    @staticmethod
    def sum(w: np.array, x: np.array) -> float:
        return w.dot(np.append(1.0, x))

    @staticmethod
    def f(x: float) -> int:
        return np.heaviside(x, 1.0)

    @staticmethod
    def a(w: np.array, x: np.array) -> int:
        return PerceptronRosenblatt.f(PerceptronRosenblatt.sum(w, x))

    def __init__(self):
       self.w = np.array([0.0, 0.0, 0.0], dtype = np.float64)

    def fit(self, x: np.array, y: np.array, eta = 0.25, iteration = 8192):
        i = 0
        while i < len(x):
            if iteration <= 0:
                raise ValueError('Sets is not linearly solvable')
            err = y[i] - PerceptronRosenblatt.a(self.w, x[i])
            if err != 0:
                self.w += eta * np.append(1.0, x[i]) * err
                i = -1
            iteration -= 1
            i += 1

    def b_k(self) -> tuple:
        ret = np.array([0.0, 0.0], dtype = np.float64)
        if abs(self.w[2]) < np.finfo(np.float64).eps:
            ret[0] = -self.w[0] / self.w[1]
            ret[1] = np.finfo(np.float64).max
        else:
            ret[0] = -self.w[0] / self.w[2]
            ret[1] = -self.w[1] / self.w[2]
        return ret

    def predict(self, x: np.array) -> np.array:
        return np.array([PerceptronRosenblatt.a(self.w, x_i) for x_i in x], dtype = np.int32)

import csv
import matplotlib.pyplot as plt
import sys

def task_1(file_name: str):
    data = [tuple(line) for line in csv.reader(open(file_name, 'r'))]
    in_data = np.array([(float(line[0]), float(line[1])) for line in data])
    out_data = np.array([int(line[2]) for line in data])

    plt.subplot(1, 1, 1)
    x_min = min(in_data, key = lambda a: a[0])[0] - 1
    x_max = max(in_data, key = lambda a: a[0])[0] + 1
    y_min = min(in_data, key = lambda a: a[1])[1] - 1
    y_max = max(in_data, key = lambda a: a[1])[1] + 1
    for i in range(len(in_data)):
        plt.plot(in_data[i][0], in_data[i][1], 'o',
            color = ['red', 'green'][out_data[i]])

    perceptron = PerceptronRosenblatt()
    perceptron.fit(in_data, out_data)
    b_k = perceptron.b_k()

    if b_k[1] == np.finfo(np.float64).max:
        plt.plot([b_k[0], b_k[0]], [y_min, y_max], color = 'blue')
    else:
        plt.plot([x_min, x_max], [b_k[0] + b_k[1] * x_min, b_k[0] + b_k[1] * x_max], color = 'blue')
    plt.show()

def task_2(file_name: str):
    data = [tuple(line) for line in csv.reader(open(file_name, 'r'))]
    in_data = np.array([(float(line[0]), float(line[1])) for line in data])
    out_data = np.array([np.array([line[2], line[3]], dtype = np.int32) for line in data])

    plt.subplot(1, 1, 1)
    x_min = min(in_data, key = lambda a: a[0])[0] - 1
    x_max = max(in_data, key = lambda a: a[0])[0] + 1
    y_min = min(in_data, key = lambda a: a[1])[1] - 1
    y_max = max(in_data, key = lambda a: a[1])[1] + 1
    for i in range(len(in_data)):
        plt.plot(in_data[i][0], in_data[i][1], 'o',
            color = ['red', 'green', 'yellow', 'blue'][out_data[i][0] + 2 * out_data[i][1]])

    perceptron = [PerceptronRosenblatt(), PerceptronRosenblatt()]
    for i in range(len(perceptron)):
        perceptron[i].fit(in_data, out_data[:, i])
        b_k = perceptron[i].b_k()
        if b_k[1] == np.finfo(np.float64).max:
            plt.plot([b_k[0], b_k[0]], [y_min, y_max], color = 'grey')
        else:
            plt.plot([x_min, x_max], [b_k[0] + b_k[1] * x_min, b_k[0] + b_k[1] * x_max], color = 'grey')
    plt.show()

def main():
    task_1('test_1_1.csv')
    task_2('test_1_2.csv')

if __name__ == "__main__":
    main()
