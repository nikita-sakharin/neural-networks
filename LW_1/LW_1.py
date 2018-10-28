import numpy as np

class PerceptronRosenblatt:

    """
        w - [Веса]
    """
    def __init__(self):
        w = np.array([0.0, 0.0, 0.0], dtype = np.float64)

    """
        функция активации
    """
    def __f__(self, x):
        y = w[0] + w[1] * x[1]
        return y > x[1]

    """
        f_i_matrix : Матрица признаков - на позиции (i, j) содержится кол-во вхождений в i-ый документ j-го слова словаря
        c_vector : Вектор классов - на i-ой позиции хранится класс i-го документа
    """
    def fit(self, f_i_matrix, c_vector):

    """
        predict([(1.0, 2.0), (-1.0, 3.0)])
    """
    def predict(self, x):
        return np.array([self.__f__(x_i) for x_i in x], dtype = np.bool)
