import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


class SGD:
    def __init__(self, tau = 0.1, n_iter = 10000, eps = 1e-5):
        self.tau = tau   # параметр регуляризации, контролирует "наказание" за слишком большие веса (L2 - регуляризация)
        self.n_iter = n_iter   # максимум итераций
        self.eps = eps   # параметр остановки


    def predict(self, X):
        return np.dot(X, self.w)


    # функция оценки качесва - чем ближе к 1, тем лучше
    # квадрат коэффициента корреляции Пирсона
    def r2_score(self, y_true, y_pred):
        # так можно делать только в случае линейной регрессии
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)

        cov = np.sum((y_true - mean_true) * (y_pred - mean_pred))
        std_true = np.sqrt(np.sum((y_true - mean_true) ** 2))
        std_pred = np.sqrt(np.sum((y_pred - mean_pred) ** 2))
        
        return (cov / (std_true * std_pred)) ** 2


        # то насколько модель мимо стреляет
        # как плохо предсказала моя модель 
        # u = np.sum((y_true - y_pred) ** 2)   # сумма квадратов остатков (ошибок модели) — RSS 
        # насколько в принципе y разбросаны от среднего
        # как плохо было бы предсказать просто средним значением
        # v = np.sum((y_true - np.mean(y_true)) ** 2)   # сумма квадратов отклонений от среднего — TSS
        # return 1 - u / v


    def loss(self, x_i, y_i):
        err = self.predict(x_i) - y_i 
        cur_loss = err ** 2 + (self.tau / 2) * np.sum(self.w[:-1] ** 2)
        
        return cur_loss


    def fit(self, X, Y):
        m, n = X.shape
        self.w = np.zeros(n)   # начальная инициализация нулями
        lyambda = 0.9   # темп забывания 
        cnt = 0

        # Первая итерация руками для ema_Q начального
        index = np.random.randint(m)
        x_i = X[index]
        y_i = Y[index]
        ema_Q = self.loss(x_i, y_i)


        for i in range(1, self.n_iter + 1):
            index = np.random.randint(m)
            x_i = X[index]
            y_i = Y[index]
            h = 1 / i   # шаг обучения (уменьшается, вначале рывки, затем более тонкие)

            err = self.predict(x_i) - y_i   # невязка
            grad = 2 * err * x_i   # производная от квадратичной ошибки (градиент функции потерь без регуляризации)
            grad[:-1] += self.tau * self.w[:-1]   # L2-регуляризация без фиктивного признака
            self.w -= h * grad   # шаг спуска (обновление весов)
            # print(self.w)

            # curr_loss = self.loss(X, Y)
            curr_loss = self.loss(x_i, y_i)

            # экспоненциальное скользящее среднее для сглаживания кривой ошибки
            ema_Q_new = lyambda * ema_Q + (1 - lyambda) * curr_loss

            if abs(ema_Q_new - ema_Q) < self.eps:
                cnt += 1
                if cnt == 10:
                    break
            else:
                cnt = 0

            ema_Q = ema_Q_new


# подбор оптимального тау
def search_tau(X_train, y_train, X_test, y_test, taus):
    r2_arr = [-1]
    best_weight = None

    for tau in taus:
        model = SGD(tau = tau)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = model.r2_score(y_test, y_pred)

        # print(r2, max(r2_arr))
        if (r2 > max(r2_arr)):
            best_weight = model.w
        r2_arr.append(r2)


    best_tau = taus[np.argmax(r2_arr)]

    print(f"Лучший tau: {best_tau}")
    print(f"R2: {max(r2_arr)}")

    return best_tau, best_weight


def plot_true_vs_pred_simple(y_true, y_pred):
    plt.scatter(y_true, y_pred)
    plt.title("График зависимости между истинными и предсказанными ответами")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth = 2)
    plt.xlabel("Истинные")
    plt.ylabel("Предсказанные")
    plt.grid(True)
    plt.show()



# ------------------------------------------


def main():
    data = load_diabetes()
    X = data.data
    Y = data.target

    X_scaled = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)   # стандартизация
    X_with_fiction = np.hstack([X_scaled, -np.ones((X_scaled.shape[0], 1))])   #обьединение массивов по горизонтали
    # то есть здесь мы приращиваем к нашему X, фиктивный признак забитый -1

    X_train, X_test, y_train, y_test = train_test_split(X_with_fiction, Y, test_size = 0.2, random_state = 42)

    taus = np.linspace(0.001, 1.0, 50)
    best_tau, best_weight = search_tau(X_train, y_train, X_test, y_test, taus)

    model = SGD(tau = best_tau)
    model.w = best_weight
    # model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    plot_true_vs_pred_simple(y_test, y_pred)


# ------------------------------------------


if __name__ == "__main__":
    main()