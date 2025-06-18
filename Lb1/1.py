import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class NearestNeighbor:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_samples = X.shape[0]

    def distance(self, x1, x2):
        # Евклидово расстояние
        sum_sq = 0
        for i in range(len(x1)):
            diff = x1[i] - x2[i]
            sum_sq += diff * diff
        return sum_sq ** 0.5

    def kernel_epanechnikov(self, r):
        # Ядро Епанечникова с индикатором
        return 0.75 * (1 - r**2) * (np.abs(r) <= 1)

    def predict(self, x, k):
        # Расстояния от x до всех объектов выборки
        distances = np.array([self.distance(x, xi) for xi in self.X])
        # Определяем ширину окна h как расстояние до k-го ближайшего соседа
        h = np.partition(distances, k)[k]
        if h == 0:
            h = 1e-10  # чтобы избежать деления на 0
        weights = self.kernel_epanechnikov(distances / h)
        # Суммируем веса по классам
        classes = np.unique(self.Y)
        weight_sums = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            weight_sums[i] = np.sum(weights[self.Y == cls])
        # Возвращаем класс с максимальным суммарным весом
        return classes[np.argmax(weight_sums)]

    def loo(self, k):
        errors = 0
        for i in range(self.n_samples):
            # Формируем обучающую выборку без i-го объекта
            X_train = np.delete(self.X, i, axis=0)
            Y_train = np.delete(self.Y, i)
            model = NearestNeighbor(X_train, Y_train)
            pred = model.predict(self.X[i], k)
            if pred != self.Y[i]:
                errors += 1
        accuracy = (self.n_samples - errors) / self.n_samples
        return errors, accuracy

# Загрузка данных
iris = load_iris()
X = iris.data
Y = iris.target

# Вывести данные из iris
# print(iris.data)

# Вывести признаки
# print(iris.feature_names)

# Таргеты
# print(iris.target_names)

# Создаем объект классификатора
nn = NearestNeighbor(X, Y)

# Подбор оптимального k
max_k = 20
loo_errors = []
accuracies = []
for k in range(1, max_k + 1):
    errors, acc = nn.loo(k)
    loo_errors.append(errors)
    accuracies.append(acc)
    print(f"k={k}: LOO errors={errors}, accuracy={acc:.3f}")

# Оптимальное k — минимальное количество ошибок
optimal_k = np.argmin(loo_errors) + 1
print(f"Оптимальное k: {optimal_k}")

# Построение графика LOO(k)
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), loo_errors, marker='o')
plt.title('LOO(k) — количество ошибок на контроле')
plt.xlabel('k')
plt.ylabel('Количество ошибок LOO')
plt.grid(True)
plt.show()
