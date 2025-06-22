import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class KMeans:
    def __init__(self, k, max_iter=300, n_init=10):
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init
        self.centroids = None # координаты центроидов
        self.labels = None # метка кластера для каждого объекта
        self.inertia_ = float('inf') 
    
    def Distance(self, a, b):
        return np.sqrt(np.sum((a - b)**2, axis=1))
    
    # Случайная инициализация центроидов в пределах диапазона данных
    def InitializationCentre(self, X):
        # Выбираем k уникальных случайных индексов из X
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]
    
    # Распределение объектов по ближайшим центроидам
    def Expectation(self, X, centroids):
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(centroids):
            distances[:, i] = self.Distance(X, centroid)
        return np.argmin(distances, axis=1)
    
    # Пересчет центроидов как средних точек кластеров
    def UpdateCentroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
    
    # Качество кластеризации
    def Quality(self, X, labels, centroids):
        return sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(self.k))
    
    def fit(self, X):
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        
        for _ in range(self.n_init):
            centroids = self.InitializationCentre(X)
            labels_prev = np.zeros(X.shape[0])
            
            for _ in range(self.max_iter):
                labels = self.Expectation(X, centroids)
                centroids = self.UpdateCentroids(X, labels)
                
                if np.array_equal(labels, labels_prev):
                    break
                labels_prev = labels
            
            inertia = self.Quality(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
        
        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia_ = best_inertia
    
    def predict(self, X):
        return self.Expectation(X, self.centroids)

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Параметры алгоритма
k = 3
n_init = 10

# Кластеризация
kmeans = KMeans(k=k, n_init=n_init)
kmeans.fit(X)
labels = kmeans.labels
centroids = kmeans.centroids

# Визуализация (используем первые два признака)
plt.figure(figsize=(8, 5))

# Предсказанные кластеры
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Центроиды')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('K-means кластеризация')
plt.legend()

plt.tight_layout()
plt.show()

# Качество кластеризации
print(f"Сумма квадратов внутрикластерных расстояний: {kmeans.inertia_:.4f}")
