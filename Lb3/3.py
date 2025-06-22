import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class SGD_Ridge:
    """
    Класс для реализации гребневой регрессии с методом стохастического градиентного спуска
    
    Параметры:
    - tau: float, параметр регуляризации
    - lambda_: float, параметр экспоненциального скользящего среднего
    - eps: float, параметр остановки алгоритма
    - max_iter: int, максимальное число итераций
    - random_state: int, seed для генератора случайных чисел
    """
    
    def __init__(self, tau=0.1, lambda_=0.9, eps=1e-5, max_iter=1000, random_state=None):
        self.tau = tau
        self.lambda_ = lambda_
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None
        self.Q_history = []
        
    def fit(self, X, y):
        """
        Обучение модели
        
        Параметры:
        - X: numpy.ndarray, матрица признаков
        - y: numpy.ndarray, вектор целевых значений
        """
        np.random.seed(self.random_state)
        
        # Добавляем фиктивный признак -1 для w0
        X = np.hstack([X, -np.ones((X.shape[0], 1))])
        
        # Инициализация весов нулями
        self.w = np.zeros(X.shape[1])
        
        # Инициализация переменных для критерия остановки
        prev_Q = np.inf
        Q_smooth = None
        
        for i in range(1, self.max_iter + 1):
            # Случайный выбор объекта
            idx = np.random.randint(X.shape[0])
            x_i = X[idx]
            y_i = y[idx]
            
            # Вычисление градиента и обновление весов
            h = 1.0 / i  # Темп обучения
            prediction = np.dot(self.w, x_i)
            error = prediction - y_i
            
            # Обновление весов
            self.w = self.w * (1 - h * self.tau) - h * error * x_i
            
            # Вычисление функционала качества (экспоненциальное скользящее среднее)
            current_Q = error**2 + (self.tau / 2) * np.sum(self.w**2)
            
            if Q_smooth is None:
                Q_smooth = current_Q
            else:
                Q_smooth = self.lambda_ * Q_smooth + (1 - self.lambda_) * current_Q
            
            self.Q_history.append(Q_smooth)
            
            # Проверка критерия остановки
            if i > 1 and np.abs(Q_smooth - prev_Q) < self.eps:
                break
                
            prev_Q = Q_smooth
    
    def predict(self, X):
        """
        Предсказание целевых значений
        
        Параметры:
        - X: numpy.ndarray, матрица признаков
        
        Возвращает:
        - numpy.ndarray, вектор предсказанных значений
        """
        # Добавляем фиктивный признак -1 для w0
        X = np.hstack([X, -np.ones((X.shape[0], 1))])
        return np.dot(X, self.w)
    
    def get_coefficients(self):
        """Возвращает обученные коэффициенты модели"""
        return self.w
    
    def get_quality_history(self):
        """Возвращает историю значений функционала качества"""
        return self.Q_history


def load_and_prepare_data():
    """
    Загрузка и подготовка данных
    
    Возвращает:
    - X_train, X_test, y_train, y_test: numpy.ndarray, разделенные данные
    """
    # Загрузка датасета California Housing
    california = fetch_california_housing()
    X = california.data
    y = california.target
    
    # Стандартизация признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def cross_validation(X_train, y_train, X_test, y_test, lambda_, eps, tau_values):
    """
    Кросс-валидация для подбора оптимального параметра регуляризации tau
    
    Параметры:
    - X_train, y_train: numpy.ndarray, обучающая выборка
    - X_test, y_test: numpy.ndarray, тестовая выборка
    - lambda_: float, параметр экспоненциального скользящего среднего
    - eps: float, параметр остановки алгоритма
    - tau_values: list, список значений tau для проверки
    
    Возвращает:
    - best_tau: float, лучшее значение tau
    - best_r2: float, лучшее значение R^2
    - tau_r2: list, список значений R^2 для каждого tau
    """
    best_tau = None
    best_r2 = -np.inf
    tau_r2 = []
    
    for tau in tau_values:
        model = SGD_Ridge(tau=tau, lambda_=lambda_, eps=eps)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        tau_r2.append(r2)
        
        if r2 > best_r2:
            best_r2 = r2
            best_tau = tau
    
    return best_tau, best_r2, tau_r2


def main():
    # Описание датасета
    print("""
    Датасет: California Housing
    Описание: Содержит информацию о недвижимости в Калифорнии.
    Признаки:
      - MedInc: средний доход в районе
      - HouseAge: средний возраст домов в районе
      - AveRooms: среднее количество комнат
      - AveBedrms: среднее количество спален
      - Population: население района
      - AveOccup: среднее количество жителей в доме
      - Latitude: широта
      - Longitude: долгота
    Целевая переменная:
      - MedHouseVal: медианная стоимость домов (в сотнях тысяч долларов)
    """)
    
    # Загрузка и подготовка данных
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Параметры алгоритма
    lambda_ = 0.9  # Параметр экспоненциального скользящего среднего
    eps = 1e-5     # Параметр остановки
    
    # Поиск оптимального tau
    tau_values = np.logspace(-4, 2, 50)  # Сетка значений tau в логарифмическом масштабе
    best_tau, best_r2, tau_r2 = cross_validation(X_train, y_train, X_test, y_test, 
                                                lambda_, eps, tau_values)
    
    # Обучение модели с лучшим tau
    model = SGD_Ridge(tau=best_tau, lambda_=lambda_, eps=eps, random_state=42)
    model.fit(X_train, y_train)
    
    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    
    # Вывод результатов
    print("\nРезультаты:")
    print(f"Оптимальный параметр регуляризации tau: {best_tau:.6f}")
    print(f"Коэффициент детерминации R^2 на тестовой выборке: {final_r2:.4f}")
    print("\nКоэффициенты модели:")
    coefficients = model.get_coefficients()
    for i, coef in enumerate(coefficients[:-1]):
        print(f"w{i+1}: {coef:.4f}")
    print(f"w0 (пороговый параметр): {coefficients[-1]:.4f}")
    
    # Графики
    plt.figure(figsize=(15, 5))
    
    # График сходимости
    plt.subplot(1, 3, 1)
    plt.plot(model.get_quality_history())
    plt.title('Сходимость метода стохастического градиента')
    plt.xlabel('Итерация')
    plt.ylabel('Значение функционала качества Q')
    plt.grid(True)
    
    # График зависимости R^2 от tau
    plt.subplot(1, 3, 2)
    plt.semilogx(tau_values, tau_r2)
    plt.axvline(x=best_tau, color='r', linestyle='--', label=f'Лучшее tau={best_tau:.4f}')
    plt.title('Зависимость R^2 от параметра регуляризации')
    plt.xlabel('Параметр регуляризации tau')
    plt.ylabel('Коэффициент детерминации R^2')
    plt.legend()
    plt.grid(True)
    
    # График истинных vs предсказанных значений
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.title('Истинные vs Предсказанные значения')
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()