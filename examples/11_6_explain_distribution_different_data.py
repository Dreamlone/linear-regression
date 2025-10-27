import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Генерация данных
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y_true = np.sin(X).ravel()
noise = np.random.normal(0, 0.1, size=y_true.shape)
y = y_true + noise

# 2. Первая модель: линейная регрессия по X
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

# 3. Вторая модель: линейная регрессия по X и sin(X)
X_extended = np.hstack([X, np.sin(X)])
model_sin = LinearRegression()
model_sin.fit(X_extended, y)
y_pred_sin = model_sin.predict(X_extended)

# 4. Визуализация
plt.figure(figsize=(10, 5))

plt.scatter(X, y, color='gray', alpha=0.5, label='Шумные данные')
plt.plot(X, y_true, label='Истинная функция sin(X)', color='green', linewidth=2)

plt.plot(X, y_pred_linear, label='Линейная регрессия (X)', color='red', linestyle='--')
plt.plot(X, y_pred_sin, label='Лин. регрессия (X + sin(X))', color='blue', linestyle='-.')

plt.legend()
plt.title("Сравнение моделей: обычная линейная vs с sin(X)")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.savefig("/home/dreamlone/PycharmProjects/linear-regression/random_plot.png")