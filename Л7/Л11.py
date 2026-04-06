import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t, f, chi2
from sklearn.linear_model import LinearRegression

# --- Часть 1: Статистическая обработка двумерной выборки ---

# 1. Загрузка данных из Excel (замените путь и имя листа)
file_path = 'C:\Users\TOP\Downloads\Data_11_1.xlsx' l
sheet_name = 'Sheet13'  # Укажите имя листа с данными
data = pd.read_excel(file_path, sheet_name=sheet_name)
x = data['X'].values
y = data['Y'].values
n = len(x)

# Диаграмма рассеивания
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(x, y, color='blue', label='Данные')
plt.title('Диаграмма рассеивания')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# 2. Коэффициент корреляции Пирсона и проверка значимости
r, p_value = pearsonr(x, y)
alpha = 0.05
plt.subplot(2, 2, 2)
plt.text(0.1, 0.5, f'Коэффициент корреляции: r = {r:.4f}\n'
                    f'p-значение: {p_value:.4f}\n'
                    f'Гипотеза о значимости корреляции: {"Подтверждается" if p_value < alpha else "Не подтверждается"}',
                    fontsize=12, va='center')
plt.axis('off')

# 3. Уравнения линейной регрессии (Y на x и X на y)
x_mean = np.mean(x)
y_mean = np.mean(y)
sigma_x = np.std(x, ddof=1)
sigma_y = np.std(y, ddof=1)

# Y на x
b = r * (sigma_y / sigma_x)
a = y_mean - b * x_mean
print(f"Регрессия Y на x: y = {a:.4f} + {b:.4f} x")

# X на y
d = r * (sigma_x / sigma_y)
c = x_mean - d * y_mean
print(f"Регрессия X на y: x = {c:.4f} + {d:.4f} y")

# График регрессионных прямых
x_vals = np.linspace(min(x), max(x), 100)
y_vals_Y_on_x = a + b * x_vals
y_vals_X_on_y = (x_vals - c) / d

plt.subplot(2, 2, 3)
plt.scatter(x, y, label='Данные')
plt.plot(x_vals, y_vals_Y_on_x, color='red', label='Регрессия Y на x')
plt.plot(x_vals, y_vals_X_on_y, color='green', label='Регрессия X на y')
plt.scatter(x_mean, y_mean, color='black', label='Центроид')
plt.title('Регрессионные прямые')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 4. Контроль с помощью scikit-learn
model_y_on_x = LinearRegression()
model_y_on_x.fit(x.reshape(-1, 1), y)
print(f"Регрессия Y на x (scikit-learn): y = {model_y_on_x.intercept_:.4f} + {model_y_on_x.coef_[0]:.4f} x")

model_x_on_y = LinearRegression()
model_x_on_y.fit(y.reshape(-1, 1), x)
print(f"Регрессия X на y (scikit-learn): x = {model_x_on_y.intercept_:.4f} + {model_x_on_y.coef_[0]:.4f} y")

# --- Часть 2: Оценка качества аппроксимации ---

# 1. Оценка дисперсии ошибок
y_pred = model_y_on_x.predict(x.reshape(-1, 1))
s_squared = np.sum((y - y_pred)**2) / (len(y) - 2)
print(f"Оценка дисперсии ошибок: s² = {s_squared:.4f}")

# 2. Коэффициент детерминации R²
ss_total = np.sum((y - y_mean)**2)
ss_res = np.sum((y - y_pred)**2)
R_squared = 1 - (ss_res / ss_total)
print(f"Коэффициент детерминации: R² = {R_squared:.4f}")

# 3. Доверительные интервалы для параметров регрессии
SE_b = np.sqrt(s_squared / np.sum((x - x_mean)**2))
SE_a = np.sqrt(s_squared * (1/len(x) + x_mean**2 / np.sum((x - x_mean)**2)))
t_crit = t.ppf(1 - 0.05/2, df=len(x) - 2)

CI_a = (a - t_crit * SE_a, a + t_crit * SE_a)
CI_b = (b - t_crit * SE_b, b + t_crit * SE_b)
print(f"Доверительный интервал для a: {CI_a}")
print(f"Доверительный интервал для b: {CI_b}")

# 4. Доверительный интервал для дисперсии ошибок σ²
chi2_low = chi2.ppf(0.025, df=len(x) - 2)
chi2_high = chi2.ppf(0.975, df=len(x) - 2)
CI_sigma2 = ((len(x)-2)*s_squared / chi2_high, (len(x)-2)*s_squared / chi2_low)
print(f"Доверительный интервал для σ²: {CI_sigma2}")

# 5. Доверительные интервалы для среднего значения Y при x₀
x0_values = np.linspace(min(x), max(x), 100)
CI_lower = []
CI_upper = []

for x0 in x0_values:
    term = 1/len(x) + (x0 - x_mean)**2 / np.sum((x - x_mean)**2)
    margin = t_crit * np.sqrt(s_squared * term)
    y0_pred = a + b * x0
    CI_lower.append(y0_pred - margin)
    CI_upper.append(y0_pred + margin)

plt.subplot(2, 2, 4)
plt.scatter(x, y, label='Данные')
plt.plot(x_vals, y_vals_Y_on_x, color='red', label='Регрессия Y на x')
plt.fill_between(x0_values, CI_lower, CI_upper, color='blue', alpha=0.2, label='Доверительный интервал')
plt.title('Доверительные интервалы для среднего Y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Проверка значимости регрессии (F-тест)
F_stat = (R_squared / 1) / ((1 - R_squared) / (len(x) - 2))
F_crit = f.ppf(1 - 0.05, dfn=1, dfd=len(x) - 2)
print(f"Регрессия статистически {'значима' if F_stat > F_crit else 'не значима'}")