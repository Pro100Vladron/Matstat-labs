import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, expon, chisquare

# --- 1. Загрузка и предобработка данных ---
file_path = r'C:\Users\TOP\Downloads\Data_7_1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Обработка всех столбцов
for column in data.columns:
    data[column] = data[column].astype(str).str.replace('-', 'nan')
    data[column] = pd.to_numeric(data[column], errors='coerce')

print("Столбцы в файле:", data.columns.tolist())
print("\nПервые 5 строк данных (без NaN):")
print(data.head().dropna())

# --- 2. Построение гистограмм и визуальный анализ ---
for column in data.columns:
    sample = data[column].dropna()
    if sample.empty:
        continue

    plt.figure(figsize=(10, 6))
    n, bins, _ = plt.hist(sample, bins='auto', density=True, alpha=0.7, label='Эмпирическое')

    # Добавляем теоретические кривые
    x = np.linspace(sample.min(), sample.max(), 100)
    plt.plot(x, norm.pdf(x, sample.mean(), sample.std()), 'r--', label='Нормальное')
    plt.plot(x, expon.pdf(x, scale=sample.mean()), 'g--', label='Экспоненциальное')

    plt.title(f'Гистограмма {column}')
    plt.xlabel('Значения')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

# --- 3. Выбор и проверка выборок ---
sample_A_name = 'Sample3'  # Выборка с симметричной гистограммой
sample_B_name = 'Sample5'  # Выборка для преобразования

sample_A = data[sample_A_name].dropna()
sample_B = data[sample_B_name].dropna()

# Преобразование выборки B для экспоненциального распределения
sample_B_transformed = np.abs(sample_B - sample_B.mean())

# --- 4. Проверка нормальности выборки А ---
print("\n--- Проверка нормальности для выборки А ---")
mu, sigma = norm.fit(sample_A)
n = len(sample_A)

# Создание интервалов
bins = np.histogram_bin_edges(sample_A, bins='auto')
bins = np.insert(bins, 0, -np.inf)
bins = np.append(bins, np.inf)

observed, bins = np.histogram(sample_A, bins=bins)
expected = n * np.diff(norm.cdf(bins, mu, sigma))

# Корректировка интервалов
while len(expected) > 1 and (expected < 5).any():
    bins = np.delete(bins, -2)
    observed, _ = np.histogram(sample_A, bins=bins)
    expected = n * np.diff(norm.cdf(bins, mu, sigma))

# График проверки нормальности
plt.figure(figsize=(10, 6))
plt.hist(sample_A, bins=bins, density=True, alpha=0.5, label='Эмпирическое')
x = np.linspace(sample_A.min(), sample_A.max(), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='Теоретическое')
plt.title(f'Проверка нормальности для {sample_A_name}')
plt.legend()
plt.show()
plt.close()

# Вывод результатов
print("\nПромежутки и частоты:")
for i in range(len(bins) - 1):
    print(f"[{bins[i]:>8.2f}; {bins[i + 1]:<8.2f}): "
          f"Эмпир.={observed[i]:<5d} "
          f"Теор.={expected[i]:.2f}")

chi_stat, p_value = chisquare(observed, expected, ddof=2)
print(f"\nХи-квадрат = {chi_stat:.2f}, p-value = {p_value:.4f}")
print(f"Вывод: Гипотеза {'принимается' if p_value > 0.05 else 'отвергается'}")

# --- 5. Проверка показательного распределения для выборки В ---
print("\n--- Проверка показательного распределения для выборки В ---")
lambda_ = 1 / sample_B_transformed.mean()
n_exp = len(sample_B_transformed)

# Создание интервалов
bins_exp = np.histogram_bin_edges(sample_B_transformed, bins='auto')
bins_exp = np.insert(bins_exp, 0, 0.0)
bins_exp = np.append(bins_exp, np.inf)

observed_exp, bins_exp = np.histogram(sample_B_transformed, bins=bins_exp)
expected_exp = n_exp * np.diff(expon.cdf(bins_exp, scale=1 / lambda_))

# Корректировка интервалов
while len(expected_exp) > 1 and (expected_exp < 5).any():
    bins_exp = np.delete(bins_exp, -2)
    observed_exp, _ = np.histogram(sample_B_transformed, bins=bins_exp)
    expected_exp = n_exp * np.diff(expon.cdf(bins_exp, scale=1 / lambda_))

# График проверки экспоненциального распределения
plt.figure(figsize=(10, 6))
plt.hist(sample_B_transformed, bins=bins_exp, density=True, alpha=0.5, label='Эмпирическое')
x = np.linspace(0, sample_B_transformed.max(), 100)
plt.plot(x, expon.pdf(x, scale=1 / lambda_), 'r-', lw=2, label='Теоретическое')
plt.title(f'Проверка экспоненциального распределения для {sample_B_name}')
plt.legend()
plt.show()
plt.close()

# Вывод результатов
print("\nПромежутки и частоты:")
for i in range(len(bins_exp) - 1):
    print(f"[{bins_exp[i]:>8.2f}; {bins_exp[i + 1]:<8.2f}): "
          f"Эмпир.={observed_exp[i]:<5d} "
          f"Теор.={expected_exp[i]:.2f}")

chi_stat_exp, p_value_exp = chisquare(observed_exp, expected_exp, ddof=1)
print(f"\nХи-квадрат = {chi_stat_exp:.2f}, p-value = {p_value_exp:.4f}")
print(f"Вывод: Гипотеза {'принимается' if p_value_exp > 0.05 else 'отвергается'}")
file_path = r'C:\Users\TOP\Downloads\Data_7_1.xlsx'