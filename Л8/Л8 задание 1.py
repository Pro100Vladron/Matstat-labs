import pandas as pd
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

# Настройка отображения данных
pd.set_option('display.max_columns', None)

# Загрузка данных
file_path = r'C:\Users\TOP\Downloads\Data_8_1.xlsx'
try:
    data = pd.read_excel(file_path, engine='openpyxl')
except FileNotFoundError:
    raise SystemExit("Ошибка: Файл не найден. Проверьте путь и название файла.")

# Обработка данных
data = data.replace(['-', ''], np.nan)
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Поиск столбцов
oma_column = next((col for col in data.columns if 'ОМА' in col), None)
ma_column = next((col for col in data.columns if 'МА' in col), None)
if not oma_column or not ma_column:
    raise KeyError(f"Не найдены столбцы ОМА/МА. Доступные: {data.columns.tolist()}")

# Очистка данных
clean_data = data[[oma_column, ma_column]].dropna()
oma = clean_data[oma_column].values
ma = clean_data[ma_column].values

# Основной анализ
m_values = np.arange(0, 1.01, 0.01)
results = []

def calculate_p_value(table):
    try:
        b, c = table[0][1], table[1][0]
        n = b + c
        if n == 0: return np.nan
        p = 2 * binom.cdf(min(b, c), n, 0.5)
        return min(p, 1.0)
    except:
        return np.nan

for M in m_values:
    oma_success = oma >= M
    ma_success = ma >= M
    table = np.array([
        [np.sum(oma_success & ma_success), np.sum(oma_success & ~ma_success)],
        [np.sum(~oma_success & ma_success), np.sum(~oma_success & ~ma_success)]
    ])
    results.append({
        'M': M,
        'p_value': calculate_p_value(table)
    })

results_df = pd.DataFrame(results)

# Определение диапазонов
def find(p_values, m_values, alpha=0.05):
    ranges = []
    current_start = None
    for i in range(len(p_values)):
        significant = not np.isnan(p_values[i]) and p_values[i] < alpha
        if significant:
            if current_start is None:
                current_start = m_values[i]
            current_end = m_values[i]
        else:
            if current_start is not None:
                ranges.append((round(current_start, 2), round(current_end, 2)))
                current_start = None
    if current_start is not None:
        ranges.append((round(current_start, 2), round(current_end, 2)))
    return ranges

petr_ranges = find(results_df['p_value'].values, results_df['M'].values)
ivan_ranges = find((results_df['p_value'] >= 0.05).astype(float).values, results_df['M'].values, alpha=1)

# Вывод результатов в консоль
print("\nРЕЗУЛЬТАТЫ АНАЛИЗА:")
print(f"Диапазоны M, подтверждающие Ивана (p ≥ 0.05):")
for r in ivan_ranges:
    print(f"{r[0]:.2f} - {r[1]:.2f}")

print("\nДиапазоны M, подтверждающие Петра (p < 0.05):")
for r in petr_ranges:
    print(f"{r[0]:.2f} - {r[1]:.2f}")

# График p-value
plt.figure(figsize=(12, 6))
plt.plot(results_df['M'], results_df['p_value'], label='p-value', color='blue')
plt.axhline(y=0.05, color='red', linestyle='--', label='α=0.05')

# Закрашиваем области
plt.fill_between(results_df['M'], results_df['p_value'],
                 where=(results_df['p_value'] < 0.05),
                 color='red', alpha=0.3, label='Петр (p < 0.05)')
plt.fill_between(results_df['M'], results_df['p_value'],
                 where=(results_df['p_value'] >= 0.05),
                 color='green', alpha=0.3, label='Иван (p ≥ 0.05)')

plt.title('Зависимость p-value от порога M')
plt.xlabel('Порог M')
plt.ylabel('p-value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()