import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import pandas as pd

# Чтение данных
df = pd.read_excel('C:/Users/TOP/Downloads/Data_9_2.xlsx', header=0)

# Извлекаем оценки
scores_2022 = df.iloc[:, 0].astype(float).values
scores_2023 = df.iloc[:, 1].astype(float).values

# Удаление пропусков
scores_2022 = scores_2022[~np.isnan(scores_2022)]
scores_2023 = scores_2023[~np.isnan(scores_2023)]

# Описательные статистики
def calculate_stats(sample):
    return {
        'Среднее': np.mean(sample),
        'Медиана': np.median(sample),
        'Стд. отклонение': np.std(sample),
        'Минимум': np.min(sample),
        'Максимум': np.max(sample)
    }

print("2022 год:")
print(calculate_stats(scores_2022))
print("\n2023 год:")
print(calculate_stats(scores_2023))

# Гистограммы
plt.figure(figsize=(12, 6))
plt.hist(scores_2022, bins=10, alpha=0.5, label='2022', density=True)
plt.hist(scores_2023, bins=10, alpha=0.5, label='2023', density=True)
plt.xlabel('Оценки')
plt.ylabel('Плотность')
plt.title('Сравнение распределений')
plt.legend()
plt.show()

# Боксплоты
plt.figure(figsize=(10, 6))
plt.boxplot([scores_2022, scores_2023], labels=['2022', '2023'])
plt.ylabel('Оценки')
plt.title('Сравнение успеваемости')
plt.show()

# Критерий Манна-Уитни
stat, p_mw = sts.mannwhitneyu(scores_2022, scores_2023, alternative='two-sided')

print(f"\nP-value Манна-Уитни: {p_mw:.4f}")
alpha = 0.05
if p_mw < alpha:
    print("Есть значимые различия между 2022 и 2023 годами")
else:
    print("Нет значимых различий между годами")