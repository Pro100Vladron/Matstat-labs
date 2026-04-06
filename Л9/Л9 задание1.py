import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_excel(r'C:\Users\TOP\Downloads\Data_9_1.xlsx', header=None)
df.columns = ['Фамилия', 'Имя', 'Время', 'Оценка']

# Удаление строк с пропусками
df = df.dropna()

# Преобразование времени в минуты
def parse_time(time_str):
    if pd.isna(time_str):
        return np.nan
    parts = time_str.split()
    minutes = 0
    if 'ч.' in parts:
        hours = float(parts[0].replace('ч.', '').strip())
        minutes += hours * 60
        parts = parts[1:]
    if 'мин.' in parts:
        mins = float(parts[0].replace('мин.', '').strip())
        minutes += mins
        parts = parts[1:]
    if 'сек.' in parts:
        secs = float(parts[0].replace('сек.', '').strip())
        minutes += secs / 60
    return minutes

df['Время_мин'] = df['Время'].apply(parse_time)

# Группировка по студентам (ФИО)
df['ФИО'] = df['Фамилия'] + ' ' + df['Имя']
grouped = df.groupby('ФИО').agg(list).reset_index()

# Фильтрация студентов с двумя попытками
two_attempts = grouped[grouped['Оценка'].apply(len) == 2]
two_attempts = two_attempts.assign(
    Оценка1 = two_attempts['Оценка'].apply(lambda x: x[0]),
    Оценка2 = two_attempts['Оценка'].apply(lambda x: x[1]),
    Время1 = two_attempts['Время_мин'].apply(lambda x: x[0]),
    Время2 = two_attempts['Время_мин'].apply(lambda x: x[1])
)

# Описательные статистики
scores1 = two_attempts['Оценка1']
scores2 = two_attempts['Оценка2']
print("Оценки первой попытки:")
print(f"Среднее: {np.mean(scores1):.2f}, Медиана: {np.median(scores1):.2f}")
print("Оценки второй попытки:")
print(f"Среднее: {np.mean(scores2):.2f}, Медиана: {np.median(scores2):.2f}")

# Боксплоты
plt.figure(figsize=(10, 6))
sns.boxplot(data=two_attempts[['Оценка1', 'Оценка2']])
plt.title('Сравнение оценок первой и второй попыток')
plt.ylabel('Оценка')
plt.show()

# Критерий знаков (вручную)
differences = two_attempts['Оценка2'] - two_attempts['Оценка1']
positive = (differences > 0).sum()
negative = (differences < 0).sum()
no_change = (differences == 0).sum()

# Общее количество сравнений
n = positive + negative

# Вычисление p-value вручную (биномиальное распределение)
p_value = 0
for k in range(positive, n + 1):
    p_value += np.math.comb(n, k) * (0.5 ** k) * (0.5 ** (n - k))

print(f"Улучшений: {positive}, Ухудшений: {negative}, Без изменений: {no_change}")
print(f"P-value: {p_value:.4f}")

# Вывод результатов
if p_value < 0.01:
    print("Улучшение значимо на уровне p < 0.01")
elif p_value < 0.05:
    print("Улучшение значимо на уровне p < 0.05")
else:
    print("Нет значимого улучшения")