import numpy as np
from scipy import stats

# Генерация выборок
np.random.seed(42)
mA = 10
mB = 10.5
sigmaA = 2
sigmaB = 2.1
n_A = 30
n_B = 40

sample_A = np.random.normal(loc=mA, scale=sigmaA, size=n_A)
sample_B = np.random.normal(loc=mB, scale=sigmaB, size=n_B)

# 1) Проверка равенства дисперсий по критерию Фишера
var_A = np.var(sample_A, ddof=1)
var_B = np.var(sample_B, ddof=1)

f_stat = max(var_A, var_B) / min(var_A, var_B)
dfn = (n_A - 1) if var_A >= var_B else (n_B - 1)
dfd = (n_B - 1) if var_A >= var_B else (n_A - 1)

alpha_f = 0.05
critical_value_f = stats.f.ppf(1 - alpha_f / 2, dfn, dfd)

print(f"F-статистика: {f_stat:.4f}")
print(f"Критическое значение: {critical_value_f:.4f}")

if f_stat > critical_value_f:
    print("Отвергаем H0: дисперсии не равны.")
else:
    print("Не отвергаем H0: дисперсии равны.")

# 2) Проверка равенства математических ожиданий
reject_f = f_stat > critical_value_f
if reject_f:
    print("\nИспользуется тест Уэлча (неравные дисперсии)")
    t_stat, p_value = stats.ttest_ind(sample_A, sample_B, equal_var=False)
else:
    print("\nИспользуется t-критерий Стьюдента (равные дисперсии)")
    t_stat, p_value = stats.ttest_ind(sample_A, sample_B, equal_var=True)

print(f"t-статистика: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

print(f"\nГипотеза о равенстве математических ожиданий принимается при α < {p_value:.4f}")