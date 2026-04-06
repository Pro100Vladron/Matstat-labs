import numpy as np
import matplotlib.pyplot as plt
import math


# =============================================
# Часть 1: Линейная интерполяция для f(t) = e^t
# =============================================
def part1():
    f = lambda t: np.exp(t)
    ti = (np.arange(5) + 3) / 8  # Узлы интерполяции [0.375, 0.5, 0.625, 0.75, 0.875]
    f_ti = f(ti)
    t_plot = np.linspace(0, 1, 1000)

    # Линейная интерполяция
    linear_interp = np.interp(t_plot, ti, f_ti)

    # Построение графиков
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, f(t_plot), label='Реальная функция: $e^t$')
    plt.plot(t_plot, linear_interp, '--', label='Линейная интерполяция')
    plt.scatter(ti, f_ti, color='red', zorder=5, label='Узлы интерполяции')
    plt.title('Сравнение линейной интерполяции и реальной функции')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================
# Часть 2: Интерполяция полиномом Лагранжа
# =============================================
def lagrange_poly(t, xi, yi):
    """Классическая реализация полинома Лагранжа"""
    n = len(xi)
    total = 0.0
    for i in range(n):
        term = yi[i]
        for j in range(n):
            if i != j:
                term *= (t - xi[j]) / (xi[i] - xi[j])
        total += term
    return total


def lagrange_poly_matrix(t, xi, yi):
    """Векторизованная реализация с матричными операциями"""
    t = np.asarray(t)
    n = len(xi)
    L = np.zeros_like(t)
    for i in range(n):
        mask = np.arange(n) != i
        xi_not_i = xi[mask]
        numerator = t - xi_not_i[:, None]
        denominator = xi[i] - xi_not_i[:, None]
        Li = np.prod(numerator / denominator, axis=0)
        L += yi[i] * Li
    return L


def part2():
    f = lambda t: np.exp(t)
    ti = (np.arange(5) + 3) / 8
    f_ti = f(ti)
    t_plot = np.linspace(0, 1, 1000)

    # Вычисление значений полинома
    lagrange_vals = lagrange_poly_matrix(t_plot, ti, f_ti)

    # Расчет ошибок
    error = np.abs(f(t_plot) - lagrange_vals)
    max_error = np.max(error)
    print(f"Максимальная ошибка интерполяции: {max_error:.6f}")

    # Теоретическая оценка ошибки
    n = len(ti) - 1
    M = np.exp(1)  # Максимум n+1 производной на [0,1]
    factorial = np.math.factorial(n + 1)
    h = (ti[-1] - ti[0]) / n
    theoretical_error = (M * h ** (n + 1)) / factorial
    print(f"Теоретическая оценка ошибки: {theoretical_error:.6f}")

    # Построение графиков
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, f(t_plot), label='Реальная функция: $e^t$')
    plt.plot(t_plot, lagrange_vals, '--', label='Полином Лагранжа')
    plt.scatter(ti, f_ti, color='red', zorder=5, label='Узлы интерполяции')
    plt.title('Сравнение полинома Лагранжа и реальной функции')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================
# Часть 4: Экстраполяция в t=2
# =============================================
def part4():
    f = lambda t: np.exp(t)
    ti = (np.arange(5) + 3) / 8
    f_ti = f(ti)

    t_test = 2.0
    lagrange_val = lagrange_poly(t_test, ti, f_ti)
    real_val = f(t_test)

    print(f"\nЭкстраполяция в t=2:")
    print(f"Реальное значение: {real_val:.6f}")
    print(f"Значение полинома: {lagrange_val:.6f}")
    print(f"Разница: {abs(real_val - lagrange_val):.6f}")


# =============================================
# Часть 5-7: Функция Рунге и узлы Чебышева
# =============================================
def runge(x):
    return 1 / (1 + 25 * x ** 2)


def chebyshev_nodes(a, b, n):
    """Генерация узлов Чебышева на интервале [a, b]"""
    k = np.arange(1, n + 2)
    nodes = np.cos((2 * k - 1) * np.pi / (2 * (n + 1)))
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes


def part567():
    # Параметры
    a, b = -5, 5
    n = 10  # Количество узлов

    # Равномерные узлы
    x_uniform = np.linspace(a, b, n + 1)
    y_uniform = runge(x_uniform)

    # Узлы Чебышева
    x_cheb = chebyshev_nodes(a, b, n)
    y_cheb = runge(x_cheb)

    # Точки для построения графиков
    x_plot = np.linspace(a, b, 1000)

    # Вычисление интерполяционных полиномов
    uniform_poly = lagrange_poly_matrix(x_plot, x_uniform, y_uniform)
    cheb_poly = lagrange_poly_matrix(x_plot, x_cheb, y_cheb)

    # Сравнение в точке x=4.5
    test_point = 4.5
    print(f"\nСравнение в точке x={test_point}:")
    print(f"Реальное значение: {runge(test_point):.6f}")
    print(f"Равномерные узлы: {lagrange_poly(test_point, x_uniform, y_uniform):.6f}")
    print(f"Узлы Чебышева: {lagrange_poly(test_point, x_cheb, y_cheb):.6f}")

    # Построение графиков
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_plot, runge(x_plot), label='Функция Рунге')
    plt.plot(x_plot, uniform_poly, '--', label='Равномерные узлы')
    plt.scatter(x_uniform, y_uniform, color='red')
    plt.title('Интерполяция с равномерными узлами')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_plot, runge(x_plot), label='Функция Рунге')
    plt.plot(x_plot, cheb_poly, '--', label='Узлы Чебышева')
    plt.scatter(x_cheb, y_cheb, color='red')
    plt.title('Интерполяция с узлами Чебышева')
    plt.xlabel('x')
    plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================
# Главная функция
# =============================================
if __name__ == "__main__":
    part1()
    part2()
    part4()
    part567()