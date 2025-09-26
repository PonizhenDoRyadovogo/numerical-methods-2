import math

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple
from scipy.integrate import solve_ivp

@dataclass
class IVP:
    f: Callable[[float, float], float]  # правая часть f(x,y)
    a: float                            # левый конец интервала
    b: float                            # правый конец интервала
    y0: float                           # начальное значение y(a)

def euler(ivp: IVP, N: int) -> Tuple[np.ndarray, np.ndarray]:
    a, b, y = ivp.a, ivp.b, ivp.y0
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y_arr = np.empty(N + 1, dtype=float)
    y_arr[0] = y
    for i in range(N):
        y = y + h * ivp.f(x[i], y)
        y_arr[i + 1] = y
    return x, y_arr

def double_count_error(ivp: IVP, N: int):
    # Решение на шаге h
    x_h, y_h = euler(ivp, N)
    # Решение на шаге h/2
    x_h2, y_h2 = euler(ivp, 2 * N)
    s = 1  # порядок метода Эйлера
    err = np.abs(y_h2[::2] - y_h) / (2**s - 1)
    return x_h, y_h, err

def solve_and_plot(ivp: IVP, N: int, title: str = "Метод Эйлера"):
    x, y, err = double_count_error(ivp, N)
    max_err = float(np.max(err))
    imax = int(np.argmax(err))
    xmax = x[imax]

    def f_ivp(t, y_vec):
        return [ivp.f(t,y_vec[0])]

    sol = solve_ivp(f_ivp, (ivp.a, ivp.b), [ivp.y0], t_eval=x)


    # Графики
    plt.figure()
    plt.plot(x, y, label="Численное решение (Эйлер)", linewidth=5, color='m')
    plt.plot(sol.t, sol.y[0], label = "Эталонное решение", linestyle="--", color='k')
    plt.xlabel("x"); plt.ylabel("y"); plt.title(title); plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(x, err, label="Оценка погрешности (двойной счёт)")
    plt.scatter([xmax], [max_err], marker="o", label=f"max ≈ {max_err:.3e} при x≈{xmax:.3g}")
    plt.xlabel("x"); plt.ylabel("|ошибка|"); plt.title("Погрешность")
    plt.yscale("log")
    plt.legend(); plt.grid(True, which="both")

    print(f"Максимальная оценка погрешности на интервале: {max_err:.6e} при x ≈ {xmax:.6g}")
    plt.show()

ivp = IVP(
    f=lambda x, y: math.exp(-x) * math.cos(x - y ** 2),
    a=0.0,
    b=0.8,
    y0=0.5
)

N = 200
solve_and_plot(ivp, N, title="Эйлер для y' = e^(-x)cos(x-y^2),  y(0)=0.5,  [0,0.8]")
