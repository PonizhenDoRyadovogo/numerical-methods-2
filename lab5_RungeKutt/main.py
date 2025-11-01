from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp



@dataclass
class IVP:
    f: Callable[[float, float], float]  # y' = f(x, y)
    a: float
    b: float
    y0: float


def make_grid(a: float, b: float, n: int) -> Tuple[np.ndarray, float]:
    x = np.linspace(a, b, n + 1, dtype=float)
    h = (b - a) / n
    return x, h


def runge_error(y_h: np.ndarray, y_h2: np.ndarray, p: int) -> np.ndarray:
    y_h2_on_h = y_h2[::2]
    if len(y_h2_on_h) != len(y_h):
        y_h2_on_h = y_h2_on_h[: len(y_h)]
    return np.abs(y_h - y_h2_on_h) / (2**p - 1)


def rk4(ivp: IVP, n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, h = make_grid(ivp.a, ivp.b, n)
    y = np.empty_like(x)
    y[0] = ivp.y0
    for i in range(n):
        xi, yi = x[i], y[i]
        k1 = h * ivp.f(xi, yi)
        k2 = h * ivp.f(xi + 0.5*h, yi + 0.5*k1)
        k3 = h * ivp.f(xi + 0.5*h, yi + 0.5*k2)
        k4 = h * ivp.f(xi + h,     yi + k3)
        y[i+1] = yi + (k1 + 2*k2 + 2*k3 + k4)/6.0
    return x, y


def rk5_merson(ivp: IVP, n: int) -> tuple[np.ndarray, np.ndarray]:
    x, h = make_grid(ivp.a, ivp.b, n)
    y = np.empty_like(x)
    y[0] = ivp.y0
    for i in range(n):
        xi, yi = x[i], y[i]
        k0 = h * ivp.f(xi, yi)
        k1 = h * ivp.f(xi + h/3, yi + k0/3)
        k2 = h * ivp.f(xi + h/3, yi + k0/6 + k1/6)
        k3 = h * ivp.f(xi + h/2, yi + k0/8 + 3*k2/8)
        k4 = h * ivp.f(xi + h,   yi + k0/2 - 3*k2/2 + 2*k3)
        y[i+1] = yi + (k0 + 4*k3 + k4) / 6.0
    return x, y

def double_count(method_fn: Callable[[IVP, int], Tuple[np.ndarray, np.ndarray]],
                 ivp: IVP, n: int, p: int):
    x_h, y_h = method_fn(ivp, n)
    _,   y_h2 = method_fn(ivp, 2*n)
    R = runge_error(y_h, y_h2, p=p)
    return x_h, y_h, R

def plot_solution(xs, ys, labels, title, styles=None):
    plt.figure()
    if styles is None: styles = [{}]*len(xs)
    for (x, y, lab), st in zip(zip(xs, ys, labels), styles):
        plt.plot(x, y, label=lab, **st)
    plt.xlabel("x"); plt.ylabel("y(x)")
    plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

def plot_error(x, R, title):
    plt.figure()
    plt.plot(x, R, marker='o', linewidth=1)
    imax = int(np.argmax(R))
    plt.scatter([x[imax]], [R[imax]], s=30, zorder=3,
                label=f"max ≈ {float(R[imax]):.3e} при x≈{float(x[imax]):.3g}")
    plt.yscale("log")
    plt.xlabel("x"); plt.ylabel("ошибка (оценка Рунге)")
    plt.title(title)
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

if __name__ == "__main__":
    f = lambda x, y: math.exp(-x) * math.cos(x - y**2) - y**2
    a, b = 0.0, 0.8
    y0 = 0.5
    n = 12

    ivp = IVP(f=f, a=a, b=b, y0=y0)

    x4, y4, R4 = double_count(rk4, ivp, n, p=4)
    x5, y5, R5 = double_count(rk5_merson, ivp, n, p=5)


    # Референсное решение в тех же узлах (по x4; можно взять и по x5 — одинаковые узлы)
    def rhs(t, y): return ivp.f(t, y[0])
    ref = solve_ivp(rhs, (ivp.a, ivp.b), [ivp.y0], t_eval=x4, rtol=1e-10, atol=1e-12)
    y_ref = ref.y[0]

    print("== ЧМ2, Задание 5 ==")
    print(f"RK4: max Runge-error ≈ {float(np.max(R4)):.6e}")
    print(f"RK5: max Runge-error ≈ {float(np.max(R5)):.6e}")

    # График решений
    plot_solution(
        [x4, x5, x4],
        [y4, y5, y_ref],
        labels=["RK4 (p=4)", "RK5 (p=5)", "solve_ivp (reference)"],
        title="Рунге–Кутта 4 и 5 порядка: решение",
        styles=[{"linewidth":4}, {"linewidth":4, "linestyle":"--"}, {"linewidth":2, "linestyle":":"}]
    )


    # Графики ошибок (двойной счёт)
    plot_error(x4, R4, "RK4: погрешность")
    plot_error(x5, R5, "RK5: погрешность")

    plt.show()
