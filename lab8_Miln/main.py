from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math

try:
    from scipy.integrate import solve_ivp
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


@dataclass
class IVP:
    f: Callable[[float, float], float]
    a: float
    b: float
    y0: float
    h: float = 0.1

def make_grid(a: float, b: float, h: float) -> Tuple[np.ndarray, float]:
    n = int(round((b - a) / h))
    x = a + np.arange(n + 1, dtype=float) * h
    return x, h

def runge_error(y_h: np.ndarray, y_h2: np.ndarray, p: int = 4) -> np.ndarray:
    """Оценка Рунге: |y_h − y_{h/2}[::2]| / (2^p − 1). Для p=4 делим на 15."""
    y_h2_on_h = y_h2[::2]
    if len(y_h2_on_h) != len(y_h):
        y_h2_on_h = y_h2_on_h[:len(y_h)]
    return np.abs(y_h - y_h2_on_h) / (2**p - 1)

def rk4_bootstrap(ivp: IVP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, h = make_grid(ivp.a, ivp.b, ivp.h)
    y = np.empty_like(x); y[0] = ivp.y0
    # посчитаем y1..y3
    upto = min(3, len(x)-1)
    for i in range(upto):
        xi, yi = x[i], y[i]
        k1 = h * ivp.f(xi, yi)
        k2 = h * ivp.f(xi + 0.5*h, yi + 0.5*k1)
        k3 = h * ivp.f(xi + 0.5*h, yi + 0.5*k2)
        k4 = h * ivp.f(xi + h,     yi + k3)
        y[i+1] = yi + (1/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    fvals = np.array([ivp.f(x[i], y[i]) for i in range(upto+1)] + [np.nan]*(len(x)-(upto+1)))
    return x, y, fvals

def milne_solve(ivp: IVP) -> Tuple[np.ndarray, np.ndarray]:
    x, h = make_grid(ivp.a, ivp.b, ivp.h)
    y, fvals = np.empty_like(x), np.empty_like(x)
    # старт RK4
    _, y_boot, _ = rk4_bootstrap(ivp)
    y[:4] = y_boot[:4]
    for i in range(4):
        fvals[i] = ivp.f(x[i], y[i])

    # шаги Милна
    for n in range(3, len(x)-1):
        # предиктор
        ypred = y[n-3] + (4*h/3.0) * (2*fvals[n-2] - fvals[n-1] + 2*fvals[n])
        # один корректирующий шаг Симпсона
        ycorr = y[n-1] + (h/3.0) * (fvals[n-1] + 4*fvals[n] + ivp.f(x[n+1], ypred))
        y[n+1] = ycorr
        fvals[n+1] = ivp.f(x[n+1], y[n+1])

    return x, y


def milne_double_count(ivp: IVP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_h, y_h = milne_solve(ivp)
    ivp_half = IVP(f=ivp.f, a=ivp.a, b=ivp.b, y0=ivp.y0, h=ivp.h/2)
    x_h2, y_h2 = milne_solve(ivp_half)
    R = runge_error(y_h, y_h2, p=4)
    return x_h, y_h, R

def plot_solution(x, y, title):
    plt.figure()
    plt.plot(x, y, linewidth=3, label="Milne (PC)")
    plt.xlabel("x"); plt.ylabel("y(x)"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

def plot_error(x, R, title):
    plt.figure()
    plt.plot(x, R, marker='o', linewidth=1, label="Погрешность по методу двойного счета")
    imax = int(np.argmax(R))
    plt.scatter([x[imax]], [R[imax]], s=30, zorder=3,
                label=f"max≈{float(R[imax]):.3e} при x≈{float(x[imax]):.3g}")
    plt.yscale("log"); plt.xlabel("x"); plt.ylabel("ошибка")
    plt.title(title); plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()

if __name__ == "__main__":
    f = lambda x, y: math.exp(-x) * math.cos(x - y**2)
    ivp = IVP(f=f, a=0.0, b=0.8, y0=0.5, h=0.1)

    x, y, R = milne_double_count(ivp)

    if HAS_SCIPY:
        t_eval = x
        rhs = lambda t, Y: ivp.f(t, Y[0])
        ref = solve_ivp(rhs, (ivp.a, ivp.b), [ivp.y0], t_eval=t_eval, rtol=1e-12, atol=1e-14)
        y_ref = ref.y[0]
        plt.figure()
        plt.plot(x, y, label="Милн", linewidth=3)
        plt.plot(x, y_ref, ":", label="solve_ivp (ref)", linewidth=2)
        plt.xlabel("x"); plt.ylabel("y(x)")
        plt.title("Метод Милна и solve_ivp"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

        # ошибка к референсу
        plt.figure()
        plt.plot(x, np.abs(y - y_ref), label="|Milne - ref|")
        plt.yscale("log"); plt.xlabel("x"); plt.ylabel("ошибка к ref")
        plt.title("Ошибка метода Милна (к solve_ivp)"); plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()

    # графики задания
    plot_solution(x, y, "Метод Милна, h=0.1")
    plot_error(x, R, "Оценка погрешности")

    plt.show()
