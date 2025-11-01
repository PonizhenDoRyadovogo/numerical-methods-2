from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math


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

def modified_euler(ivp: IVP, n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, h = make_grid(ivp.a, ivp.b, n)
    y = np.empty_like(x)
    y[0] = ivp.y0
    for i in range(n):
        k1 = ivp.f(x[i], y[i])
        x_mid = x[i] + 0.5 * h
        y_mid = y[i] + 0.5 * h * k1
        k_mid = ivp.f(x_mid, y_mid)
        y[i + 1] = y[i] + h * k_mid
    return x, y


def euler_koshi(ivp: IVP, n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, h = make_grid(ivp.a, ivp.b, n)
    y = np.empty_like(x)
    y[0] = ivp.y0
    for i in range(n):
        fi = ivp.f(x[i], y[i])
        y_star = y[i] + h * fi
        fi1 = ivp.f(x[i + 1], y_star)
        y[i + 1] = y[i] + 0.5 * h * (fi + fi1)
    return x, y


def modified_euler_koshi(ivp: IVP, n: int, iters: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    x, h = make_grid(ivp.a, ivp.b, n)
    y = np.empty_like(x)
    y[0] = ivp.y0
    for i in range(n):
        fi = ivp.f(x[i], y[i])
        z = y[i]
        for _ in range(iters):
            z = y[i] + 0.5 * h * (fi + ivp.f(x[i + 1], z))
        y[i + 1] = z
    return x, y

def double_count(method_fn: Callable[[IVP, int], Tuple[np.ndarray, np.ndarray]],
                 ivp: IVP, n: int, p: int = 2):
    x_h, y_h = method_fn(ivp, n)
    _,  y_h2 = method_fn(ivp, 2 * n)
    R = runge_error(y_h, y_h2, p=p)
    return x_h, y_h, R


def plot_error(x, R, title, annotate_max=True):
    plt.figure()
    plt.plot(x, R, marker='o', linewidth=1, label="оценка Рунге")
    if annotate_max:
        imax = int(np.argmax(R))
        plt.scatter([x[imax]], [R[imax]], s=30, zorder=3, label=f"max ≈ {R[imax]:.3e} при x≈{x[imax]:.3g}")
    plt.yscale("log")
    plt.xlabel("x"); plt.ylabel("ошибка")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

if __name__ == "__main__":
    f = lambda x, y: math.exp(-x) * math.cos(x - y**2)
    a, b = 0.0, 0.8
    y0 = 0.5
    n = 200

    ivp = IVP(f=f, a=a, b=b, y0=y0)


    x_a, y_a = modified_euler(ivp, n)
    x_b, y_b = euler_koshi(ivp, n)

    delta_ab = np.abs(y_a - y_b)
    delta_ab_max = float(np.max(delta_ab))
    x_at_ab_max = float(x_a[int(np.argmax(delta_ab))])


    x_a_R, y_a_R, R_a = double_count(modified_euler, ivp, n, p=2)
    x_b_R, y_b_R, R_b = double_count(euler_koshi, ivp, n, p=2)


    x_c, y_c = modified_euler_koshi(ivp, n, iters=4)
    x_c_R, y_c_R, R_c = double_count(lambda ivp_, n_: modified_euler_koshi(ivp_, n_, iters=4),
                                     ivp, n, p=2)

    print("== ЧМ2, Задание 4 ==")
    print(f"[а] Модифицированный метод ломанных:       max Runge-error ≈ {float(np.max(R_a)):.6e}")
    print(f"[б] Усовершенстованный метод Эйлера-Коши (явный):                     max Runge-error ≈ {float(np.max(R_b)):.6e}")
    print(f"[с] Усовершенствованный метод Эйлера-Коши (неявный): max Runge-error ≈ {float(np.max(R_c)):.6e}")
    print(f"(6) Δ_max = max_i |y_a - y_b| ≈ {delta_ab_max:.6e} при x ≈ {x_at_ab_max:.6g}")

    # --- графики решений ---
    plt.figure()
    plt.plot(x_a, y_a, label="(а) Модифицированный метод ломаных", linewidth=6)
    plt.plot(x_b, y_b, label="(б) Усовершенстованный метод Эйлера-Коши (явный)", linewidth=3)
    plt.plot(x_c, y_c, label="(в) Усовершенствованный метод Эйлера-Коши (неявный) (4 итерации)", linewidth=2, linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Модифицированные методы Эйлера (а, б, в)")
    plt.legend()
    plt.grid(True, alpha=0.3);
    plt.tight_layout()

    # --- графики погрешностей (двойной счёт p=2) ---
    plot_error(x_a_R, R_a, title="Погрешность (а) Модифицированный метод ломаных")
    plot_error(x_b_R, R_b, title="Погрешность (б) Усовершенстованный метод Эйлера-Коши (явный)")
    plot_error(x_c_R, R_c, title="Погрешность (в) Усовершенствованный метод Эйлера-Коши (неявный) (4 итерации)")

    # --- график Δ_i между (а) и (б) по формуле (6) ---
    plt.figure()
    plt.plot(x_a, delta_ab, marker='o', linewidth=1, label=r"$\Delta_i=|y_a-y_b|$")
    imax = int(np.argmax(delta_ab))
    plt.scatter([x_a[imax]], [delta_ab[imax]], s=30, zorder=3, label=f"Δ_max≈{delta_ab_max:.3e} при x≈{x_a[imax]:.3g}")
    plt.yscale("log")
    plt.xlabel("x"); plt.ylabel("Δ_i")
    plt.title("Сравнение (а) vs (б) по формуле (*)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    plt.show()
