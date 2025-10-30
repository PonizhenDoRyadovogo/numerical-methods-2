import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, List
from scipy.integrate import solve_ivp

@dataclass
class IVP:
    f: Callable[[float, float], float]
    a: float
    b: float
    y0: float

def implicit_euler(
    ivp: IVP,
    N: int,
    tol: float = 1e-4,
    max_iter: int = 1000,
    use_predictor: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:

    a, b, y0 = ivp.a, ivp.b, ivp.y0
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = np.empty(N + 1, dtype=float)
    y[0] = y0

    iters_per_step: List[int] = []

    for k in range(N):
        xk, xk1 = x[k], x[k + 1]
        # предиктор: один явный шаг — обычно улучшает сходимость
        z = y[k] + h * ivp.f(xk, y[k]) if use_predictor else y[k]

        it = 0
        while True:
            it += 1
            z_next = y[k] + h * ivp.f(xk1, z)
            if abs(z_next - z) <= tol:
                z = z_next
                break
            z = z_next
            if it >= max_iter:
                raise RuntimeError(
                    f"Не сошлось на шаге k={k} (x={xk1:.6g}) за {max_iter} итераций; "
                    f"уменьши шаг (увеличь N) или ослабь tol/max_iter."
                )
        y[k + 1] = z
        iters_per_step.append(it)

    return x, y, iters_per_step

def euler_explicit(ivp: IVP, N: int) -> Tuple[np.ndarray, np.ndarray]:
    a, b, y = ivp.a, ivp.b, ivp.y0
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y_arr = np.empty(N + 1, dtype=float)
    y_arr[0] = y
    for i in range(N):
        y = y + h * ivp.f(x[i], y)
        y_arr[i + 1] = y
    return x, y_arr

def double_count_error(
    ivp: IVP,
    N: int,
    tol: float = 1e-4,
    max_iter: int = 1000,
):
    x_h, y_h, _ = implicit_euler(ivp, N, tol=tol, max_iter=max_iter)
    x_h2, y_h2, _ = implicit_euler(ivp, 2 * N, tol=tol, max_iter=max_iter)
    s = 1
    err = np.abs(y_h2[::2] - y_h) / (2**s - 1)  # тут деление на 1, оставлено для ясности формы
    return x_h, y_h, err

def solve_and_plot(
    ivp: IVP,
    N: int,
    title: str = "Неявный метод Эйлера",
    tol: float = 1e-4,
    max_iter: int = 1000,
    show_iters_stats: bool = True,
):
    x, y, err = double_count_error(ivp, N, tol=tol, max_iter=max_iter)
    max_err = float(np.max(err))
    imax = int(np.argmax(err))
    xmax = float(x[imax])

    # Эталон solve_ivp в тех же узлах
    def f_ivp(t, y_vec):
        return [ivp.f(t, y_vec[0])]
    ref = solve_ivp(f_ivp, (ivp.a, ivp.b), [ivp.y0], t_eval=x)

    # График решения
    plt.figure()
    plt.plot(x, y, label="Неявный Эйлер", linewidth=3)
    plt.plot(ref.t, ref.y[0], "--", label="Эталон (solve_ivp)", linewidth=2)
    plt.xlabel("x"); plt.ylabel("y"); plt.title(title); plt.legend(); plt.grid(True)

    # График погрешности (двойной счёт)
    plt.figure()
    plt.plot(x, err, label="Оценка погрешности (двойной счёт)")
    plt.scatter([xmax], [max_err], marker="o", label=f"max ≈ {max_err:.3e} при x≈{xmax:.3g}")
    plt.xlabel("x"); plt.ylabel("|ошибка|"); plt.title("Погрешность")
    plt.yscale("log"); plt.legend(); plt.grid(True, which="both")

    print(f"Максимальная оценка погрешности на интервале: {max_err:.6e} при x ≈ {xmax:.6g}")
    plt.show()

ivp = IVP(
    f=lambda x, y: math.exp(-x) * math.cos(x - y**2),  # <--- Можешь заменить на свою f(x,y)
    a=0.0,
    b=0.8,
    y0=0.5
)

N = 200
solve_and_plot(ivp, N, title="Неявный Эйлер для y' = e^{-x} cos(x - y^2),  y(0)=0.5,  [0,0.8]")
