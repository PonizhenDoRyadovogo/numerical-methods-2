import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple
from scipy.integrate import solve_ivp

@dataclass
class IVP2:
    F: Callable[[float, float, float], float]
    a: float
    b: float
    y0: float
    yp0: float

def euler_second_order(ivp: IVP2, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, y, yp = ivp.a, ivp. b, ivp.y0, ivp.yp0
    N = int(round(b-a)/h)
    x = np.linspace(a, a + N*h, N+1, dtype=float)
    y_arr = np.empty(N+1, dtype=float)
    yp_arr = np.empty(N+1, dtype=float)
    y_arr[0], yp_arr[0] = y, yp

    for i in range(N):
        y_next = y + h * yp
        yp_next = yp + h * ivp.F(x[i], y, yp)
        y, yp = y_next, yp_next
        y_arr[i+1] = y
        yp_arr[i+1] = yp
    return x, y_arr, yp_arr

def double_count_error(ivp: IVP2, h: int):
    x_h, y_h, yp_h = euler_second_order(ivp, h)
    x_h2, y_h2, yp_h2 = euler_second_order(ivp, h/2)

    y_h2_on_h = y_h2[::2]
    yp_h2_on_h = yp_h2[::2]

    err_y = np.abs(y_h - y_h2_on_h)
    err_yp = np.abs(yp_h - yp_h2_on_h)
    return (x_h, y_h, yp_h), (x_h2, y_h2, yp_h2), (err_y, err_yp)

def find_max_h(ivp: IVP2, tol=5e-4, h_start=1e-3, grow=2.0, max_growth_iters=25, bisect_iters=25):
    h_good, h_bad = None, None
    h = h_start
    for _ in range(max_growth_iters):
        try:
            (_,_,_),(_,_,_),(R_y,_) = double_count_error(ivp, h)
            err = float(np.max(R_y))
        except FloatingPointError:
            err = float("inf")
        if np.isnan(err) or err > tol:
            h_bad = h
            break
        h_good = h
        h *= grow
    if h_good is None:
        raise RuntimeError("Даже стартовый шаг не проходит допуск — уменьшите h_start.")
    if h_bad is None:
        return h_good

    left, right = h_good, h_bad
    for _ in range(bisect_iters):
        mid = 0.5*(left+right)
        (_, _, _), (_, _, _), (R_y, _) = double_count_error(ivp, mid)
        if np.max(R_y) <= tol:
            left = mid
        else:
            right = mid
    return left

def solve_and_plot(ivp: IVP2, h: int, title: str="Эйлер для y'' = F(x,y,y')"):
    (x, y, yp), (_, _, _), (err_y, err_yp) = double_count_error(ivp, h)
    max_err = float(np.max(err_y))
    imax = int(np.argmax(err_y))
    x_max = x[imax]

    def f_system(t, Y):
        y, yp = Y
        return [yp, ivp.F(t, y, yp)]
    sol = solve_ivp(f_system, (ivp.a, ivp.b), [ivp.y0, ivp.yp0], t_eval=x)

    plt.figure()
    plt.plot(x, y, label=f"Эйлер", linewidth=2)
    plt.plot(sol.t, sol.y[0], "--", label="Референс")
    plt.xlabel("x"); plt.ylabel("y"); plt.title(title); plt.legend(); plt.grid(True, which="both", alpha=0.3)

    plt.figure()
    plt.plot(x, err_y, label="|y_h - y_{h/2}|")
    plt.scatter([x_max], [max_err], marker="o", label=f"max ≈ {max_err:.3e} при x≈{x_max:.3g}")
    plt.xlabel("x");
    plt.ylabel("узловая оценка ошибки")
    plt.yscale("log")
    plt.title("Погрешность")
    plt.legend();
    plt.grid(True, which="both", alpha=0.3)

    plt.show()
    print(f"Максимальная оценка погрешности по y: {max_err:.4e} при x = {x_max:.4g}")


ivp = IVP2(
    F = lambda x, y, yp: math.exp(-x) * math.tan(x - y) - yp,
    a = 0.0,
    b = 1,
    y0 = 0.4,
    yp0 = 0.95
)

h_opt = find_max_h(ivp)
solve_and_plot(ivp, h_opt, title="График явного метода Эйлера для решения ДУ 2-го порядка и график встроенного решения")
print(f"Оптимальный шаг: {h_opt:.4g}")