from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp


@dataclass
class IVP:
    f: Callable[[float, float], float]
    a: float
    b: float
    y0: float
    n: int

def make_grid(a: float, b: float, n: int) -> Tuple[np.ndarray, float]:
    x = np.linspace(a, b, n + 1, dtype=float)
    h = (b - a) / n
    return x, h

def rk4_bootstrap(ivp: IVP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, h = make_grid(ivp.a, ivp.b, ivp.n)
    y = np.empty_like(x)
    y[0] = ivp.y0
    # рассчитать узлы 1..3
    steps_needed = min(3, ivp.n)
    for i in range(steps_needed):
        xi, yi = x[i], y[i]
        k1 = ivp.f(xi, yi)
        k2 = ivp.f(xi + 0.5*h, yi + 0.5*h*k1)
        k3 = ivp.f(xi + 0.5*h, yi + 0.5*h*k2)
        k4 = ivp.f(xi + h,     yi + h*k3)
        y[i+1] = yi + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    fvals = np.array([ivp.f(x[i], y[i]) for i in range(steps_needed+1)] + [np.nan]*(len(x)-(steps_needed+1)))
    return x, y, fvals


def adams_bashforth_4(ivp: IVP, x_boot: np.ndarray, y_boot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Явный Адамс (AB4): y_{n+1} = y_n + h/24*(55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})."""
    x, h = make_grid(ivp.a, ivp.b, ivp.n)
    y = np.empty_like(x)
    y[:4] = y_boot[:4]
    fvals = np.array([ivp.f(x[i], y[i]) for i in range(4)] + [np.nan]*(len(x)-4))
    for i in range(3, ivp.n):
        y[i+1] = y[i] + (h/24.0)*(55*fvals[i] - 59*fvals[i-1] + 37*fvals[i-2] - 9*fvals[i-3])
        fvals[i+1] = ivp.f(x[i+1], y[i+1])
    return x, y


def adams_moulton_4(
    ivp: IVP,
    x_boot: np.ndarray,
    y_boot: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Неявный Адамс (AM4): y_{n+1} = y_n + h/24*(9 f_{n+1} + 19 f_n - 5 f_{n-1} + f_{n-2})."""
    x, h = make_grid(ivp.a, ivp.b, ivp.n)
    y = np.empty_like(x)
    y[:4] = y_boot[:4]
    fvals = np.array([ivp.f(x[i], y[i]) for i in range(4)] + [np.nan]*(len(x)-4))
    iters: List[int] = []

    for i in range(3, ivp.n):
        # предиктор AB4
        y_pred = y[i] + (h/24.0)*(55*fvals[i] - 59*fvals[i-1] + 37*fvals[i-2] - 9*fvals[i-3])

        # фикс-точка AM4
        z = y_pred
        it = 0
        while True:
            it += 1
            rhs = 9*ivp.f(x[i+1], z) + 19*fvals[i] - 5*fvals[i-1] + fvals[i-2]
            z_next = y[i] + (h/24.0)*rhs
            if abs(z_next - z) <= tol:
                z = z_next
                break
            z = z_next
            if it >= max_iter:
                raise RuntimeError(f"AM4: нет сходимости на i={i}, x={x[i+1]:.6g}")
        y[i+1] = z
        fvals[i+1] = ivp.f(x[i+1], y[i+1])
        iters.append(it)

    return x, y, iters


def reference_solve_ivp(ivp: IVP, method: str = "RK45") -> Tuple[np.ndarray, np.ndarray]:
    x, _ = make_grid(ivp.a, ivp.b, ivp.n)
    def rhs(t, Y): return ivp.f(t, Y[0])
    sol = solve_ivp(rhs, (ivp.a, ivp.b), [ivp.y0], t_eval=x, rtol=1e-12, atol=1e-14, method=method)
    return x, sol.y[0]


def solve_all(ivp: IVP):
    x_boot, y_boot, _ = rk4_bootstrap(ivp)
    x_ab4, y_ab4 = adams_bashforth_4(ivp, x_boot, y_boot)
    x_am4, y_am4, iters = adams_moulton_4(ivp, x_boot, y_boot)
    x_ref, y_ref = reference_solve_ivp(ivp, method="RK45")
    assert np.allclose(x_ab4, x_am4) and np.allclose(x_ab4, x_ref)
    delta = np.abs(y_ab4 - y_am4)  # формула (11)
    return x_ref, y_ab4, y_am4, y_ref, delta, iters

def plot_all(x, y_ab4, y_am4, y_ref, delta, title_suffix=""):
    # Решения
    plt.figure()
    plt.plot(x, y_ab4, label="Явный Адамс", linewidth=3)
    plt.plot(x, y_am4, "--", label="Неявный Адамс)", linewidth=3)
    plt.plot(x, y_ref, "-", label="solve_ivp (ref)", linewidth=2, alpha=0.8)
    plt.xlabel("x"); plt.ylabel("y(x)")
    plt.title(f"Адамс 4-го порядка + reference {title_suffix}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

    # Разность AB4–AM4 (формула 11)
    plt.figure()
    plt.plot(x, delta, marker='o', linewidth=1, label=r"$\Delta_i=|y_{AB4}-y_{AM4}|$")
    imax = int(np.argmax(delta))
    plt.scatter([x[imax]], [delta[imax]], s=30, zorder=3,
                label=f"Δ_max≈{float(delta[imax]):.3e} при x≈{float(x[imax]):.3g}")
    plt.yscale("log")
    plt.xlabel("x"); plt.ylabel("ошибка")
    plt.title("Явный Адамс vs Неявный Адамс: разность решений")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()

    print(f"Δ_max (AB4 vs AM4) = {float(delta.max()):.6e} в x ≈ {float(x[imax]):.6g}")


if __name__ == "__main__":
    f = lambda x, y: math.exp(-x) * math.cos(x - y**2)
    ivp = IVP(f=f, a=0.0, b=0.8, y0=0.5, n=12)

    x, y_ab4, y_am4, y_ref, delta, iters = solve_all(ivp)
    plot_all(x, y_ab4, y_am4, y_ref, delta)
    plt.show()
