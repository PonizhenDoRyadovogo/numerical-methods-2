import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CNProblem:
    A: np.ndarray
    x_T: np.ndarray
    T: float
    N: int = 10


def step_matrix(A: np.ndarray, h: float) -> np.ndarray:
    n = A.shape[0]
    I = np.eye(n)
    Ah = h * A
    Ah2 = Ah @ Ah
    B = I + 0.5 * Ah + (1.0 / 12.0) * Ah2
    C = I - 0.5 * Ah + (1.0 / 12.0) * Ah2
    Lambda = np.linalg.solve(C, B)
    return Lambda


def fundamental_matrix_and_step(problem: CNProblem) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Считает фундаментальную матрицу Φ(0,T) ≈ Λ(h)^{2^N} повторным возведением в квадрат.
    Возвращает Φ, Λ(h), h, M=2^N.
    """
    A, T, N = problem.A, problem.T, problem.N
    M = 2 ** N
    h = T / M
    Lambda = step_matrix(A, h)

    # повторное возведение в квадрат: после N шагов Φ = Λ^{2^N}
    Φ = Lambda.copy()
    for _ in range(N):
        Φ = Φ @ Φ

    return Φ, Lambda, h, M


def solve_CN(problem: CNProblem):
    Phi, Lambda, h, M = fundamental_matrix_and_step(problem)
    x0 = np.linalg.solve(Phi, problem.x_T)

    n = problem.A.shape[0]
    X = np.zeros((n, M + 1), dtype=float)
    X[:, 0] = x0
    for k in range(M):
        X[:, k + 1] = Lambda @ X[:, k]

    t = np.linspace(0.0, problem.T, M + 1)
    return t, X, x0, h


def double_count_error(problem: CNProblem):
    # решение с шагом h (N)
    t_h, X_h, x0_h, h = solve_CN(problem)

    # решение с шагом h/2 (N+1)
    fine_prob = CNProblem(A=problem.A, x_T=problem.x_T, T=problem.T, N=problem.N + 1)
    t_h2, X_h2, x0_h2, h2 = solve_CN(fine_prob)

    if not np.isclose(h2, h / 2):
        print("Предупреждение: h2 != h/2 (что-то не так с N).")

    M = X_h.shape[1] - 1  # число шагов на грубой сетке
    n = X_h.shape[0]

    # сравниваем узлы t_i: t_h[i] == t_h2[2*i]
    err = np.zeros(M + 1)
    for i in range(M + 1):
        coarse = X_h[:, i]
        fine = X_h2[:, 2 * i]  # соответствующий узел на мелкой сетке
        err[i] = np.max(np.abs(coarse - fine)) / 31.0

    return t_h, X_h, err

def plot_solution(t: np.ndarray, X: np.ndarray, title: str = "Решение x(t)"):
    plt.figure()
    for j in range(X.shape[0]):
        plt.plot(t, X[j], label=f"x_{j+1}(t)", linewidth=2)
    plt.xlabel("t"); plt.ylabel("x(t)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_error(t: np.ndarray, err: np.ndarray, title: str = "Оценка погрешности (двойной счёт)"):
    plt.figure()
    plt.plot(t, err, marker="o", linewidth=1, label="Δ_i ≈ max |x^{(h)}-x^{(h/2)}| / 31")
    i_max = int(np.argmax(err))
    plt.scatter([t[i_max]], [err[i_max]], s=30, zorder=3,
                label=f"Δ_max≈{err[i_max]:.3e} при t≈{t[i_max]:.3g}", color="black")
    plt.yscale("log")
    plt.xlabel("t"); plt.ylabel("ошибка")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    print(f"Δ_max ≈ {err[i_max]:.6e} при t ≈ {t[i_max]:.6g}")

if __name__ == "__main__":
    A = np.array([[-0.25,  -1.1],
                  [0.1, -0.2]], dtype=float)
    x_T = np.array([2.8, -3.1], dtype=float)
    T = 8.2
    N = 10

    problem = CNProblem(A=A, x_T=x_T, T=T, N=N)

    # Решение и оценка погрешности
    t, X, x0, h = solve_CN(problem)
    print("Приближенное x(0):", x0)
    print("Шаг h =", h, ", число шагов =", X.shape[1]-1)

    t_h, X_h, err = double_count_error(problem)

    # Графики
    plot_solution(t_h, X_h, title="Метод Кранка–Никольсона 5-го порядка: решение")
    plot_error(t_h, err, title="Метод Кранка–Никольсона 5-го порядка: двойной счёт")

    plt.show()
