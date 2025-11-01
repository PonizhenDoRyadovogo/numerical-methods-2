import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# параметры
c0 = 2.1
z0 = 22.0
v0 = 19.0
w0 = 125.0
T  = 0.9
tau = 0.3

p  = 400.0
k0 = 200.0
k1 = 0.0001
k2 = 0.12
k3 = 0.09
c  = 8.0
u  = 20.0
t0 = 0.0

h = 0.001  # шаг

def sales(t: float, z: float, v: float) -> float:
    return k0 * np.exp(-c) * (1.0 - v / p) * z

def solveEuler() -> Tuple[np.ndarray, np.ndarray]:
    n = int(round((T + tau) / h))
    t = np.linspace(-tau, T, n + 1)
    x = np.zeros((3, n + 1), dtype=float)  # [z; v; w]
    s = np.zeros(n + 1, dtype=float)

    # сколько шагов в задержке
    first_interval = int(round(tau / h))   # <-- фикс off-by-one

    # история на [-tau, 0]
    hist_slice = slice(0, first_interval + 1)  # включает узел t≈0
    tt = t[hist_slice] - t0
    x[:, hist_slice] = np.vstack([
        z0 + 3.0 * tt,          # z(t)
        v0 - 1.0 * tt,          # v(t)
        w0 - 0.5 * tt,          # w(t)
    ])

    # s(t) на истории
    s[hist_slice] = sales(t[hist_slice], x[0, hist_slice], x[1, hist_slice])

    # явный Эйлер на (0, T]
    for i in range(first_interval, n):     # <-- начинаем с i = first_interval
        z_prev = x[0, i]
        v_prev = x[1, i]
        w_prev = x[2, i]

        # s(t_i) и s(t_i - tau)
        s_now   = s[i]
        s_delay = s[i - first_interval]    # соответствует времени t[i] - tau

        # шаги Эйлера
        x[0, i+1] = z_prev + h * (-k1 * z_prev + u - s_now)
        x[1, i+1] = v_prev + h * (-k2 * v_prev + s_now)
        x[2, i+1] = w_prev + h * ( c * s_delay - c0 * u - k3 * z_prev)  # проверь знаки по твоему PDF

        # s на следующем узле (использует новые z,v при времени t[i+1])
        s[i+1] = sales(t[i+1], x[0, i+1], x[1, i+1])

    return x, t

x, t = solveEuler()

fig, ax = plt.subplots(3, sharex=True, figsize=(10,6))
ax[0].plot(t, x[0], linewidth=2)
ax[0].set_ylabel('z(t)')

ax[1].plot(t, x[1], linewidth=2)
ax[1].set_ylabel('v(t)')

ax[2].plot(t, x[2], linewidth=2)
ax[2].set_ylabel('w(t)')
ax[2].set_xlabel('t')

for a in ax:
    a.grid(True)
    a.axvline(x=0.0, color='r', linestyle='--', linewidth=1, label='t0=0')
ax[0].legend(loc='best')

plt.tight_layout()
plt.show()
