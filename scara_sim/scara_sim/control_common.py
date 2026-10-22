#!/usr/bin/env python3
"""
Общие инструменты для управляемого движения SCARA.

Модуль не зависит от ROS, поэтому используется и в ROS-ноде
pid_controller, и в офлайн-симуляторе offline_pid_sim.

Содержимое:
    * wrap_to_pi            - приведение угла к [-pi, pi]
    * JointPID              - покоординатный ПИД-регулятор
    * Trajectory            - загрузка trajectory.csv и интерполяция
    * fk_planar             - прямая кинематика плоской цепи nR
    * DCMotorParams         - параметры ДПТ с редуктором
    * DCMotorBank           - модель ДПТ (напряжение -> момент на шарнире)
    * MotorFeedforward      - обратная модель ДПТ (момент -> напряжение)
"""

import csv
import math
import re

import numpy as np


# ============================================================
# УГЛЫ
# ============================================================

def wrap_to_pi(angle):
    """Привести угол (или массив углов) к диапазону [-pi, pi)."""

    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


# ============================================================
# ПИД-РЕГУЛЯТОР
# ============================================================

class JointPID:
    """
    Независимый ПИД-регулятор для каждого сочленения.

        u_pid = Kp * e + Ki * integral(e) + Kd * e_dot

    Выход u - момент (Н·м) при прямом управлении моментом
    или напряжение (В) при управлении через модель ДПТ.

        e     = q_des     - q
        e_dot = q_dot_des - q_dot

    Производная берётся не численным дифференцированием ошибки,
    а по измеренной скорости сочленения (state interface velocity).
    Это классическая форма "PD по скорости", она не усиливает шум
    и не даёт "удара производной" при скачке уставки.

    Защита от интегрального насыщения (anti-windup):
        1) интеграл ограничен по модулю integral_limit;
        2) при насыщении момента интегрирование в сторону
           насыщения прекращается (conditional integration).
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        integral_limit=None,
        output_limit=None,
    ):

        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)

        n = len(self.kp)

        if integral_limit is None:
            integral_limit = [np.inf] * n

        if output_limit is None:
            output_limit = [np.inf] * n

        self.integral_limit = np.asarray(integral_limit, dtype=float)
        self.torque_limit = np.asarray(output_limit, dtype=float)

        self.integral = np.zeros(n)

    def reset(self):
        self.integral[:] = 0.0

    def update(self, e, e_dot, dt, tau_ff=None):
        """
        Вернуть (tau_total, tau_pid).

        tau_ff - прямая связь (feedforward), добавляется к ПИД
        до ограничения момента, чтобы anti-windup учитывал
        суммарный момент.
        """

        e = np.asarray(e, dtype=float)
        e_dot = np.asarray(e_dot, dtype=float)

        if tau_ff is None:
            tau_ff = np.zeros_like(e)

        # Пробный интеграл
        integral_new = np.clip(
            self.integral + e * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        tau_pid = (
            self.kp * e
            + self.ki * integral_new
            + self.kd * e_dot
        )

        tau_raw = tau_ff + tau_pid

        tau = np.clip(tau_raw, -self.torque_limit, self.torque_limit)

        # Conditional integration: не накапливаем интеграл,
        # если момент упёрся в ограничение и ошибка тянет дальше.
        saturated = (tau != tau_raw) & (np.sign(e) == np.sign(tau_raw))

        self.integral = np.where(saturated, self.integral, integral_new)

        return tau, tau - tau_ff


# ============================================================
# ТРАЕКТОРИЯ
# ============================================================

class Trajectory:
    """
    Программная траектория из CSV, созданного trajectory_generator.

    Число сочленений определяется автоматически по столбцам
    q1, q2, ..., qN. Ожидаемые столбцы:

        t, x, y, q1..qN, q1_dot..qN_dot, tau1..tauN

    Остальные столбцы (ускорения, phi и т.п.) игнорируются.
    """

    def __init__(self, path):

        with open(path, 'r') as file:
            reader = csv.reader(file)
            header = [h.strip() for h in next(reader)]
            rows = [[float(v) for v in row] for row in reader if row]

        if not rows:
            raise RuntimeError(f'Trajectory CSV is empty: {path}')

        data = np.array(rows, dtype=float)
        col = {name: i for i, name in enumerate(header)}

        self.n = len([h for h in header if re.fullmatch(r'q\d+', h)])

        if self.n == 0:
            raise RuntimeError(f'No q1..qN columns in {path}')

        self.t = data[:, col['t']]

        self.q = np.column_stack(
            [data[:, col[f'q{i + 1}']] for i in range(self.n)])

        self.q_dot = np.column_stack(
            [data[:, col[f'q{i + 1}_dot']] for i in range(self.n)])

        self.tau = np.column_stack(
            [data[:, col[f'tau{i + 1}']] for i in range(self.n)])

        # Ускорения нужны для прямой связи по напряжению
        # (инерция ротора). Если столбцов нет - считаем численно.
        if all(f'q{i + 1}_ddot' in col for i in range(self.n)):
            self.q_ddot = np.column_stack(
                [data[:, col[f'q{i + 1}_ddot']] for i in range(self.n)])
        else:
            self.q_ddot = np.gradient(self.q_dot, self.t, axis=0)

        self.x = data[:, col['x']] if 'x' in col else None
        self.y = data[:, col['y']] if 'y' in col else None

        self.duration = float(self.t[-1])

    def _interp(self, t, table):
        return np.array([
            np.interp(t, self.t, table[:, i]) for i in range(self.n)
        ])

    def sample(self, t):
        """Вернуть (q_des, q_dot_des, tau_ff) в момент t."""

        return (
            self._interp(t, self.q),
            self._interp(t, self.q_dot),
            self._interp(t, self.tau),
        )

    def sample_xy(self, t):
        if self.x is None:
            return float('nan'), float('nan')

        return (
            float(np.interp(t, self.t, self.x)),
            float(np.interp(t, self.t, self.y)),
        )


# ============================================================
# КИНЕМАТИКА
# ============================================================

def fk_planar(q, lengths):
    """
    Прямая кинематика плоской цепи из n вращательных сочленений
    с вертикальными осями (горизонтальная SCARA).

        x = sum_i L_i cos(q1 + ... + qi)
        y = sum_i L_i sin(q1 + ... + qi)
    """

    theta = np.cumsum(q)

    x = float(np.sum(np.asarray(lengths) * np.cos(theta)))
    y = float(np.sum(np.asarray(lengths) * np.sin(theta)))

    return x, y


# ============================================================
# ДВИГАТЕЛЬ ПОСТОЯННОГО ТОКА С РЕДУКТОРОМ
# ============================================================
#
#   Электрическая часть:  L di/dt = U - R i - ke * w_m
#   Момент двигателя:     tau_m   = kt * i
#   Редуктор:             w_m     = N * q_dot
#                         tau     = eta * N * tau_m
#                                   - N^2 * (J_r * q_ddot + b_m * q_dot)
#
#   tau - момент, приложенный к шарниру (подаётся в Gazebo).
#   Слагаемое N^2 * J_r * q_ddot - приведённая инерция ротора,
#   N^2 * b_m * q_dot - приведённое вязкое трение в двигателе.
#
# Все параметры - массивы по числу сочленений.
# ============================================================

MOTOR_PARAM_NAMES = (
    'R', 'L', 'kt', 'ke', 'J_rotor', 'b_motor', 'gear_ratio', 'gear_eff', 'U_max',
)


class DCMotorParams:

    def __init__(self, **kw):
        n = None
        for name in MOTOR_PARAM_NAMES:
            v = np.asarray(kw[name], dtype=float)
            setattr(self, name, v)
            if n is None:
                n = len(v)
            elif len(v) != n:
                raise ValueError(f'Motor parameter {name} must have {n} values')
        self.n = n

    @classmethod
    def from_getter(cls, get, prefix='motor.'):
        """Собрать параметры функцией get(name) -> list."""
        return cls(**{name: get(prefix + name) for name in MOTOR_PARAM_NAMES})

    @property
    def tau_e(self):
        """Электрическая постоянная времени L/R, с."""
        return self.L / self.R

    @property
    def reflected_inertia(self):
        return self.gear_ratio**2 * self.J_rotor

    @property
    def stall_torque(self):
        """Момент на шарнире при U_max и q_dot = 0."""
        return self.gear_eff * self.gear_ratio * self.kt * self.U_max / self.R

    @property
    def no_load_speed(self):
        """Скорость шарнира холостого хода при U_max, рад/с."""
        return self.U_max / (self.ke * self.gear_ratio)

    def describe(self):
        return (
            f'R={self.R.tolist()} Ohm, L={self.L.tolist()} H, '
            f'kt={self.kt.tolist()}, ke={self.ke.tolist()}, '
            f'J_r={self.J_rotor.tolist()}, N={self.gear_ratio.tolist()}, '
            f'eta={self.gear_eff.tolist()}, U_max={self.U_max.tolist()} V\n'
            f'  stall torque (joint) = {np.round(self.stall_torque, 2).tolist()} N*m, '
            f'no-load speed (joint) = {np.round(self.no_load_speed, 2).tolist()} rad/s, '
            f'L/R = {np.round(1000 * self.tau_e, 3).tolist()} ms')


class DCMotorBank:
    """
    Модель n двигателей. Электрическое уравнение интегрируется
    точно (экспонента) на каждом шаге dt при постоянных U и w_m,
    поэтому устойчиво при любом соотношении L/R и dt.
    """

    def __init__(self, params: DCMotorParams):
        self.p = params
        self.i = np.zeros(params.n)

    def reset(self):
        self.i[:] = 0.0

    def step(self, U, q_dot, q_ddot, dt):
        """
        Сделать шаг dt. Вернуть словарь:
            U        - фактическое напряжение (после ограничения)
            i        - ток в конце шага
            i_avg    - средний ток за шаг
            tau_m    - момент на валу двигателя (средний)
            tau      - момент на шарнире
        """

        p = self.p

        U = np.clip(np.asarray(U, dtype=float), -p.U_max, p.U_max)

        w_m = p.gear_ratio * np.asarray(q_dot, dtype=float)

        i_ss = (U - p.ke * w_m) / p.R
        a = np.exp(-dt / p.tau_e)

        i0 = self.i
        i_avg = i_ss + (i0 - i_ss) * p.tau_e / dt * (1.0 - a)
        self.i = i_ss + (i0 - i_ss) * a

        tau_m = p.kt * i_avg

        tau = (
            p.gear_eff * p.gear_ratio * tau_m
            - p.gear_ratio**2 * (p.J_rotor * np.asarray(q_ddot, dtype=float)
                                 + p.b_motor * np.asarray(q_dot, dtype=float))
        )

        return dict(U=U, i=self.i.copy(), i_avg=i_avg, tau_m=tau_m, tau=tau)


class AccelerationEstimator:
    """
    Оценка ускорения шарнира по измеренной скорости:
    разностная производная + фильтр 1-го порядка.
    Нужна, чтобы учесть инерцию ротора N^2*J_r*q_ddot,
    которой нет в модели Gazebo.
    """

    def __init__(self, n, cutoff_hz=100.0):
        self.cutoff = cutoff_hz
        self.prev = None
        self.a = np.zeros(n)

    def update(self, q_dot, dt):
        q_dot = np.asarray(q_dot, dtype=float)
        if self.prev is None or dt <= 0.0:
            self.prev = q_dot.copy()
            return self.a
        raw = (q_dot - self.prev) / dt
        self.prev = q_dot.copy()
        alpha = dt / (dt + 1.0 / (2.0 * math.pi * self.cutoff))
        self.a = self.a + alpha * (raw - self.a)
        return self.a


class MotorFeedforward:
    """
    Прямая связь по напряжению (обратная модель ДПТ).

    По заданным q_dot_des, q_ddot_des и моменту обратной динамики
    робота tau_ff (из trajectory.csv) вычисляется напряжение:

        tau_m = (tau_ff + N^2 (J_r q_ddot + b_m q_dot)) / (eta N)
        i     = tau_m / kt
        U_ff  = R i + L di/dt + ke N q_dot

    Таблица U_ff(t) считается один раз при загрузке траектории.
    """

    def __init__(self, traj: Trajectory, p: DCMotorParams):
        self.traj = traj
        N = p.gear_ratio

        tau_m = (traj.tau + N**2 * (p.J_rotor * traj.q_ddot + p.b_motor * traj.q_dot)) \
            / (p.gear_eff * N)
        i = tau_m / p.kt
        di = np.gradient(i, traj.t, axis=0)

        self.i = i
        self.U = p.R * i + p.L * di + p.ke * N * traj.q_dot

    def sample(self, t):
        return np.array([
            np.interp(t, self.traj.t, self.U[:, k]) for k in range(self.traj.n)
        ])
