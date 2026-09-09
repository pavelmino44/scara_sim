#!/usr/bin/env python3

import os
import csv
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


# ============================================================
# ПАРАМЕТРЫ НОВОГО SCARA
# ============================================================

L1 = 0.40          # длина первого звена, м
L2 = 0.30          # длина второго звена, м

M1 = 2.0           # масса первого звена, кг
M2 = 1.5           # масса второго звена, кг

l1 = L1 / 2.0      # положение центра масс первого звена
l2 = L2 / 2.0      # положение центра масс второго звена

# Моменты инерции относительно оси вращения z.
#
# Для цилиндра, ориентированного вдоль x:
# Izz = (1/12) * m * (3*r^2 + L^2)
#
# r = 0.025 м

RADIUS = 0.025

I1zz = (1.0 / 12.0) * M1 * (
    3.0 * RADIUS**2 + L1**2
)

I2zz = (1.0 / 12.0) * M2 * (
    3.0 * RADIUS**2 + L2**2
)

# Вязкое трение.
# Должно соответствовать модели, используемой в расчёте.
B1 = 0.1
B2 = 0.1


# ============================================================
# ПАРАМЕТРЫ ТРАЕКТОРИИ
# ============================================================

SIDE = 0.50        # сторона П, м
V = 0.05           # скорость, м/с = 5 см/с

DT = 0.001         # шаг расчёта траектории, с

# Начальная точка П
START_X = -0.25
START_Y = -0.25

# Высота инструмента относительно world.
# В URDF:
# world -> base_link: 0.10 м
# joint2: + radius*2 = +0.05 м
# поэтому tool0 находится примерно на z = 0.15 м.
TOOL_Z = 0.15


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class TrajectoryPoint:
    t: float

    x: float
    y: float

    x_dot: float
    y_dot: float

    x_ddot: float
    y_ddot: float

    q1: float
    q2: float

    q1_dot: float
    q2_dot: float

    q1_ddot: float
    q2_ddot: float

    tau1: float
    tau2: float


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def wrap_to_pi(angle):
    """Привести угол к диапазону [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_angle(angle, previous):
    """
    Выбрать эквивалентный угол, ближайший к предыдущему.
    Это предотвращает скачки q на +/- 2*pi.
    """
    return previous + wrap_to_pi(angle - previous)


# ============================================================
# ОБРАТНАЯ КИНЕМАТИКА
# ============================================================

def inverse_kinematics(x, y, previous_q=None):
    """
    Обратная кинематика плоского 2R манипулятора.

    Используется ветвь q2 > 0.
    """

    D = (
        x**2
        + y**2
        - L1**2
        - L2**2
    ) / (2.0 * L1 * L2)

    # Защита от ошибки округления
    D = np.clip(D, -1.0, 1.0)

    # Выбираем elbow-up ветвь
    q2 = math.acos(D)

    q1 = math.atan2(y, x) - math.atan2(
        L2 * math.sin(q2),
        L1 + L2 * math.cos(q2)
    )

    # В начале просто возвращаем решение
    if previous_q is None:
        return q1, q2

    # Затем сохраняем непрерывность углов
    q1 = unwrap_angle(q1, previous_q[0])
    q2 = unwrap_angle(q2, previous_q[1])

    return q1, q2


# ============================================================
# КИНЕМАТИКА
# ============================================================

def forward_kinematics(q1, q2):
    """Прямая кинематика."""
    x = (
        L1 * math.cos(q1)
        + L2 * math.cos(q1 + q2)
    )

    y = (
        L1 * math.sin(q1)
        + L2 * math.sin(q1 + q2)
    )

    return x, y


def jacobian(q1, q2):
    """Якобиан 2R манипулятора."""

    return np.array([
        [
            -L1 * math.sin(q1)
            - L2 * math.sin(q1 + q2),

            -L2 * math.sin(q1 + q2)
        ],

        [
            L1 * math.cos(q1)
            + L2 * math.cos(q1 + q2),

            L2 * math.cos(q1 + q2)
        ]
    ])


def jacobian_dot(q1, q2, q1_dot, q2_dot):
    """Производная Якобиана по времени."""

    s12 = math.sin(q1 + q2)
    c12 = math.cos(q1 + q2)

    return np.array([
        [
            -L1 * math.cos(q1) * q1_dot
            - L2 * c12 * (q1_dot + q2_dot),

            -L2 * c12 * (q1_dot + q2_dot)
        ],

        [
            -L1 * math.sin(q1) * q1_dot
            - L2 * s12 * (q1_dot + q2_dot),

            -L2 * s12 * (q1_dot + q2_dot)
        ]
    ])


# ============================================================
# ДИНАМИКА ЛАГРАНЖА
# ============================================================

def mass_matrix(q1, q2):
    """
    Матрица инерции M(q).

        M(q) q_ddot + C(q,q_dot) q_dot + B q_dot = tau
    """

    c2 = math.cos(q2)

    M11 = (
        I1zz
        + I2zz
        + M1 * l1**2
        + M2 * (
            L1**2
            + l2**2
            + 2.0 * L1 * l2 * c2
        )
    )

    M12 = (
        I2zz
        + M2 * (
            l2**2
            + L1 * l2 * c2
        )
    )

    M22 = (
        I2zz
        + M2 * l2**2
    )

    return np.array([
        [M11, M12],
        [M12, M22]
    ])


def coriolis_vector(q1, q2, q1_dot, q2_dot):
    """
    Вектор центробежных/Кориолисовых членов.

    C(q,q_dot) q_dot.
    """

    s2 = math.sin(q2)

    h = M2 * L1 * l2 * s2

    c1 = (
        -h * q2_dot * (
            q1_dot + q2_dot
        )
    )

    c2 = h * q1_dot**2

    return np.array([c1, c2])


def calculate_torques(
    q1,
    q2,
    q1_dot,
    q2_dot,
    q1_ddot,
    q2_ddot
):
    """
    Расчёт требуемых моментов:

        tau = M(q) q_ddot
              + C(q,q_dot) q_dot
              + B q_dot

    Для горизонтального SCARA
    гравитационные моменты относительно
    вертикальных осей отсутствуют.
    """

    M = mass_matrix(q1, q2)

    q_ddot = np.array([
        q1_ddot,
        q2_ddot
    ])

    Cqdot = coriolis_vector(
        q1,
        q2,
        q1_dot,
        q2_dot
    )

    damping = np.array([
        B1 * q1_dot,
        B2 * q2_dot
    ])

    tau = (
        M @ q_ddot
        + Cqdot
        + damping
    )

    return tau[0], tau[1]


# ============================================================
# ГЕНЕРАЦИЯ П-ОБРАЗНОЙ ТРАЕКТОРИИ
# ============================================================

def generate_segment(
    start,
    end,
    t_start,
    previous_q
):
    """
    Генерирует один прямолинейный участок
    с постоянной Cartesian скоростью V.

    Важно:
    в углах П направление скорости меняется
    скачкообразно.

    Мы НЕ создаём искусственный
    импульсный момент.

    Момент рассчитывается только для
    текущего участка.
    """

    x0, y0 = start
    x1, y1 = end

    dx = x1 - x0
    dy = y1 - y0

    distance = math.sqrt(dx**2 + dy**2)

    duration = distance / V

    direction_x = dx / distance
    direction_y = dy / distance

    num_points = int(round(duration / DT))

    points = []

    q_previous = previous_q

    for i in range(num_points + 1):

        local_t = min(i * DT, duration)

        # Для последней точки гарантируем точное попадание
        if local_t >= duration:
            s = distance
            x_dot = 0.0
            y_dot = 0.0

        else:
            s = V * local_t
            x_dot = V * direction_x
            y_dot = V * direction_y

        x = x0 + direction_x * s
        y = y0 + direction_y * s

        # На прямом участке Cartesian acceleration = 0
        x_ddot = 0.0
        y_ddot = 0.0

        # В последний момент участка скорость
        # здесь обнуляется только в точке.
        #
        # Это НЕ физическое плавное торможение.
        # Оно нужно, чтобы не передавать в следующий
        # участок скорость предыдущего направления.
        if local_t >= duration:
            x_dot = 0.0
            y_dot = 0.0

        q1, q2 = inverse_kinematics(
            x,
            y,
            q_previous
        )

        q_previous = (q1, q2)

        J = jacobian(q1, q2)

        det_J = np.linalg.det(J)

        if abs(det_J) < 1e-6:
            raise RuntimeError(
                f"Jacobian is close to singularity: "
                f"det(J)={det_J}"
            )

        cartesian_velocity = np.array([
            x_dot,
            y_dot
        ])

        q_dot = np.linalg.solve(
            J,
            cartesian_velocity
        )

        # Получаем q_ddot из:
        #
        # x_ddot = J q_ddot + J_dot q_dot
        #
        # q_ddot = J^-1 (x_ddot - J_dot q_dot)

        J_dot = jacobian_dot(
            q1,
            q2,
            q_dot[0],
            q_dot[1]
        )

        cartesian_acceleration = np.array([
            x_ddot,
            y_ddot
        ])

        q_ddot = np.linalg.solve(
            J,
            cartesian_acceleration
            - J_dot @ q_dot
        )

        tau1, tau2 = calculate_torques(
            q1,
            q2,
            q_dot[0],
            q_dot[1],
            q_ddot[0],
            q_ddot[1]
        )

        points.append(
            TrajectoryPoint(
                t=t_start + local_t,

                x=x,
                y=y,

                x_dot=x_dot,
                y_dot=y_dot,

                x_ddot=x_ddot,
                y_ddot=y_ddot,

                q1=q1,
                q2=q2,

                q1_dot=q_dot[0],
                q2_dot=q_dot[1],

                q1_ddot=q_ddot[0],
                q2_ddot=q_ddot[1],

                tau1=tau1,
                tau2=tau2
            )
        )

    return points, q_previous, duration


def generate_trajectory():
    """
    Полная П-образная траектория:

        (-0.25, -0.25)
              |
              |
              |
        (-0.25, 0.25) ----> (0.25, 0.25)
                              |
                              |
                              |
                        (0.25, -0.25)
    """

    corners = [
        (START_X, START_Y),
        (START_X, START_Y + SIDE),
        (START_X + SIDE, START_Y + SIDE),
        (START_X + SIDE, START_Y)
    ]

    trajectory = []

    # Начальное положение вычисляем отдельно
    q_initial = inverse_kinematics(
        START_X,
        START_Y
    )

    q_previous = q_initial

    current_time = 0.0

    for segment_index in range(3):

        start = corners[segment_index]
        end = corners[segment_index + 1]

        segment_points, q_previous, duration = (
            generate_segment(
                start,
                end,
                current_time,
                q_previous
            )
        )

        # Не добавляем повторно начальную точку
        # последующих сегментов.
        if segment_index > 0:
            segment_points = segment_points[1:]

        trajectory.extend(segment_points)

        current_time += duration

    return trajectory


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

def save_trajectory(trajectory):
    package_path = get_package_share_directory(
        'scara_sim'
    )

    data_dir = os.path.join(
        package_path,
        'data'
    )

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    csv_path = os.path.join(
        data_dir,
        'trajectory.csv'
    )

    with open(
        csv_path,
        'w',
        newline=''
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            't',

            'x',
            'y',

            'x_dot',
            'y_dot',

            'x_ddot',
            'y_ddot',

            'q1',
            'q2',

            'q1_dot',
            'q2_dot',

            'q1_ddot',
            'q2_ddot',

            'tau1',
            'tau2'
        ])

        for p in trajectory:
            writer.writerow([
                f'{p.t:.6f}',

                f'{p.x:.9f}',
                f'{p.y:.9f}',

                f'{p.x_dot:.9f}',
                f'{p.y_dot:.9f}',

                f'{p.x_ddot:.9f}',
                f'{p.y_ddot:.9f}',

                f'{p.q1:.9f}',
                f'{p.q2:.9f}',

                f'{p.q1_dot:.9f}',
                f'{p.q2_dot:.9f}',

                f'{p.q1_ddot:.9f}',
                f'{p.q2_ddot:.9f}',

                f'{p.tau1:.9f}',
                f'{p.tau2:.9f}'
            ])

    return csv_path


# ============================================================
# ГРАФИКИ
# ============================================================

def plot_trajectory(trajectory):

    t = np.array([p.t for p in trajectory])

    x = np.array([p.x for p in trajectory])
    y = np.array([p.y for p in trajectory])

    q1 = np.array([p.q1 for p in trajectory])
    q2 = np.array([p.q2 for p in trajectory])

    tau1 = np.array([p.tau1 for p in trajectory])
    tau2 = np.array([p.tau2 for p in trajectory])

    # --------------------------------------------------------
    # XY trajectory
    # --------------------------------------------------------

    plt.figure()

    plt.plot(x, y)

    plt.plot(
        x[0],
        y[0],
        'o'
    )

    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.title('Desired SCARA trajectory')

    plt.axis('equal')
    plt.grid(True)

    # --------------------------------------------------------
    # Joint positions
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        t,
        q1,
        label='q1'
    )

    plt.plot(
        t,
        q2,
        label='q2'
    )

    plt.xlabel('Time, s')
    plt.ylabel('Angle, rad')

    plt.title('Joint positions')

    plt.grid(True)
    plt.legend()

    # --------------------------------------------------------
    # Moments
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        t,
        tau1,
        label='tau1'
    )

    plt.plot(
        t,
        tau2,
        label='tau2'
    )

    plt.xlabel('Time, s')
    plt.ylabel('Torque, N*m')

    plt.title('Lagrange inverse dynamics')

    plt.grid(True)
    plt.legend()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print('Generating trajectory...')

    trajectory = generate_trajectory()

    print(
        f'Number of points: {len(trajectory)}'
    )

    print(
        f'Total time: {trajectory[-1].t:.3f} s'
    )

    print(
        f'Initial q1: {trajectory[0].q1:.6f} rad'
    )

    print(
        f'Initial q2: {trajectory[0].q2:.6f} rad'
    )

    csv_path = save_trajectory(
        trajectory
    )

    print(
        f'Trajectory saved to:\n{csv_path}'
    )

    plot_trajectory(
        trajectory
    )


if __name__ == '__main__':
    main()