import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ament_index_python.packages import get_package_share_directory


# ============================================================
# ПАРАМЕТРЫ РОБОТА
# ============================================================

# Длины звеньев.
# Они должны полностью совпадать с scara.urdf.xacro.
L1 = 0.40  # м
L2 = 0.30  # м

# Радиус цилиндрических звеньев.
radius = 0.025  # м

# Массы.
m1 = 2.0  # кг
m2 = 1.5  # кг

# Расстояние от шарнира до центра масс.
l1 = L1 / 2.0
l2 = L2 / 2.0

# Моменты инерции относительно оси Z.
#
# Для цилиндра:
#
# Izz = m/12 * (3*r^2 + L^2)
#
I1zz = (1.0 / 12.0) * m1 * (3.0 * radius**2 + L1**2)
I2zz = (1.0 / 12.0) * m2 * (3.0 * radius**2 + L2**2)

# Вязкое трение в шарнирах.
b1 = 0.1
b2 = 0.1


# ============================================================
# ПАРАМЕТРЫ ТРАЕКТОРИИ
# ============================================================

# Скорость движения TCP.
# 5 см/с = 0.05 м/с.
CARTESIAN_SPEED = 0.05

# Шаг вычисления траектории.
#
# В исходном задании Δt = 1 сек.
# Здесь 1 секунда остаётся характерным шагом задания,
# однако для физического управления Gazebo нам нужна
# значительно более частая дискретизация.
#
# Поэтому рассчитываем траекторию с частотой 1000 Гц.
DT = 0.001


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class TrajectoryPoint:
    t: float

    x: float
    y: float

    q1: float
    q2: float

    q1_dot: float
    q2_dot: float

    q1_ddot: float
    q2_ddot: float

    tau1: float
    tau2: float


@dataclass
class Segment:
    start: Tuple[float, float]
    end: Tuple[float, float]


# ============================================================
# ОБРАТНАЯ КИНЕМАТИКА
# ============================================================

def ik(
    x: float,
    y: float,
    q_prev: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Обратная кинематика плоского двухзвенного SCARA.

    Используется классическое аналитическое решение:

        cos(q2) =
            (x² + y² - L1² - L2²) / (2 L1 L2)

    Затем:

        q1 =
            atan2(y, x)
            -
            atan2(L2 sin(q2),
                  L1 + L2 cos(q2))

    Из двух возможных конфигураций выбирается та,
    которая ближе к предыдущему положению.
    """

    D = (
        x**2 + y**2
        - L1**2
        - L2**2
    ) / (2.0 * L1 * L2)

    # Из-за ошибок округления D иногда может оказаться
    # чуть меньше -1 или чуть больше 1.
    D = np.clip(D, -1.0, 1.0)

    # Две кинематические конфигурации:
    # "локоть вверх" и "локоть вниз".
    q2_candidates = [
        np.arctan2(np.sqrt(1.0 - D**2), D),
        np.arctan2(-np.sqrt(1.0 - D**2), D)
    ]

    # Выбираем ветвь, которая обеспечивает
    # наиболее плавное изменение q2.
    q2 = min(
        q2_candidates,
        key=lambda value: abs(value - q_prev[1])
    )

    q1 = (
        np.arctan2(y, x)
        -
        np.arctan2(
            L2 * np.sin(q2),
            L1 + L2 * np.cos(q2)
        )
    )

    return q1, q2


# ============================================================
# ЯКОБИАН
# ============================================================

def jacobian(q1: float, q2: float) -> np.ndarray:
    """
    Якобиан SCARA:

        [x_dot]   [J11 J12] [q1_dot]
        [y_dot] = [J21 J22] [q2_dot]
    """

    s1 = np.sin(q1)
    c1 = np.cos(q1)

    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)

    return np.array([
        [
            -L1 * s1 - L2 * s12,
            -L2 * s12
        ],
        [
            L1 * c1 + L2 * c12,
            L2 * c12
        ]
    ])


# ============================================================
# МАТРИЦА ИНЕРЦИИ M(q)
# ============================================================

def mass_matrix(q2: float) -> np.ndarray:
    """
    Матрица инерции манипулятора.

        M(q) =
            [M11 M12]
            [M12 M22]
    """

    cos_q2 = np.cos(q2)

    M11 = (
        I1zz
        + I2zz
        + m1 * l1**2
        + m2 * (
            L1**2
            + l2**2
            + 2.0 * L1 * l2 * cos_q2
        )
    )

    M12 = (
        I2zz
        + m2 * (
            l2**2
            + L1 * l2 * cos_q2
        )
    )

    M22 = (
        I2zz
        + m2 * l2**2
    )

    return np.array([
        [M11, M12],
        [M12, M22]
    ])


# ============================================================
# КОРИОЛИСОВЫЕ / ЦЕНТРОБЕЖНЫЕ ЧЛЕНЫ
# ============================================================

def coriolis_vector(
    q2: float,
    q1_dot: float,
    q2_dot: float
) -> np.ndarray:
    """
    Возвращает вектор C(q, q_dot) q_dot.

    Для нашего манипулятора:

        h = m2 L1 l2 sin(q2)

        c1 =
            -h * (2 q1_dot q2_dot + q2_dot²)

        c2 =
             h * q1_dot²
    """

    h = m2 * L1 * l2 * np.sin(q2)

    c1 = -h * (
        2.0 * q1_dot * q2_dot
        + q2_dot**2
    )

    c2 = h * q1_dot**2

    return np.array([c1, c2])


# ============================================================
# ОБРАТНАЯ ДИНАМИКА
# ============================================================

def compute_dynamics(
    q1: float,
    q2: float,
    q1_dot: float,
    q2_dot: float,
    q1_ddot: float,
    q2_ddot: float
) -> Tuple[float, float]:
    """
    Вычисляет необходимые моменты:

        τ = M(q) q̈
            + C(q,q̇)q̇
            + B q̇

    Так как наша SCARA-модель работает в горизонтальной
    плоскости, гравитационный момент относительно
    вертикальных осей Z отсутствует.
    """

    M = mass_matrix(q2)

    C = coriolis_vector(
        q2,
        q1_dot,
        q2_dot
    )

    B = np.array([
        b1 * q1_dot,
        b2 * q2_dot
    ])

    q_ddot = np.array([
        q1_ddot,
        q2_ddot
    ])

    tau = M @ q_ddot + C + B

    return tau[0], tau[1]


# ============================================================
# ГЕНЕРАЦИЯ ТРАЕКТОРИИ
# ============================================================

def generate_trajectory(
    segments: List[Segment],
    dt: float = DT
) -> List[TrajectoryPoint]:
    """
    Генерирует траекторию движения TCP.

    На каждом прямолинейном участке:

        x(t) = x0 + vx*t
        y(t) = y0 + vy*t

    Скорость TCP постоянна и равна 0.05 м/с.

    После получения x,y выполняется обратная кинематика,
    затем вычисляются q_dot и q_ddot через Якобиан.
    """

    points = []

    # Начальная точка траектории.
    x0, y0 = segments[0].start

    # Начальная конфигурация.
    q1, q2 = ik(
        x0,
        y0,
        (0.0, 0.0)
    )

    current_time = 0.0

    for segment_index, segment in enumerate(segments):

        x_start, y_start = segment.start
        x_end, y_end = segment.end

        dx = x_end - x_start
        dy = y_end - y_start

        # Длина сегмента.
        length = np.hypot(dx, dy)

        # Время движения по сегменту:
        #
        # T = S / V
        #
        duration = length / CARTESIAN_SPEED

        # Единичный вектор направления.
        ux = dx / length
        uy = dy / length

        # Постоянная декартова скорость.
        x_dot = CARTESIAN_SPEED * ux
        y_dot = CARTESIAN_SPEED * uy

        # Количество шагов.
        n_steps = int(round(duration / dt))

        for i in range(n_steps):

            # Не дублируем первую точку последующих сегментов.
            if segment_index > 0 and i == 0:
                continue

            local_t = i * dt

            # Не допускаем выхода за пределы сегмента.
            local_t = min(local_t, duration)

            # Положение TCP.
            x = x_start + x_dot * local_t
            y = y_start + y_dot * local_t

            # Обратная кинематика.
            q1, q2 = ik(
                x,
                y,
                (q1, q2)
            )

            # Якобиан.
            J = jacobian(q1, q2)

            xy_dot = np.array([
                x_dot,
                y_dot
            ])

            # q_dot = J^-1 * x_dot
            q_dot = np.linalg.solve(
                J,
                xy_dot
            )

            q1_dot = q_dot[0]
            q2_dot = q_dot[1]

            # Производная Якобиана.
            s1 = np.sin(q1)
            c1 = np.cos(q1)

            s12 = np.sin(q1 + q2)
            c12 = np.cos(q1 + q2)

            dJ = np.array([
                [
                    -L1 * c1 * q1_dot
                    - L2 * c12 * (q1_dot + q2_dot),

                    -L2 * c12 * (q1_dot + q2_dot)
                ],

                [
                    -L1 * s1 * q1_dot
                    - L2 * s12 * (q1_dot + q2_dot),

                    -L2 * s12 * (q1_dot + q2_dot)
                ]
            ])

            # Так как x_ddot = y_ddot = 0:
            #
            # J q_ddot + J_dot q_dot = 0
            #
            # q_ddot = -J^-1 J_dot q_dot
            q_ddot = -np.linalg.solve(
                J,
                dJ @ q_dot
            )

            q1_ddot = q_ddot[0]
            q2_ddot = q_ddot[1]

            # Обратная динамика.
            tau1, tau2 = compute_dynamics(
                q1,
                q2,
                q1_dot,
                q2_dot,
                q1_ddot,
                q2_ddot
            )

            points.append(
                TrajectoryPoint(
                    t=current_time + local_t,

                    x=x,
                    y=y,

                    q1=q1,
                    q2=q2,

                    q1_dot=q1_dot,
                    q2_dot=q2_dot,

                    q1_ddot=q1_ddot,
                    q2_ddot=q2_ddot,

                    tau1=tau1,
                    tau2=tau2
                )
            )

        current_time += duration

    # ========================================================
    # КОНЕЧНАЯ ТОЧКА
    # ========================================================

    # Добавляем последнюю точку траектории явно.
    last_x, last_y = segments[-1].end

    q1_final, q2_final = ik(
        last_x,
        last_y,
        (q1, q2)
    )

    points.append(
        TrajectoryPoint(
            t=current_time,

            x=last_x,
            y=last_y,

            q1=q1_final,
            q2=q2_final,

            q1_dot=0.0,
            q2_dot=0.0,

            q1_ddot=0.0,
            q2_ddot=0.0,

            tau1=0.0,
            tau2=0.0
        )
    )

    return points


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

def save_data(points: List[TrajectoryPoint]):

    package_dir = get_package_share_directory(
        'scara_sim'
    )

    data_dir = os.path.join(
        package_dir,
        'data'
    )

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Полная траектория
    # --------------------------------------------------------

    trajectory_path = os.path.join(
        data_dir,
        'trajectory.csv'
    )

    with open(
        trajectory_path,
        'w',
        newline=''
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            'time',
            'x',
            'y',
            'q1',
            'q2',
            'q1_dot',
            'q2_dot',
            'q1_ddot',
            'q2_ddot',
            'tau1_ff',
            'tau2_ff'
        ])

        for p in points:

            writer.writerow([
                p.t,
                p.x,
                p.y,
                p.q1,
                p.q2,
                p.q1_dot,
                p.q2_dot,
                p.q1_ddot,
                p.q2_ddot,
                p.tau1,
                p.tau2
            ])

    # --------------------------------------------------------
    # Отдельный файл моментов
    # --------------------------------------------------------

    moments_path = os.path.join(
        data_dir,
        'moments.csv'
    )

    with open(
        moments_path,
        'w',
        newline=''
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            'time',
            'tau1',
            'tau2'
        ])

        for p in points:

            writer.writerow([
                p.t,
                p.tau1,
                p.tau2
            ])

    print(
        f'Trajectory saved to: {trajectory_path}'
    )

    print(
        f'Moments saved to: {moments_path}'
    )


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

def main():

    # ========================================================
    # БУКВА П
    # ========================================================
    #
    # Каждая сторона = 0.5 м.
    #
    # Начальная точка:
    #
    # (-0.25, -0.25)
    #
    # Робот движется:
    #
    # вверх → вправо → вниз
    #
    # Скорость = 0.05 м/с.
    #
    # Поэтому каждый участок длиной 0.5 м
    # занимает:
    #
    # 0.5 / 0.05 = 10 секунд.
    #
    # Вся траектория:
    #
    # 30 секунд.
    # ========================================================

    segments = [

        Segment(
            start=(-0.25, -0.25),
            end=(-0.25, 0.25)
        ),

        Segment(
            start=(-0.25, 0.25),
            end=(0.25, 0.25)
        ),

        Segment(
            start=(0.25, 0.25),
            end=(0.25, -0.25)
        )
    ]

    points = generate_trajectory(
        segments,
        DT
    )

    save_data(points)

    # ========================================================
    # ПРОВЕРКА НАЧАЛЬНОЙ КОНФИГУРАЦИИ
    # ========================================================

    first = points[0]

    print()
    print('Initial configuration:')
    print(f'q1(0) = {first.q1:.10f} rad')
    print(f'q2(0) = {first.q2:.10f} rad')

    print()
    print('Robot parameters:')
    print(f'L1 = {L1} m')
    print(f'L2 = {L2} m')
    print(f'm1 = {m1} kg')
    print(f'm2 = {m2} kg')
    print(f'I1zz = {I1zz:.8f} kg*m^2')
    print(f'I2zz = {I2zz:.8f} kg*m^2')

    # ========================================================
    # ГРАФИКИ
    # ========================================================

    time = np.array([p.t for p in points])

    q1 = np.array([p.q1 for p in points])
    q2 = np.array([p.q2 for p in points])

    tau1 = np.array([p.tau1 for p in points])
    tau2 = np.array([p.tau2 for p in points])

    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])

    # --------------------------------------------------------
    # Декартова траектория
    # --------------------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.plot(x, y)

    plt.xlabel('X, м')
    plt.ylabel('Y, м')

    plt.title('П-образная траектория TCP')

    plt.axis('equal')
    plt.grid(True)

    # --------------------------------------------------------
    # q1
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(time, q1)

    plt.xlabel('Время, с')
    plt.ylabel('q1, рад')

    plt.title('Обобщённая координата q1')

    plt.grid(True)

    # --------------------------------------------------------
    # q2
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(time, q2)

    plt.xlabel('Время, с')
    plt.ylabel('q2, рад')

    plt.title('Обобщённая координата q2')

    plt.grid(True)

    # --------------------------------------------------------
    # Момент первого двигателя
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(time, tau1)

    plt.xlabel('Время, с')
    plt.ylabel('τ1, Н·м')

    plt.title('Расчётный момент τ1')

    plt.grid(True)

    # --------------------------------------------------------
    # Момент второго двигателя
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(time, tau2)

    plt.xlabel('Время, с')
    plt.ylabel('τ2, Н·м')

    plt.title('Расчётный момент τ2')

    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()