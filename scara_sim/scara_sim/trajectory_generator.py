#!/usr/bin/env python3

import os
import csv
import math
import argparse

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from ament_index_python.packages import get_package_share_directory


# ============================================================
# ПАРАМЕТРЫ НОВОГО SCARA
# ============================================================

L1 = 0.40          # длина первого звена, м
L2 = 0.30          # длина второго звена, м

M1 = 3.54          # масса первого звена, кг
M2 = 2.66          # масса второго звена, кг

l1 = L1 / 2.0
l2 = L2 / 2.0

RADIUS = 0.025


# Моменты инерции цилиндрических звеньев.
#
# Ось цилиндра направлена вдоль x.
#
# Izz = 1/12 * m * (3*r^2 + L^2)

I1zz = (
    1.0 / 12.0
    * M1
    * (3.0 * RADIUS**2 + L1**2)
)

I2zz = (
    1.0 / 12.0
    * M2
    * (3.0 * RADIUS**2 + L2**2)
)


# Вязкое трение

B1 = 0.1
B2 = 0.1


# ============================================================
# ПАРАМЕТРЫ ТРАЕКТОРИИ
# ============================================================

SIDE = 0.50          # сторона буквы П, м

V = 0.05             # постоянная скорость, м/с

DT = 0.001            # шаг расчёта, с

START_X = -0.25
START_Y = -0.25

TOOL_Z = 0.15


# ============================================================
# СТРУКТУРА ТОЧКИ ТРАЕКТОРИИ
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
    """
    Привести угол к диапазону [-pi, pi].
    """

    return (
        (angle + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


def unwrap_angle(angle, previous):
    """
    Выбрать эквивалентный угол,
    ближайший к предыдущему значению.

    Это предотвращает скачки на 2*pi.
    """

    return (
        previous
        + wrap_to_pi(angle - previous)
    )


# ============================================================
# ОБРАТНАЯ КИНЕМАТИКА
# ============================================================

def inverse_kinematics(
    x,
    y,
    previous_q=None
):
    """
    Обратная кинематика плоского 2R
    манипулятора.

    Используется ветвь q2 > 0.
    """

    D = (
        x**2
        + y**2
        - L1**2
        - L2**2
    ) / (2.0 * L1 * L2)

    D = np.clip(
        D,
        -1.0,
        1.0
    )

    q2 = math.acos(D)

    q1 = (
        math.atan2(y, x)
        - math.atan2(
            L2 * math.sin(q2),
            L1 + L2 * math.cos(q2)
        )
    )

    if previous_q is None:
        return q1, q2

    q1 = unwrap_angle(
        q1,
        previous_q[0]
    )

    q2 = unwrap_angle(
        q2,
        previous_q[1]
    )

    return q1, q2


# ============================================================
# ПРЯМАЯ КИНЕМАТИКА
# ============================================================

def forward_kinematics(
    q1,
    q2
):

    x = (
        L1 * math.cos(q1)
        + L2 * math.cos(q1 + q2)
    )

    y = (
        L1 * math.sin(q1)
        + L2 * math.sin(q1 + q2)
    )

    return x, y


# ============================================================
# ЯКОБИАН
# ============================================================

def jacobian(
    q1,
    q2
):

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


# ============================================================
# ПРОИЗВОДНАЯ ЯКОБИАНА
# ============================================================

def jacobian_dot(
    q1,
    q2,
    q1_dot,
    q2_dot
):

    s12 = math.sin(
        q1 + q2
    )

    c12 = math.cos(
        q1 + q2
    )

    return np.array([

        [
            -L1 * math.cos(q1) * q1_dot
            - L2 * c12 * (
                q1_dot + q2_dot
            ),

            -L2 * c12 * (
                q1_dot + q2_dot
            )
        ],

        [
            -L1 * math.sin(q1) * q1_dot
            - L2 * s12 * (
                q1_dot + q2_dot
            ),

            -L2 * s12 * (
                q1_dot + q2_dot
            )
        ]
    ])


# ============================================================
# ЛАГРАНЖ
# ============================================================

def lagrange_torques(
    q1,
    q2,
    q1_dot,
    q2_dot,
    q1_ddot,
    q2_ddot
):
    """
    Расчёт моментов методом Лагранжа:

        tau = M(q) * q_ddot
              + C(q, q_dot) * q_dot
              + B * q_dot

    Для горизонтального SCARA
    гравитационный момент вокруг
    вертикальных осей отсутствует.
    """

    c2 = math.cos(q2)
    s2 = math.sin(q2)

    # --------------------------------------------------------
    # Матрица инерции M(q)
    # --------------------------------------------------------

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

    M = np.array([
        [M11, M12],
        [M12, M22]
    ])

    q_ddot = np.array([
        q1_ddot,
        q2_ddot
    ])

    # --------------------------------------------------------
    # Кориолисовы и центробежные члены
    # --------------------------------------------------------

    h = (
        M2
        * L1
        * l2
        * s2
    )

    # ВАЖНО:
    #
    # C1 = -h * (
    #       2*q1_dot*q2_dot
    #       + q2_dot^2
    # )
    #
    # В предыдущем варианте был потерян
    # член q1_dot*q2_dot.

    C1 = (
        -h
        * (
            2.0 * q1_dot * q2_dot
            + q2_dot**2
        )
    )

    C2 = (
        h
        * q1_dot**2
    )

    C = np.array([
        C1,
        C2
    ])

    # --------------------------------------------------------
    # Вязкое трение
    # --------------------------------------------------------

    damping = np.array([
        B1 * q1_dot,
        B2 * q2_dot
    ])

    # --------------------------------------------------------
    # Итоговый момент
    # --------------------------------------------------------

    tau = (
        M @ q_ddot
        + C
        + damping
    )

    return tau[0], tau[1]


# ============================================================
# НЬЮТОН–ЭЙЛЕР
# ============================================================

def newton_euler_torques(
    q1,
    q2,
    q1_dot,
    q2_dot,
    q1_ddot,
    q2_ddot
):
    """
    Расчёт моментов методом Ньютона–Эйлера.

    Используется плоская форма
    рекурсивного алгоритма Ньютона–Эйлера.

    Для каждого звена рассчитываются:

        F = m * a

        N = I * alpha + r x F

    Затем силы и моменты передаются
    от второго звена к первому.

    Гравитация не учитывается, поскольку
    робот движется в горизонтальной плоскости
    вокруг вертикальной оси z.
    """

    # --------------------------------------------------------
    # Углы звеньев
    # --------------------------------------------------------

    theta1 = q1
    theta2 = q1 + q2

    # Угловые скорости

    omega1 = q1_dot
    omega2 = q1_dot + q2_dot

    # Угловые ускорения

    alpha1 = q1_ddot
    alpha2 = q1_ddot + q2_ddot

    # --------------------------------------------------------
    # Единичные векторы вдоль звеньев
    # --------------------------------------------------------

    e1 = np.array([
        math.cos(theta1),
        math.sin(theta1)
    ])

    e2 = np.array([
        math.cos(theta2),
        math.sin(theta2)
    ])

    # --------------------------------------------------------
    # Векторное произведение в плоскости XY
    #
    # Возвращает z-компонент:
    #
    # r x F
    # --------------------------------------------------------

    def cross_z(r, F):

        return (
            r[0] * F[1]
            - r[1] * F[0]
        )

    # --------------------------------------------------------
    # Ускорение точки, вращающейся вокруг начала
    #
    # a = alpha x r
    #     + omega x (omega x r)
    #
    # В 2D:
    #
    # alpha x r =
    # [-alpha*y, alpha*x]
    #
    # omega x (omega x r) =
    # [-omega^2*x, -omega^2*y]
    # --------------------------------------------------------

    def rotational_acceleration(
        r,
        omega,
        alpha
    ):

        return np.array([

            -alpha * r[1]
            - omega**2 * r[0],

            alpha * r[0]
            - omega**2 * r[1]
        ])

    # ========================================================
    # ПРЯМОЙ ПРОХОД
    # ========================================================

    # --------------------------------------------------------
    # Центр масс первого звена
    # --------------------------------------------------------

    r_c1 = l1 * e1

    a_c1 = rotational_acceleration(
        r_c1,
        omega1,
        alpha1
    )

    # --------------------------------------------------------
    # Точка второго шарнира
    # --------------------------------------------------------

    r_12 = L1 * e1

    a_joint2 = rotational_acceleration(
        r_12,
        omega1,
        alpha1
    )

    # --------------------------------------------------------
    # Центр масс второго звена
    # --------------------------------------------------------

    r_c2 = l2 * e2

    a_c2 = (
        a_joint2
        + rotational_acceleration(
            r_c2,
            omega2,
            alpha2
        )
    )

    # ========================================================
    # ОБРАТНЫЙ ПРОХОД
    # ========================================================

    # --------------------------------------------------------
    # Звено 2
    # --------------------------------------------------------

    F2 = M2 * a_c2

    N2 = (
        I2zz * alpha2
        + cross_z(
            r_c2,
            F2
        )
    )

    # Момент второго шарнира

    tau2 = N2

    # --------------------------------------------------------
    # Звено 1
    # --------------------------------------------------------

    F1 = (
        M1 * a_c1
        + F2
    )

    tau1 = (
        I1zz * alpha1

        + cross_z(
            r_c1,
            M1 * a_c1
        )

        + cross_z(
            r_12,
            F2
        )

        + N2
    )

    # --------------------------------------------------------
    # Вязкое трение
    # --------------------------------------------------------

    tau1 += B1 * q1_dot
    tau2 += B2 * q2_dot

    return tau1, tau2


# ============================================================
# ВЫБОР МЕТОДА
# ============================================================

def calculate_torques(
    method,
    q1,
    q2,
    q1_dot,
    q2_dot,
    q1_ddot,
    q2_ddot
):

    if method == 'lagrange':

        return lagrange_torques(
            q1,
            q2,
            q1_dot,
            q2_dot,
            q1_ddot,
            q2_ddot
        )

    if method == 'newton-euler':

        return newton_euler_torques(
            q1,
            q2,
            q1_dot,
            q2_dot,
            q1_ddot,
            q2_ddot
        )

    raise ValueError(
        f'Unknown method: {method}'
    )


# ============================================================
# ГЕНЕРАЦИЯ ОДНОГО УЧАСТКА
# ============================================================

def generate_segment(
    start,
    end,
    t_start,
    previous_q,
    method
):
    """
    Прямолинейный участок с ПОСТОЯННОЙ
    линейной скоростью V.

    Никакого разгона.

    Никакого торможения.

    Никакого импульсного момента.

    На участке:

        |v| = V

        a = 0
    """

    x0, y0 = start
    x1, y1 = end

    dx = x1 - x0
    dy = y1 - y0

    distance = math.sqrt(
        dx**2 + dy**2
    )

    duration = distance / V

    direction_x = dx / distance
    direction_y = dy / distance

    num_points = int(
        round(duration / DT)
    )

    points = []

    q_previous = previous_q

    for i in range(num_points + 1):

        local_t = min(
            i * DT,
            duration
        )

        # ====================================================
        # ПОСТОЯННАЯ СКОРОСТЬ
        # ====================================================

        s = V * local_t

        if s > distance:
            s = distance

        x = (
            x0
            + direction_x * s
        )

        y = (
            y0
            + direction_y * s
        )

        # Скорость постоянна на всём участке

        x_dot = (
            V * direction_x
        )

        y_dot = (
            V * direction_y
        )

        # Ускорение на прямом участке отсутствует

        x_ddot = 0.0
        y_ddot = 0.0

        # ====================================================
        # IK
        # ====================================================

        q1, q2 = inverse_kinematics(
            x,
            y,
            q_previous
        )

        q_previous = (
            q1,
            q2
        )

        # ====================================================
        # JACOBIAN
        # ====================================================

        J = jacobian(
            q1,
            q2
        )

        det_J = np.linalg.det(J)

        if abs(det_J) < 1e-6:

            raise RuntimeError(
                'Jacobian is close to singularity: '
                f'det(J) = {det_J}'
            )

        # ====================================================
        # JOINT VELOCITIES
        # ====================================================

        cartesian_velocity = np.array([
            x_dot,
            y_dot
        ])

        q_dot = np.linalg.solve(
            J,
            cartesian_velocity
        )

        # ====================================================
        # JOINT ACCELERATIONS
        #
        # x_ddot = J*q_ddot + J_dot*q_dot
        #
        # x_ddot = 0
        #
        # q_ddot =
        # J^-1 * (-J_dot*q_dot)
        # ====================================================

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

        # ====================================================
        # MOMENTS
        # ====================================================

        tau1, tau2 = calculate_torques(
            method,

            q1,
            q2,

            q_dot[0],
            q_dot[1],

            q_ddot[0],
            q_ddot[1]
        )

        # ====================================================
        # СОХРАНЕНИЕ
        # ====================================================

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

    return (
        points,
        q_previous,
        duration
    )


# ============================================================
# ПОЛНАЯ ТРАЕКТОРИЯ
# ============================================================

def generate_trajectory(method):

    corners = [

        (
            START_X,
            START_Y
        ),

        (
            START_X,
            START_Y + SIDE
        ),

        (
            START_X + SIDE,
            START_Y + SIDE
        ),

        (
            START_X + SIDE,
            START_Y
        )
    ]

    trajectory = []

    q_previous = inverse_kinematics(
        START_X,
        START_Y
    )

    current_time = 0.0

    for segment_index in range(3):

        start = corners[
            segment_index
        ]

        end = corners[
            segment_index + 1
        ]

        (
            segment_points,
            q_previous,
            duration
        ) = generate_segment(

            start,
            end,

            current_time,

            q_previous,

            method
        )

        # Убираем повторную точку
        # на границе участков.

        if segment_index > 0:

            segment_points = (
                segment_points[1:]
            )

        trajectory.extend(
            segment_points
        )

        current_time += duration

    return trajectory


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def save_trajectory(
    trajectory,
    method
):

    package_path = (
        get_package_share_directory(
            'scara_sim'
        )
    )

    data_dir = os.path.join(
        package_path,
        'data'
    )

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    # Один общий файл для publisher.
    #
    # Метод сохраняется в отдельном файле
    # с результатами сравнения.

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

    # Сохраняем также копию с названием метода.
    method_csv = os.path.join(
        data_dir,
        f'trajectory_{method}.csv'
    )

    import shutil

    shutil.copyfile(
        csv_path,
        method_csv
    )

    return csv_path


# ============================================================
# ГРАФИКИ
# ============================================================

def plot_trajectory(
    trajectory,
    method
):

    t = np.array([
        p.t
        for p in trajectory
    ])

    x = np.array([
        p.x
        for p in trajectory
    ])

    y = np.array([
        p.y
        for p in trajectory
    ])

    q1 = np.array([
        p.q1
        for p in trajectory
    ])

    q2 = np.array([
        p.q2
        for p in trajectory
    ])

    tau1 = np.array([
        p.tau1
        for p in trajectory
    ])

    tau2 = np.array([
        p.tau2
        for p in trajectory
    ])

    # --------------------------------------------------------
    # XY
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        x,
        y
    )

    plt.plot(
        x[0],
        y[0],
        'o'
    )

    plt.xlabel(
        'X, m'
    )

    plt.ylabel(
        'Y, m'
    )

    plt.title(
        f'Desired trajectory '
        f'({method})'
    )

    plt.axis(
        'equal'
    )

    plt.grid(
        True
    )

    # --------------------------------------------------------
    # JOINT POSITIONS
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

    plt.xlabel(
        'Time, s'
    )

    plt.ylabel(
        'Angle, rad'
    )

    plt.title(
        'Joint positions'
    )

    plt.grid(
        True
    )

    plt.legend()

    # --------------------------------------------------------
    # JOINT VELOCITIES
    # --------------------------------------------------------

    q1_dot = np.array([
        p.q1_dot
        for p in trajectory
    ])

    q2_dot = np.array([
        p.q2_dot
        for p in trajectory
    ])

    plt.figure()

    plt.plot(
        t,
        q1_dot,
        label='q1_dot'
    )

    plt.plot(
        t,
        q2_dot,
        label='q2_dot'
    )

    plt.xlabel(
        'Time, s'
    )

    plt.ylabel(
        'Angular velocity, rad/s'
    )

    plt.title(
        'Joint velocities'
    )

    plt.grid(
        True
    )

    plt.legend()

    # --------------------------------------------------------
    # TORQUES
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

    plt.xlabel(
        'Time, s'
    )

    plt.ylabel(
        'Torque, N*m'
    )

    plt.title(
        f'Inverse dynamics: {method}'
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.show()


# ============================================================
# ПРОВЕРКА ДВУХ МЕТОДОВ
# ============================================================

def compare_methods():

    print()
    print(
        'Comparing Lagrange and Newton-Euler...'
    )

    lagrange_trajectory = (
        generate_trajectory(
            'lagrange'
        )
    )

    newton_euler_trajectory = (
        generate_trajectory(
            'newton-euler'
        )
    )

    tau_l = np.array([
        [
            p.tau1,
            p.tau2
        ]
        for p in lagrange_trajectory
    ])

    tau_ne = np.array([
        [
            p.tau1,
            p.tau2
        ]
        for p in newton_euler_trajectory
    ])

    difference = np.abs(
        tau_l - tau_ne
    )

    print()
    print(
        f'Max |tau_L - tau_NE| = '
        f'{np.max(difference):.12e} N*m'
    )

    print(
        f'Mean |tau_L - tau_NE| = '
        f'{np.mean(difference):.12e} N*m'
    )

    print()

    return (
        lagrange_trajectory,
        newton_euler_trajectory
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            'SCARA trajectory generator'
        )
    )

    parser.add_argument(
        '--method',
        choices=[
            'lagrange',
            'newton-euler',
            'compare'
        ],
        default='lagrange',
        help=(
            'Dynamics calculation method'
        )
    )

    args, _ = parser.parse_known_args()

    # ========================================================
    # СРАВНЕНИЕ
    # ========================================================

    if args.method == 'compare':

        lagrange_trajectory, \
        newton_euler_trajectory = (
            compare_methods()
        )

        # Показываем Лагранжа
        # как основной результат.

        csv_path = save_trajectory(
            lagrange_trajectory,
            'lagrange'
        )

        print(
            f'Trajectory saved to:\n'
            f'{csv_path}'
        )

        plot_trajectory(
            lagrange_trajectory,
            'lagrange'
        )

        return

    # ========================================================
    # ОДИН МЕТОД
    # ========================================================

    print()
    print(
        '========================================'
    )

    print(
        f'Dynamics method: {args.method}'
    )

    print(
        f'L1 = {L1:.3f} m'
    )

    print(
        f'L2 = {L2:.3f} m'
    )

    print(
        f'M1 = {M1:.3f} kg'
    )

    print(
        f'M2 = {M2:.3f} kg'
    )

    print(
        f'V  = {V:.3f} m/s'
    )

    print(
        f'dt = {DT:.4f} s'
    )

    print(
        '========================================'
    )

    print()
    print(
        'Generating trajectory...'
    )

    trajectory = generate_trajectory(
        args.method
    )

    print(
        f'Number of points: '
        f'{len(trajectory)}'
    )

    print(
        f'Total time: '
        f'{trajectory[-1].t:.3f} s'
    )

    print(
        f'Initial q1: '
        f'{trajectory[0].q1:.6f} rad'
    )

    print(
        f'Initial q2: '
        f'{trajectory[0].q2:.6f} rad'
    )

    csv_path = save_trajectory(
        trajectory,
        args.method
    )

    print()
    print(
        f'Trajectory saved to:\n'
        f'{csv_path}'
    )

    plot_trajectory(
        trajectory,
        args.method
    )


if __name__ == '__main__':
    main()