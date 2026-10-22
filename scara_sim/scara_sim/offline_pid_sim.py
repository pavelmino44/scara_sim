#!/usr/bin/env python3
"""
Офлайн-симулятор SCARA для подбора коэффициентов ПИД без Gazebo.

Модель объекта: плоская цепь из n звеньев-цилиндров с вертикальными
осями вращения (как в URDF), вязкое трение в сочленениях.
Динамика строится в общем виде:

    M(q) = sum_i ( m_i Jv_i^T Jv_i + I_i Jw_i^T Jw_i )
    c(q, q_dot) - через символы Кристоффеля (численные производные M)
    q_ddot = M^-1 (tau - c - B q_dot)

Регулятор - тот же класс JointPID, что и в ROS-ноде, с тем же
периодом 1 мс и настраиваемой задержкой измерения (имитирует
задержки ROS / ros2_control).

Пример:
    ros2 run scara_sim offline_pid_sim --robot 2dof --actuator motor \\
        --traj install/scara_sim/share/scara_sim/data/trajectory.csv \\
        --mode ff+pid --kp 300 100 --ki 300 100 --kd 8 3
"""

import argparse

import numpy as np

from scara_sim.control_common import (
    AccelerationEstimator, DCMotorBank, DCMotorParams, JointPID,
    MotorFeedforward, Trajectory, fk_planar, wrap_to_pi,
)


# ============================================================
# ПАРАМЕТРЫ РОБОТОВ (совпадают с URDF)
# ============================================================

def cylinder_izz(m, length, r=0.025):
    return m / 12.0 * (3.0 * r**2 + length**2)


ROBOTS = {
    '2dof': dict(
        L=[0.40, 0.30],
        m=[3.54, 2.66],
        B=[0.1, 0.1],
    ),
    '3dof': dict(
        L=[0.40, 0.30, 0.15],
        m=[3.54, 2.66, 1.33],
        B=[0.1, 0.1, 0.1],
    ),
}


class PlanarChain:

    def __init__(self, L, m, B, r=0.025):
        self.L = np.asarray(L, dtype=float)
        self.m = np.asarray(m, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.lc = self.L / 2.0
        self.I = np.array([cylinder_izz(mi, Li, r) for mi, Li in zip(m, L)])
        self.n = len(L)

    def mass_matrix(self, q):
        n = self.n
        th = np.cumsum(q)
        perp = np.column_stack([-np.sin(th), np.cos(th)])  # e_perp(theta_j)
        M = np.zeros((n, n))

        for i in range(n):
            Jv = np.zeros((2, n))
            for k in range(i + 1):
                v = self.lc[i] * perp[i]
                for j in range(k, i):
                    v = v + self.L[j] * perp[j]
                Jv[:, k] = v
            Jw = np.zeros(n)
            Jw[:i + 1] = 1.0
            M += self.m[i] * Jv.T @ Jv + self.I[i] * np.outer(Jw, Jw)

        return M

    def coriolis(self, q, qd, h=1e-6):
        n = self.n
        dM = np.zeros((n, n, n))  # dM[k] = dM/dq_k
        for k in range(n):
            dq = np.zeros(n)
            dq[k] = h
            dM[k] = (self.mass_matrix(q + dq) - self.mass_matrix(q - dq)) / (2 * h)

        c = np.zeros(n)
        for k in range(n):
            s = 0.0
            for i in range(n):
                for j in range(n):
                    gamma = 0.5 * (dM[i][k, j] + dM[j][k, i] - dM[k][i, j])
                    s += gamma * qd[i] * qd[j]
            c[k] = s
        return c

    def forward_dynamics(self, q, qd, tau):
        M = self.mass_matrix(q)
        c = self.coriolis(q, qd)
        return np.linalg.solve(M, tau - c - self.B * qd)

    def inverse_dynamics(self, q, qd, qdd):
        return (self.mass_matrix(q) @ qdd
                + self.coriolis(q, qd) + self.B * qd)


# ============================================================
# СИМУЛЯЦИЯ
# ============================================================

def simulate(
    robot,
    traj,
    mode='ff+pid',
    kp=None,
    ki=None,
    kd=None,
    integral_limit=None,
    output_limit=None,
    motor=None,
    accel_cutoff_hz=100.0,
    control_dt=0.001,
    substeps=10,
    delay_steps=1,
    t_hold=1.0,
    q0=None,
):
    """
    Вернуть словарь массивов с результатами моделирования.

    motor=None           - регулятор выдаёт момент прямо на шарнир;
    motor=DCMotorParams  - регулятор выдаёт напряжение, момент
                           формирует модель ДПТ (как нода dc_motor_sim).
    """

    n = robot.n
    use_ff = mode in ('ff', 'ff+pid')
    use_pid = mode in ('pid', 'ff+pid')

    pid = JointPID(
        kp if use_pid else np.zeros(n),
        ki if use_pid else np.zeros(n),
        kd if use_pid else np.zeros(n),
        integral_limit,
        output_limit,
    )

    if motor is not None:
        motors = DCMotorBank(motor)
        ff_u = MotorFeedforward(traj, motor)
        acc = AccelerationEstimator(n, accel_cutoff_hz)

    q = np.array(traj.q[0] if q0 is None else q0, dtype=float)
    qd = np.zeros(n)

    hist_q = [q.copy()] * (delay_steps + 1)
    hist_qd = [qd.copy()] * (delay_steps + 1)

    t_end = traj.duration + t_hold
    steps = int(round(t_end / control_dt))
    h = control_dt / substeps

    keys = ('t', 'q_des', 'q', 'u', 'u_ff', 'tau', 'i', 'xy_des', 'xy')
    log = {k: [] for k in keys}

    for s in range(steps + 1):
        t = s * control_dt

        if t <= traj.duration:
            q_des, qd_des, tau_ff = traj.sample(t)
            u_ff = ff_u.sample(t) if motor is not None else tau_ff
        else:
            q_des, qd_des, u_ff = traj.q[-1], np.zeros(n), np.zeros(n)

        if not use_ff:
            u_ff = np.zeros(n)

        q_meas = hist_q[0]
        qd_meas = hist_qd[0]

        e = wrap_to_pi(q_des - q_meas)
        e_dot = qd_des - qd_meas
        u, _ = pid.update(e, e_dot, control_dt, u_ff)

        if motor is not None:
            a_est = acc.update(qd_meas, control_dt)
            m = motors.step(u, qd_meas, a_est, control_dt)
            tau, cur = m['tau'], m['i_avg']
        else:
            tau, cur = u, np.zeros(n)

        log['t'].append(t)
        log['q_des'].append(q_des)
        log['q'].append(q.copy())
        log['u'].append(u)
        log['u_ff'].append(u_ff)
        log['tau'].append(tau)
        log['i'].append(cur)
        log['xy_des'].append(fk_planar(q_des, robot.L))
        log['xy'].append(fk_planar(q, robot.L))

        for _ in range(substeps):
            qdd = robot.forward_dynamics(q, qd, tau)
            qd = qd + qdd * h
            q = q + qd * h

        hist_q = hist_q[1:] + [q.copy()]
        hist_qd = hist_qd[1:] + [qd.copy()]

    return {k: np.array(v) for k, v in log.items()}


def metrics(res, t_max=None):
    t = res['t']
    sel = t <= (t_max if t_max is not None else t[-1])
    e_q = wrap_to_pi(res['q_des'][sel] - res['q'][sel])
    e_xy = np.linalg.norm(res['xy_des'][sel] - res['xy'][sel], axis=1)
    return dict(
        rms_q=np.sqrt(np.mean(e_q**2, axis=0)),
        max_q=np.max(np.abs(e_q), axis=0),
        rms_xy=float(np.sqrt(np.mean(e_xy**2))),
        max_xy=float(np.max(e_xy)),
        max_tau=np.max(np.abs(res['tau'][sel]), axis=0),
        max_u=np.max(np.abs(res['u'][sel]), axis=0),
        max_i=np.max(np.abs(res['i'][sel]), axis=0),
    )


# Параметры ДПТ по умолчанию (совпадают с config/dc_motors.yaml)
DEFAULT_MOTOR = dict(
    R=[2.0, 2.0], L=[1.0e-3, 1.0e-3], kt=[0.05, 0.05], ke=[0.05, 0.05],
    J_rotor=[5.0e-6, 5.0e-6], b_motor=[1.0e-6, 1.0e-6],
    gear_ratio=[50.0, 30.0], gear_eff=[0.9, 0.9], U_max=[24.0, 24.0],
)


def load_motor_yaml(path, n):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    params = next(iter(cfg.values()))['ros__parameters']
    m = params['motor'] if 'motor' in params else {
        k.split('.', 1)[1]: v for k, v in params.items() if k.startswith('motor.')}
    return DCMotorParams(**{k: m[k][:n] for k in DEFAULT_MOTOR})


def main():
    p = argparse.ArgumentParser(description='Offline SCARA PID simulation')
    p.add_argument('--robot', choices=list(ROBOTS), default='2dof')
    p.add_argument('--traj', required=True, help='trajectory CSV')
    p.add_argument('--mode', choices=['ff', 'pid', 'ff+pid'], default='ff+pid')
    p.add_argument('--kp', type=float, nargs='+')
    p.add_argument('--ki', type=float, nargs='+')
    p.add_argument('--kd', type=float, nargs='+')
    p.add_argument('--delay', type=int, default=1, help='задержка измерения, тактов по 1 мс')
    p.add_argument('--actuator', choices=['motor', 'torque'], default='motor',
                   help='motor - ПИД выдаёт напряжение на ДПТ, torque - момент')
    p.add_argument('--motors', default='', help='YAML с параметрами ДПТ (dc_motors.yaml)')
    p.add_argument('--limit', type=float, nargs='+', help='ограничение выхода (В или Н·м)')
    p.add_argument('--plot', action='store_true')
    a = p.parse_args()

    robot = PlanarChain(**ROBOTS[a.robot])
    traj = Trajectory(a.traj)
    n = robot.n

    kp = a.kp or [0.0] * n
    ki = a.ki or [0.0] * n
    kd = a.kd or [0.0] * n

    motor = None
    if a.actuator == 'motor':
        if a.motors:
            motor = load_motor_yaml(a.motors, n)
        else:
            motor = DCMotorParams(**{k: v[:n] if len(v) >= n else v + v[-1:] * (n - len(v))
                                     for k, v in DEFAULT_MOTOR.items()})
        print('DC motors:', motor.describe())

    limit = a.limit or (list(motor.U_max) if motor is not None else None)

    res = simulate(robot, traj, a.mode, kp, ki, kd, output_limit=limit,
                   motor=motor, delay_steps=a.delay)
    m = metrics(res, traj.duration)

    np.set_printoptions(precision=5, suppress=True)
    print(f'mode={a.mode}  Kp={kp}  Ki={ki}  Kd={kd}')
    print(f'RMS joint error, rad : {m["rms_q"]}')
    print(f'Max joint error, rad : {m["max_q"]}')
    print(f'RMS XY error, mm     : {1000 * m["rms_xy"]:.3f}')
    print(f'Max XY error, mm     : {1000 * m["max_xy"]:.3f}')
    print(f'Max |tau|, N*m       : {m["max_tau"]}')
    if motor is not None:
        print(f'Max |U|, V           : {m["max_u"]}')
        print(f'Max |i|, A           : {m["max_i"]}')

    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].plot(res['xy_des'][:, 0], res['xy_des'][:, 1], 'k--', label='desired')
        ax[0].plot(res['xy'][:, 0], res['xy'][:, 1], label=a.mode)
        ax[0].axis('equal')
        ax[0].grid(True)
        ax[0].legend()
        ax[1].plot(res['t'], 1000 * np.linalg.norm(res['xy_des'] - res['xy'], axis=1))
        ax[1].set_xlabel('t, s')
        ax[1].set_ylabel('|e_xy|, mm')
        ax[1].grid(True)
        plt.show()


if __name__ == '__main__':
    main()
