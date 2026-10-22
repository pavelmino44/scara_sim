#!/usr/bin/env python3
"""
Управляемое движение SCARA: ПИД-регулятор положения сочленений.

Два варианта исполнительного устройства (параметр actuator):

  motor  (по умолчанию) - регулятор выдаёт НАПРЯЖЕНИЕ на двигатели
         постоянного тока (нода dc_motor_sim), момент на шарнире
         формирует модель ДПТ с редуктором:

             U = U_ff(t) + Kp*e + Ki*int(e)dt + Kd*e_dot      [В]

             U_ff = R*i_ff + L*di_ff/dt + ke*N*q_dot_des
             i_ff = (tau_ff + N^2 (J_r q_ddot_des + b_m q_dot_des)) / (eta N kt)

  torque - регулятор выдаёт МОМЕНТ прямо в Gazebo (идеальный привод):

             tau = tau_ff(t) + Kp*e + Ki*int(e)dt + Kd*e_dot  [Н·м]

  e = wrap(q_des - q),  e_dot = q_dot_des - q_dot
  tau_ff, q_des, q_dot_des, q_ddot_des - из trajectory.csv.

Режимы (параметр mode):
    ff      - только прямая связь (разомкнутое управление)
    pid     - только ПИД-регулятор
    ff+pid  - прямая связь + ПИД (по умолчанию)

Работа ноды:
    1. Ждёт первое сообщение /joint_states.
    2. start_delay секунд удерживает начальную точку траектории.
    3. Отрабатывает траекторию.
    4. Удерживает конечную точку; через hold_after секунд сохраняет
       лог и печатает метрики качества.

Нода работает с любым числом сочленений: число берётся
из trajectory.csv, имена - из параметра joint_names.
"""

import csv
import os
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory

from scara_sim.control_common import (
    MOTOR_PARAM_NAMES, DCMotorParams, JointPID, MotorFeedforward,
    Trajectory, fk_planar, wrap_to_pi,
)


class PidController(Node):

    def __init__(self):

        super().__init__('pid_controller')

        # ====================================================
        # ПАРАМЕТРЫ
        # ====================================================

        self.declare_parameter('joint_names', ['joint1', 'joint2'])
        self.declare_parameter('link_lengths', [0.40, 0.30])
        self.declare_parameter('actuator', 'motor')
        self.declare_parameter('mode', 'ff+pid')

        self.declare_parameter('kp', [300.0, 100.0])
        self.declare_parameter('ki', [300.0, 100.0])
        self.declare_parameter('kd', [8.0, 3.0])
        self.declare_parameter('integral_limit', [0.05, 0.05])
        # Ограничение выхода: В (motor) или Н·м (torque).
        # Пусто -> для motor берётся U_max двигателей.
        self.declare_parameter('output_limit', [0.0])

        for name in MOTOR_PARAM_NAMES:
            self.declare_parameter('motor.' + name, [0.0])

        self.declare_parameter('control_rate', 1000.0)
        self.declare_parameter('start_delay', 1.0)
        self.declare_parameter('hold_after', 1.0)

        self.declare_parameter('trajectory_file', '')
        self.declare_parameter('log_file', '')
        # Пусто -> /dc_motors/voltage (motor) или /effort_controller/commands (torque)
        self.declare_parameter('command_topic', '')
        self.declare_parameter('motor_state_topic', '/dc_motors/state')

        gp = self.get_parameter

        self.joint_names = list(gp('joint_names').value)
        self.link_lengths = np.array(gp('link_lengths').value, dtype=float)
        self.mode = gp('mode').value
        self.actuator = gp('actuator').value

        if self.mode not in ('ff', 'pid', 'ff+pid'):
            raise ValueError(f'Unknown mode "{self.mode}". Use ff | pid | ff+pid')

        if self.actuator not in ('motor', 'torque'):
            raise ValueError(f'Unknown actuator "{self.actuator}". Use motor | torque')

        self.use_ff = self.mode in ('ff', 'ff+pid')
        self.use_pid = self.mode in ('pid', 'ff+pid')
        self.use_motor = self.actuator == 'motor'
        self.unit = 'V' if self.use_motor else 'N*m'

        self.start_delay = float(gp('start_delay').value)
        self.hold_after = float(gp('hold_after').value)

        # ====================================================
        # ТРАЕКТОРИЯ
        # ====================================================

        data_dir = os.path.join(get_package_share_directory('scara_sim'), 'data')

        traj_path = gp('trajectory_file').value or os.path.join(data_dir, 'trajectory.csv')

        self.traj = Trajectory(traj_path)
        self.n = self.traj.n

        if len(self.joint_names) != self.n:
            raise ValueError(
                f'joint_names has {len(self.joint_names)} entries, '
                f'trajectory has {self.n} joints')

        log_path = gp('log_file').value
        if not log_path:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_tag = self.mode.replace('+', '_')
            log_path = os.path.join(
                data_dir, f'pid_log_{self.actuator}_{mode_tag}_{stamp}.csv')
        self.log_path = log_path

        # ====================================================
        # ПРИВОДЫ
        # ====================================================

        if self.use_motor:
            self.motor = DCMotorParams.from_getter(lambda k: gp(k).value)
            if self.motor.n != self.n:
                raise ValueError(
                    f'Motor parameters have {self.motor.n} values, '
                    f'trajectory has {self.n} joints. Check dc_motors.yaml')
            self.ff_voltage = MotorFeedforward(self.traj, self.motor)

        # ====================================================
        # РЕГУЛЯТОР
        # ====================================================

        def vec(name):
            v = np.array(gp(name).value, dtype=float)
            if len(v) != self.n:
                raise ValueError(f'Parameter {name} must have {self.n} values')
            return v

        kp, ki, kd = vec('kp'), vec('ki'), vec('kd')

        if not self.use_pid:
            kp, ki, kd = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)

        limit = np.array(gp('output_limit').value, dtype=float)
        if len(limit) != self.n or not np.all(limit > 0):
            limit = self.motor.U_max if self.use_motor else np.full(self.n, np.inf)

        self.pid = JointPID(kp, ki, kd, vec('integral_limit'), limit)

        # ====================================================
        # ROS
        # ====================================================

        self.q = None
        self.q_dot = None
        self.motor_state = np.full(3 * self.n, np.nan)

        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        if self.use_motor:
            self.create_subscription(
                Float64MultiArray, gp('motor_state_topic').value, self.motor_state_cb, 10)

        topic = gp('command_topic').value or (
            '/dc_motors/voltage' if self.use_motor else '/effort_controller/commands')

        self.publisher = self.create_publisher(Float64MultiArray, topic, 10)

        self.t0 = None          # момент начала траектории (sim time), с
        self.t_prev = None
        self.log_done = False
        self.log_rows = []

        self.timer = self.create_timer(
            1.0 / float(gp('control_rate').value), self.control_cb)

        info = (
            f'PID controller: actuator={self.actuator}, mode={self.mode}, '
            f'joints={self.joint_names}\n'
            f'  trajectory: {traj_path} ({self.traj.duration:.2f} s)\n'
            f'  Kp={kp.tolist()} Ki={ki.tolist()} Kd={kd.tolist()} '
            f'(output in {self.unit}, limit {limit.tolist()})\n'
            f'  command topic: {topic}\n'
            f'  log: {self.log_path}')

        if self.use_motor:
            info += f'\n  motors: {self.motor.describe()}'

        self.get_logger().info(info)

    # ========================================================
    # ИЗМЕРЕНИЯ
    # ========================================================

    def joint_state_cb(self, msg: JointState):

        index = {name: i for i, name in enumerate(msg.name)}

        try:
            idx = [index[name] for name in self.joint_names]
        except KeyError:
            return

        self.q = np.array([msg.position[i] for i in idx])

        if len(msg.velocity) == len(msg.name):
            self.q_dot = np.array([msg.velocity[i] for i in idx])
        else:
            self.q_dot = np.zeros(self.n)

    def motor_state_cb(self, msg: Float64MultiArray):
        if len(msg.data) == 3 * self.n:
            self.motor_state = np.array(msg.data, dtype=float)

    # ========================================================
    # ЦИКЛ УПРАВЛЕНИЯ
    # ========================================================

    def control_cb(self):

        if self.q is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.t0 is None:
            self.t0 = now + self.start_delay
            self.t_prev = now
            self.get_logger().info(
                f'Joint states received. Trajectory starts in {self.start_delay:.1f} s.')
            return

        dt = now - self.t_prev

        # Симуляционное время ещё не продвинулось - не дублируем такт
        if dt <= 0.0:
            return

        dt = min(dt, 0.02)
        self.t_prev = now

        t = now - self.t0
        T = self.traj.duration
        zeros = np.zeros(self.n)

        # ----------------------------------------------------
        # Уставка и прямая связь
        # ----------------------------------------------------

        if t < 0.0:
            # Удержание начальной точки
            q_des, q_dot_des, u_ff = self.traj.q[0], zeros, zeros

        elif t <= T:
            q_des, q_dot_des, tau_ff = self.traj.sample(t)
            u_ff = self.ff_voltage.sample(t) if self.use_motor else tau_ff

        else:
            # Удержание конечной точки
            q_des, q_dot_des, u_ff = self.traj.q[-1], zeros, zeros

        if not self.use_ff:
            u_ff = zeros

        # ----------------------------------------------------
        # Регулятор
        # ----------------------------------------------------

        e = wrap_to_pi(q_des - self.q)
        e_dot = q_dot_des - self.q_dot

        u, u_pid = self.pid.update(e, e_dot, dt, u_ff)

        self.publish(u)

        # ----------------------------------------------------
        # Лог
        # ----------------------------------------------------

        if 0.0 <= t <= T + self.hold_after:
            q_meas = q_des - e      # измеренный угол в той же "ветке", что и q_des
            x_des, y_des = fk_planar(q_des, self.link_lengths)
            x, y = fk_planar(q_meas, self.link_lengths)

            if self.use_motor:
                current = self.motor_state[self.n:2 * self.n]
                tau = self.motor_state[2 * self.n:]
            else:
                current, tau = np.full(self.n, np.nan), u

            self.log_rows.append(
                [t, x_des, y_des, x, y]
                + list(q_des) + list(q_meas) + list(e)
                + list(q_dot_des) + list(self.q_dot)
                + list(u_ff) + list(u_pid) + list(u)
                + list(current) + list(tau))

        if t > T + self.hold_after and not self.log_done:
            self.finish()

    # ========================================================
    # ВЫВОД
    # ========================================================

    def publish(self, u):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in u]
        self.publisher.publish(msg)

    def finish(self):

        # Регулятор продолжает удерживать конечную точку,
        # лог сохраняется один раз.
        self.log_done = True
        self.save_log()
        self.print_metrics()

    def save_log(self):

        n = self.n
        header = ['t', 'x_des', 'y_des', 'x', 'y']
        header += [f'q{i + 1}_des' for i in range(n)]
        header += [f'q{i + 1}' for i in range(n)]
        header += [f'e{i + 1}' for i in range(n)]
        header += [f'q{i + 1}_dot_des' for i in range(n)]
        header += [f'q{i + 1}_dot' for i in range(n)]
        # u - выход регулятора: В (motor) или Н·м (torque)
        header += [f'u{i + 1}_ff' for i in range(n)]
        header += [f'u{i + 1}_pid' for i in range(n)]
        header += [f'u{i + 1}' for i in range(n)]
        header += [f'i{i + 1}' for i in range(n)]
        header += [f'tau{i + 1}' for i in range(n)]

        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

        with open(self.log_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows([[f'{v:.9g}' for v in row] for row in self.log_rows])

        self.get_logger().info(f'Log saved: {self.log_path}')

    def print_metrics(self):

        if not self.log_rows:
            return

        data = np.array(self.log_rows)
        data = data[data[:, 0] <= self.traj.duration]

        n = self.n
        e = data[:, 5 + 2 * n: 5 + 3 * n]
        u = data[:, 5 + 7 * n: 5 + 8 * n]
        cur = data[:, 5 + 8 * n: 5 + 9 * n]
        e_xy = np.hypot(data[:, 1] - data[:, 3], data[:, 2] - data[:, 4])

        rms_q = np.sqrt(np.mean(e**2, axis=0))
        max_q = np.max(np.abs(e), axis=0)

        text = (
            '\n========== КАЧЕСТВО СЛЕЖЕНИЯ ==========\n'
            f'actuator / mode : {self.actuator} / {self.mode}\n'
            f'RMS e_q, rad    : {np.round(rms_q, 6).tolist()}\n'
            f'max |e_q|, rad  : {np.round(max_q, 6).tolist()}\n'
            f'RMS e_xy, mm    : {1000 * np.sqrt(np.mean(e_xy**2)):.3f}\n'
            f'max e_xy, mm    : {1000 * np.max(e_xy):.3f}\n'
            f'max |u|, {self.unit:<6}: {np.round(np.max(np.abs(u), axis=0), 3).tolist()}\n')

        if self.use_motor:
            text += f'max |i|, A      : {np.round(np.nanmax(np.abs(cur), axis=0), 3).tolist()}\n'

        self.get_logger().info(text + '=======================================')


def main(args=None):

    rclpy.init(args=args)

    node = PidController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node.log_rows and not node.log_done:
            node.save_log()
            node.print_metrics()

        node.publish(np.zeros(node.n))
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
