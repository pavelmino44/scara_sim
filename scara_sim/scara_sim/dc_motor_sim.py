#!/usr/bin/env python3
"""
Модель приводов SCARA: двигатели постоянного тока с редукторами.

Gazebo умеет только прикладывать к шарниру заданный момент.
Эта нода стоит между регулятором и Gazebo и играет роль
"железа" - двигателя с усилителем мощности:

    /dc_motors/voltage (U, В)  -->  [ ДПТ + редуктор ]  -->  /effort_controller/commands (tau, Н·м)
                                          ^
                                   /joint_states (q_dot)

Уравнения (для каждого сочленения):

    L di/dt = U - R i - ke * N * q_dot          (электрическая часть)
    tau_m   = kt * i                            (момент на валу)
    tau     = eta * N * tau_m
              - N^2 * (J_r * q_ddot + b_m * q_dot)  (момент на шарнире)

Инерция ротора N^2*J_r учитывается через оценку ускорения
шарнира (фильтрованная производная скорости), т.к. в модели
Gazebo её нет.

Состояние публикуется в /dc_motors/state (Float64MultiArray):
    [U_1..U_n, i_1..i_n, tau_1..tau_n]
"""

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from scara_sim.control_common import (
    MOTOR_PARAM_NAMES, AccelerationEstimator, DCMotorBank, DCMotorParams,
)


class DCMotorSim(Node):

    def __init__(self):

        super().__init__('dc_motor_sim')

        self.declare_parameter('joint_names', ['joint1', 'joint2'])

        for name in MOTOR_PARAM_NAMES:
            self.declare_parameter('motor.' + name, [0.0])

        self.declare_parameter('rate', 1000.0)
        self.declare_parameter('command_timeout', 0.1)
        self.declare_parameter('include_rotor_inertia', True)
        self.declare_parameter('accel_cutoff_hz', 100.0)

        self.declare_parameter('voltage_topic', '/dc_motors/voltage')
        self.declare_parameter('torque_topic', '/effort_controller/commands')
        self.declare_parameter('state_topic', '/dc_motors/state')

        gp = self.get_parameter

        self.joint_names = list(gp('joint_names').value)
        self.n = len(self.joint_names)

        self.params = DCMotorParams.from_getter(lambda k: gp(k).value)

        if self.params.n != self.n:
            raise ValueError(
                f'Motor parameters have {self.params.n} values, '
                f'joint_names has {self.n}. Check dc_motors.yaml')

        self.motors = DCMotorBank(self.params)
        self.acc = AccelerationEstimator(self.n, float(gp('accel_cutoff_hz').value))
        self.include_rotor_inertia = bool(gp('include_rotor_inertia').value)
        self.timeout = float(gp('command_timeout').value)

        self.U_cmd = np.zeros(self.n)
        self.t_cmd = None
        self.q_dot = None
        self.t_prev = None

        self.create_subscription(
            Float64MultiArray, gp('voltage_topic').value, self.voltage_cb, 10)
        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)

        self.torque_pub = self.create_publisher(
            Float64MultiArray, gp('torque_topic').value, 10)
        self.state_pub = self.create_publisher(
            Float64MultiArray, gp('state_topic').value, 10)

        self.timer = self.create_timer(1.0 / float(gp('rate').value), self.step_cb)

        self.get_logger().info(
            f'DC motors for {self.joint_names}:\n  {self.params.describe()}')

    # --------------------------------------------------------

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def voltage_cb(self, msg: Float64MultiArray):
        if len(msg.data) != self.n:
            self.get_logger().warn(
                f'Voltage command must have {self.n} values', throttle_duration_sec=2.0)
            return
        self.U_cmd = np.array(msg.data, dtype=float)
        self.t_cmd = self.now()

    def joint_state_cb(self, msg: JointState):
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            idx = [index[name] for name in self.joint_names]
        except KeyError:
            return
        if len(msg.velocity) == len(msg.name):
            self.q_dot = np.array([msg.velocity[i] for i in idx])

    # --------------------------------------------------------

    def step_cb(self):

        if self.q_dot is None:
            return

        now = self.now()

        if self.t_prev is None:
            self.t_prev = now
            return

        dt = now - self.t_prev
        if dt <= 0.0:
            return
        dt = min(dt, 0.02)
        self.t_prev = now

        # Нет свежей команды -> усилитель выключен (U = 0)
        if self.t_cmd is None or now - self.t_cmd > self.timeout:
            U = np.zeros(self.n)
        else:
            U = self.U_cmd

        a = self.acc.update(self.q_dot, dt)
        if not self.include_rotor_inertia:
            a = np.zeros(self.n)

        m = self.motors.step(U, self.q_dot, a, dt)

        msg = Float64MultiArray()
        msg.data = [float(v) for v in m['tau']]
        self.torque_pub.publish(msg)

        st = Float64MultiArray()
        st.data = [float(v) for v in np.concatenate([m['U'], m['i_avg'], m['tau']])]
        self.state_pub.publish(st)

    def stop(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * self.n
        self.torque_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)
    node = DCMotorSim()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
