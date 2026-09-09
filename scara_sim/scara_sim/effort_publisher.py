import csv
import os

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from ament_index_python.packages import get_package_share_directory


class EffortPublisher(Node):

    def __init__(self):

        super().__init__(
            'effort_publisher'
        )

        # ====================================================
        # PUBLISHER
        # ====================================================

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/effort_controller/commands',
            10
        )

        # ====================================================
        # SUBSCRIBER
        # ====================================================

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # ====================================================
        # ПАРАМЕТРЫ PD-КОНТРОЛЛЕРА
        # ====================================================

        # Начальные значения.
        #
        # Их потом можно подобрать экспериментально.
        #
        self.Kp = np.array([
            80.0,
            60.0
        ])

        self.Kd = np.array([
            8.0,
            6.0
        ])

        # Ограничение момента.
        self.max_effort = 100.0

        # ====================================================
        # ЗАГРУЗКА ТРАЕКТОРИИ
        # ====================================================

        self.trajectory = self.load_trajectory()

        self.trajectory_index = 0

        # ====================================================
        # СОСТОЯНИЕ РОБОТА
        # ====================================================

        self.current_q = None
        self.current_q_dot = None

        # ====================================================
        # ВРЕМЯ
        # ====================================================

        self.start_time = None

        # Управление с частотой 1000 Гц.
        self.timer = self.create_timer(
            0.001,
            self.control_callback
        )

        self.get_logger().info(
            'Effort controller started.'
        )

    # ========================================================
    # ЗАГРУЗКА CSV
    # ========================================================

    def load_trajectory(self):

        package_dir = get_package_share_directory(
            'scara_sim'
        )

        csv_path = os.path.join(
            package_dir,
            'data',
            'trajectory.csv'
        )

        trajectory = []

        with open(
            csv_path,
            'r'
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                trajectory.append({
                    'time': float(row['time']),

                    'q1': float(row['q1']),
                    'q2': float(row['q2']),

                    'q1_dot': float(row['q1_dot']),
                    'q2_dot': float(row['q2_dot']),

                    'tau1': float(row['tau1_ff']),
                    'tau2': float(row['tau2_ff'])
                })

        self.get_logger().info(
            f'Loaded {len(trajectory)} trajectory points.'
        )

        return trajectory

    # ========================================================
    # JOINT STATES
    # ========================================================

    def joint_state_callback(
        self,
        msg: JointState
    ):

        try:

            i1 = msg.name.index('joint1')
            i2 = msg.name.index('joint2')

            self.current_q = np.array([
                msg.position[i1],
                msg.position[i2]
            ])

            self.current_q_dot = np.array([
                msg.velocity[i1],
                msg.velocity[i2]
            ])

        except ValueError:

            self.get_logger().warn(
                'joint1 or joint2 not found in /joint_states'
            )

    # ========================================================
    # ОСНОВНОЙ КОНТРОЛЛЕР
    # ========================================================

    def control_callback(self):

        # Без состояния робота управление невозможно.
        if self.current_q is None:
            return

        if self.start_time is None:

            self.start_time = (
                self.get_clock().now()
            )

        # Текущее время движения.
        elapsed = (
            self.get_clock().now()
            - self.start_time
        ).nanoseconds / 1e9

        # ====================================================
        # ЗАВЕРШЕНИЕ ТРАЕКТОРИИ
        # ====================================================

        if (
            self.trajectory_index
            >= len(self.trajectory)
        ):

            self.publish_effort(
                np.zeros(2)
            )

            return

        # ====================================================
        # ПОИСК АКТУАЛЬНОЙ ТОЧКИ
        # ====================================================

        # Двигаемся по CSV до тех пор,
        # пока время точки меньше текущего времени.
        while (
            self.trajectory_index
            < len(self.trajectory) - 1
            and
            self.trajectory[
                self.trajectory_index + 1
            ]['time'] <= elapsed
        ):

            self.trajectory_index += 1

        point = self.trajectory[
            self.trajectory_index
        ]

        # ====================================================
        # ЗАДАННЫЕ КООРДИНАТЫ
        # ====================================================

        q_desired = np.array([
            point['q1'],
            point['q2']
        ])

        q_dot_desired = np.array([
            point['q1_dot'],
            point['q2_dot']
        ])

        tau_feedforward = np.array([
            point['tau1'],
            point['tau2']
        ])

        # ====================================================
        # ОШИБКИ
        # ====================================================

        position_error = (
            q_desired
            - self.current_q
        )

        velocity_error = (
            q_dot_desired
            - self.current_q_dot
        )

        # ====================================================
        # PD-КОРРЕКЦИЯ
        # ====================================================

        tau_feedback = (
            self.Kp * position_error
            +
            self.Kd * velocity_error
        )

        # ====================================================
        # ОБЩИЙ МОМЕНТ
        # ====================================================

        tau = (
            tau_feedforward
            +
            tau_feedback
        )

        # Защита от слишком больших моментов.
        tau = np.clip(
            tau,
            -self.max_effort,
            self.max_effort
        )

        self.publish_effort(tau)

    # ========================================================
    # ПУБЛИКАЦИЯ МОМЕНТА
    # ========================================================

    def publish_effort(
        self,
        tau
    ):

        msg = Float64MultiArray()

        msg.data = [
            float(tau[0]),
            float(tau[1])
        ]

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(
        args=args
    )

    node = EffortPublisher()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()