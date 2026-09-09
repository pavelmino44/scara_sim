#!/usr/bin/env python3

import os
import csv

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory


class EffortPublisher(Node):

    def __init__(self):

        super().__init__(
            'effort_publisher'
        )

        # ====================================================
        # ПАРАМЕТРЫ
        # ====================================================

        self.control_dt = 0.001

        # ====================================================
        # ЗАГРУЗКА TRAJECTORY.CSV
        # ====================================================

        package_path = get_package_share_directory(
            'scara_sim'
        )

        csv_path = os.path.join(
            package_path,
            'data',
            'trajectory.csv'
        )

        self.get_logger().info(
            f'Loading trajectory:\n{csv_path}'
        )

        self.data = self.load_csv(
            csv_path
        )

        self.time = self.data[:, 0]

        # tau1 / tau2 находятся
        # в последних двух колонках
        self.tau1 = self.data[:, -2]
        self.tau2 = self.data[:, -1]

        self.duration = self.time[-1]

        self.get_logger().info(
            f'Loaded {len(self.time)} points'
        )

        self.get_logger().info(
            f'Trajectory duration: '
            f'{self.duration:.3f} s'
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
        # ВРЕМЯ ЗАПУСКА
        # ====================================================

        self.start_time = None

        self.finished = False

        # ====================================================
        # TIMER
        # ====================================================

        self.timer = self.create_timer(
            self.control_dt,
            self.control_callback
        )

        self.get_logger().info(
            'Effort publisher started.'
        )

    # ========================================================
    # CSV
    # ========================================================

    def load_csv(self, path):

        rows = []

        with open(
            path,
            'r'
        ) as file:

            reader = csv.reader(file)

            # header
            next(reader)

            for row in reader:

                rows.append([
                    float(value)
                    for value in row
                ])

        if len(rows) == 0:
            raise RuntimeError(
                'Trajectory CSV is empty.'
            )

        return np.array(
            rows,
            dtype=float
        )

    # ========================================================
    # INTERPOLATION
    # ========================================================

    def get_torque(self, t):

        tau1 = np.interp(
            t,
            self.time,
            self.tau1
        )

        tau2 = np.interp(
            t,
            self.time,
            self.tau2
        )

        return tau1, tau2

    # ========================================================
    # CALLBACK
    # ========================================================

    def control_callback(self):

        now = self.get_clock().now()

        if self.start_time is None:

            self.start_time = now

            self.get_logger().info(
                'Trajectory execution started.'
            )

            return

        elapsed = (
            now - self.start_time
        ).nanoseconds * 1e-9

        # ====================================================
        # КОНЕЦ ТРАЕКТОРИИ
        # ====================================================

        if elapsed >= self.duration:

            if not self.finished:

                self.publish_torque(
                    0.0,
                    0.0
                )

                self.finished = True

                self.get_logger().info(
                    'Trajectory finished.'
                )

            return

        # ====================================================
        # ПОЛУЧАЕМ МОМЕНТ
        # ====================================================

        tau1, tau2 = self.get_torque(
            elapsed
        )

        # ====================================================
        # ПРЯМОЕ УПРАВЛЕНИЕ МОМЕНТАМИ
        # ====================================================

        self.publish_torque(
            tau1,
            tau2
        )

    # ========================================================
    # PUBLISH
    # ========================================================

    def publish_torque(
        self,
        tau1,
        tau2
    ):

        msg = Float64MultiArray()

        msg.data = [
            float(tau1),
            float(tau2)
        ]

        self.publisher.publish(
            msg
        )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = EffortPublisher()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.publish_torque(
            0.0,
            0.0
        )

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()