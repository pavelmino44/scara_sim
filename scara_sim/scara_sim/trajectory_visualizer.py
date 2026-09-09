import rclpy

from rclpy.node import Node

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class TrajectoryVisualizer(Node):

    def __init__(self):

        super().__init__(
            'trajectory_visualizer'
        )

        self.publisher = self.create_publisher(
            Marker,
            '/trajectory_marker',
            10
        )

        # Публикуем траекторию периодически.
        self.timer = self.create_timer(
            1.0,
            self.publish_marker
        )

        self.publish_marker()

    def publish_marker(self):

        marker = Marker()

        # Траектория задаётся в системе координат world.
        marker.header.frame_id = 'world'

        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        marker.ns = 'reference_trajectory'
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Толщина линии.
        marker.scale.x = 0.01

        # Цвет маркера.
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # ====================================================
        # БУКВА П
        # ====================================================

        trajectory_points = [
            (-0.25, -0.25, 0.0),
            (-0.25,  0.25, 0.0),
            ( 0.25,  0.25, 0.0),
            ( 0.25, -0.25, 0.0)
        ]

        for x, y, z in trajectory_points:

            point = Point()

            point.x = x
            point.y = y
            point.z = z

            marker.points.append(
                point
            )

        self.publisher.publish(
            marker
        )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = TrajectoryVisualizer()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()