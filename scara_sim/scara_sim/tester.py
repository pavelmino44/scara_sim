import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

class JointTest(Node):
    def __init__(self):
        super().__init__('joint_test')
        self.publisher = self.create_publisher(Float64MultiArray, '/effort_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100 Гц
        self.start_time = self.get_clock().now()
        self.duration = 3.0  # секунд движения
        self.amplitude = 5.0  # Н·м на каждый сустав

    def timer_callback(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        msg = Float64MultiArray()
        if elapsed < self.duration:
            msg.data = [self.amplitude, self.amplitude]
        else:
            msg.data = [0.0, 0.0]
            # Останавливаем таймер после окончания
            if elapsed > self.duration + 0.5:
                self.timer.cancel()
                self.get_logger().info('Test finished, shutting down...')
                self.destroy_node()
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointTest()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()