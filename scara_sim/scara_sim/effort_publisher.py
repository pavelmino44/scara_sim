import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import csv
import os
from ament_index_python import get_package_share_directory

class EffortPublisher(Node):
    def __init__(self):
        super().__init__('effort_publisher')
        # Публикуем команды в топик контроллера усилий
        self.pub = self.create_publisher(Float64MultiArray, '/effort_controller/commands', 10)
        self.timer = self.create_timer(0.001, self.timer_callback)  # 1000 Гц
        self.data = self.load_csv()
        self.idx = 0
        self.start_time = self.get_clock().now()

    def load_csv(self):
        # Путь к файлу CSV (можно передать параметром)
        pkg_scara_sim = get_package_share_directory('scara_sim')
        csv_path = os.path.join(pkg_scara_sim, 'data', 'moments.csv')
        
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # пропустить заголовок
            for row in reader:
                data.append((float(row[0]), float(row[1]), float(row[2])))
        return data

    def timer_callback(self):
        if self.idx >= len(self.data):
            return
        t_sim = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        t_csv, tau1, tau2 = self.data[self.idx]
        # Простая синхронизация по номеру шага (если частота совпадает)
        msg = Float64MultiArray()
        msg.data = [tau1, tau2]
        self.pub.publish(msg)
        self.idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = EffortPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()