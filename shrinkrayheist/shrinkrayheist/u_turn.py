import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class UTurnNode(Node):
    def __init__(self):
        super().__init__('three_point_turn_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # speeds ------------------- can change these!!
        self.forward_speed = 0.5
        self.reverse_speed = -0.5
        self.turn_speed = 0.6

        # times for each point ------ can change these!!
        self.forward_time = 2.0
        self.reverse_time = 2.0
        self.final_forward_time = 2.0

        self.turn_direction = 1  # +1 = right, -1 = left

        self.start_time = self.get_clock().now().nanosec / 1e9
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.state = 0  # 0: forward turn, 1: reverse turn, 2: final align, 3: stop/start replanning

        self.state_start_time = time.time()
        self.get_logger().info("Starting 3-Point Turn.")

    def timer_callback(self):
        now = time.time()
        elapsed = now - self.state_start_time
        twist = Twist()

        if self.state == 0 and elapsed < self.forward_time:
            # forward to the right
            twist.linear.x = self.forward_speed
            twist.angular.z = -self.turn_direction * self.turn_speed
        elif self.state == 0:
            self.state += 1
            self.state_start_time = now
            return

        elif self.state == 1 and elapsed < self.reverse_time:
            # reverse turn to the left
            twist.linear.x = self.reverse_speed
            twist.angular.z = self.turn_direction * self.turn_speed
        elif self.state == 1:
            self.state += 1
            self.state_start_time = now
            return

        elif self.state == 2 and elapsed < self.final_forward_time:
            # forward to align back to straight
            twist.linear.x = self.forward_speed
            twist.angular.z = 0.0
        elif self.state == 2:
            self.state += 1
            self.state_start_time = now
            return

        else:
            # stop ------------------------------------- change this to going back to re-planner
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("3-point turn complete.")
            self.destroy_timer(self.timer)

        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ThreePointTurnNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
