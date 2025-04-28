import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class PDController(Node):
    def __init__(self):
        super().__init__('pd_controller')

        # parameters to change:
        self.kp_steering = 1.5
        self.kd_steering = 0.2

        self.prev_steering_error = 0.0

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped,
            '/raw_drive',  # lane_follower publishing to /raw_drive instead of /drive
            self.drive_callback,
            10
        )

        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.target_steering_angle = 0.0
        self.target_speed = 0.0

    def drive_callback(self, msg):
        # storing incoming desired values
        self.target_steering_angle = msg.drive.steering_angle
        self.target_speed = msg.drive.speed

    def control_loop(self):
        steering_error = self.target_steering_angle
        derivative = steering_error - self.prev_steering_error

        steering_command = (
            self.kp_steering * steering_error +
            self.kd_steering * derivative
        )

        self.prev_steering_error = steering_error

        # new drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = steering_command
        drive_msg.drive.speed = self.target_speed  # keep speed

        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
