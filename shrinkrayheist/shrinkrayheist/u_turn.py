import rclpy
from rclpy.node import Node
from rclpy.time import Time
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class UTurn(Node):
    def __init__(self):
        super().__init__('u_turn')

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/low_level/input/safety', 10)
        self.odom_sub = self.create_subscription(Odometry, '/pf/pose/odom', self.odom_cb, 10)
        self.path_pub = self.create_publisher(Path, '/turn_path', 10)
        self.turn_done_pub = self.create_publisher(Bool, '/turn_completed', 10)
        self.safety_enable_pub = self.create_publisher(Bool, '/safety_enable', 10)

        self.trigger_turn_sub = self.create_subscription(Bool, '/trigger_uturn', self.trigger_callback, 10)

        # TODO: tune these parameters to real car turn in skinny hallway
        self.forward_speed = 1.0
        self.reverse_speed = -1.0
        self.steering_angle = 0.25
        self.turn_direction = 1 # 1 for right, -1 for left

        # timer for each segment of 3-point turn
        self.forward_time = 2.0
        self.reverse_time = 2.0
        self.final_forward_time = 2.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'
        self.visualizing = True
        self.state = -1 # idle state

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.timer_callback)

        # self.get_logger().info("Starting 3-point turn.")
    
    def trigger_callback(self, msg: Bool):
        if msg.data and self.state == -1:
            self.get_logger().info("Starting 3-point turn.")
            self.state = 0
            self.start_time = self.get_clock().now()
            self.visualizing = True
            self.path_msg.poses.clear()

    def timer_callback(self):
        if self.state == -1:
            return # idle, waiting to get to goal pose

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = now.to_msg()

        if self.state == 0 and elapsed < self.forward_time:
            # forward turn to the right
            drive_cmd.drive.speed = self.forward_speed
            drive_cmd.drive.steering_angle = -self.turn_direction * self.steering_angle

        elif self.state == 0:
            self.state = 1
            self.start_time = now
            return

        elif self.state == 1 and elapsed < self.reverse_time:
            # reverse turn to the left
            drive_cmd.drive.speed = self.reverse_speed
            drive_cmd.drive.steering_angle = self.turn_direction * self.steering_angle

        elif self.state == 1:
            self.state = 2
            self.start_time = now
            return

        elif self.state == 2 and elapsed < self.final_forward_time:
            # forward straightening out
            drive_cmd.drive.speed = self.forward_speed
            drive_cmd.drive.steering_angle = 0.0

        elif self.state == 2:
            # done with turn
            self.state = 3
            self.visualizing = False
            self.get_logger().info("3-point turn complete.")

            self.turn_done_pub.publish(Bool(data=True))

            # turning safety controller back on
            self.safety_enable_pub.publish(Bool(data=True))
            self.start_time = now
            return

        else:
            # stopping car after turn
            #TODO: add call back to path replanner to continue back to start point
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)

            self.get_logger().info("Turn finished, stopping and exiting.")
            self.destroy_timer(self.timer)
            return

        self.drive_pub.publish(drive_cmd)

    def odom_cb(self, msg: Odometry):
        if not self.visualizing:
            return

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = 'odom'
        pose_stamped.pose = msg.pose.pose

        self.path_msg.poses.append(pose_stamped)
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = UTurn()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
