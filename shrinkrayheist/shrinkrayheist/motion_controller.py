import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from std_msgs.msg import Header

class AutonomousReplanner(Node):
    def __init__(self):
        super().__init__('autonomous_replanner')

        # Publishers to path planning's topics
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Example setup: Subscribing to a topic giving PF estimate
        self.create_subscription(Pose, '/pf_estimated_pose', self.pf_pose_callback, 10)

        # Example: Save your original goal pose here
        self.goal_pose = self.create_goal_pose(4.0, 0.0, 0.0)  # x, y, yaw
        self.replanned = False

        self.get_logger().info('Autonomous replanner node started.')

    def pf_pose_callback(self, msg: Pose):
        """ Callback to receive particle filter estimated pose. """

        if not self.replanned:
            self.get_logger().info('Received particle filter pose, sending replanning request...')

            self.publish_initial_pose(msg)
            self.publish_goal_pose(self.goal_pose)

            self.replanned = True  # Only replan once unless you want to re-enable it

    def publish_initial_pose(self, pose: Pose):
        """ Publish the current robot pose as initial pose. """
        msg = PoseWithCovarianceStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.pose = pose
        # Optional: Fill covariance matrix if you have better estimates
        # msg.pose.covariance = [0.0] * 36

        self.initial_pose_pub.publish(msg)
        self.get_logger().info('Published initial pose.')

    def publish_goal_pose(self, pose: Pose):
        """ Publish the final goal pose. """
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose = pose

        self.goal_pose_pub.publish(msg)
        self.get_logger().info('Published goal pose.')

    def create_goal_pose(self, x, y, yaw):
        """ Utility function to generate a Pose from (x, y, yaw) """
        goal_pose = Pose()
        goal_pose.position.x = x
        goal_pose.position.y = y
        goal_pose.position.z = 0.0

        import math
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, yaw)
        goal_pose.orientation.x = q[0]
        goal_pose.orientation.y = q[1]
        goal_pose.orientation.z = q[2]
        goal_pose.orientation.w = q[3]

        return goal_pose

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousReplanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
