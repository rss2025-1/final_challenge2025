#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from tf_transformations import quaternion_from_euler


class AutonomousReplanner(Node):
    def __init__(self):
        super().__init__('autonomous_replanner')

        # Publisher for AMCL initial pose
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # Publisher for planner goal
        self.goal_pose_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 10)

        # Subscribe to your particle-filter estimated robot pose
        self.create_subscription(
            Pose,
            '/pf_estimated_pose',
            self.pf_pose_callback,
            10)

        # Define your two intermediate stops + final goal as (x, y, yaw)
        stops = [
            (1.0,  2.0,  0.0),    # Stop #1
            (2.5, -1.0, -1.57),   # Stop #2
            (4.0,  0.0,  0.0),    # Final goal
        ]
        # Preconvert to Pose objects
        self.waypoints = [ self._make_pose(x, y, yaw) for x, y, yaw in stops ]
        self.current_idx = 0

        self.started = False
        self.threshold = 0.2  # meters: how close counts as “reached”

        self.get_logger().info('Autonomous replanner started.')

    def pf_pose_callback(self, msg: Pose):
        # First PF reading: publish initial pose + first goal
        if not self.started:
            self.get_logger().info('PF pose received → publishing initial & first goal')
            self._publish_initial_pose(msg)
            self._publish_goal(self.waypoints[self.current_idx])
            self.started = True
            return

        # Subsequent PF: check distance to current goal
        wp = self.waypoints[self.current_idx]
        dx = msg.position.x - wp.position.x
        dy = msg.position.y - wp.position.y
        dist = math.hypot(dx, dy)

        if dist < self.threshold:
            self.get_logger().info(
                f'Reached waypoint {self.current_idx+1} (dist={dist:.2f} m)')
            self.current_idx += 1

            if self.current_idx < len(self.waypoints):
                self.get_logger().info(
                    f'Sending waypoint #{self.current_idx+1}')
                self._publish_goal(self.waypoints[self.current_idx])
            else:
                self.get_logger().info('✅ All waypoints reached.')

    def _publish_initial_pose(self, pose: Pose):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose = pose
        # Optionally set msg.pose.covariance here
        self.initial_pose_pub.publish(msg)
        self.get_logger().info('→ Initial pose published.')

    def _publish_goal(self, pose: Pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose = pose
        self.goal_pose_pub.publish(msg)
        self.get_logger().info(
            f'→ Goal published: x={pose.position.x:.2f}, y={pose.position.y:.2f}')

    def _make_pose(self, x: float, y: float, yaw: float) -> Pose:
        p = Pose()
        p.position.x = x
        p.position.y = y
        p.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, yaw)
        p.orientation.x = q[0]
        p.orientation.y = q[1]
        p.orientation.z = q[2]
        p.orientation.w = q[3]
        return p


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousReplanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

