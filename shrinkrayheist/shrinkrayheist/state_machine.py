#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        
        # States
        self.state = "PATH_PLANNING"  # other possible: "STOPPED"
        self.banana_detected = False
        self.person_detected = False
        self.red_light_detected = False
        
        # Subscribers 
        self.banana_sub = self.create_subscription(Bool, '/banana_detected', self.banana_callback, 10)
        self.person_sub = self.create_subscription(Bool, '/shoe_detected', self.person_callback, 10)
        self.traffic_light_sub = self.create_subscription(Bool, '/red_light_detected', self.traffic_light_callback, 10)
        self.path_planned = self.create_subscription(Bool, '/path_planned', self.path_planned_callback, 10)
        self.ackermann_sub = self.create_subscription(
            AckermannDriveStamped,
            '/vesc/low_level/ackermann_cmd',
            self.ackermann_callback,
            10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_cb, 10)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose",self.goal_cb, 10)
        
        #Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/safety", 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.start_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.timer = self.create_timer(0.1, self.run_state_machine)  #Timer to keep checking / acting

        self.odom_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.odom_callback, 10) # Additional stopping logic
       
        self.stop_points = PoseArray()

      

        # state machine bookkeeping
        self.current_stop_index = 1
        self.state = "PREPLAN"    # will cycle: PREPLAN GO_TO_STOP, DWELL, GO_TO_STOP, DWELL, GO_TO_GOAL, STOPPED
        self.stop_threshold = 0.2    # meters
        self.dwell_duration = 3.0    # seconds to wait at each stop
        self.stop_start_time = None



        self.get_logger().info("State Machine Initialized.")

    # helper functions for replanning
    def pose_cb(self, msg: PoseWithCovarianceStamped):
        pose = Pose()
        pose.position = msg.pose.pose.position
        pose.orientation = msg.pose.pose.orientation
        self.stop_points.poses.append(pose)

    def goal_cb(self, msg):
        self.stop_points.poses.append(msg.pose)
    def odom_callback(self, msg: Odometry):
            self.latest_pose = msg
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.current_pose = np.array([x, y])

    def arrived(self, target: Pose):
        if self.current_pose is None:
            return False
        dx = target.position.x - self.current_pose[0]
        dy = target.position.y - self.current_pose[1]
        return np.hypot(dx, dy) < self.stop_threshold

    def ackermann_callback(self, msg):
        self.steering_angle = msg.drive.steering_angle
    def banana_callback(self, msg: Bool):
        self.banana_detected = msg.data
        if self.banana_detected:
            self.get_logger().info(f"Banana detected: {msg.data}")
            # Stop the car while banana is detected
            # stop_msg = AckermannDriveStamped()
            # stop_msg.drive.speed = 0.0
            # self.drive_pub.publish(stop_msg)

    def person_callback(self, msg: Bool): #not necessary if using safety controller
        self.person_detected = msg.data
        if msg.data:
            self.get_logger().info(f"Person detected: {msg.data}")            

    def traffic_light_callback(self, msg: Bool):
        self.red_light_detected = msg.data
        if msg.data:
            self.get_logger().info(f"Red Light detected: {msg.data}")

    def path_planned_callback(self, msg: Bool):
        self.red_light_detected = msg.data
        if msg.data:
            self.get_logger().info(f"Path Planning has Finished: {msg.data}")

    def run_state_machine(self):
            # 1) high-level navigation state
        if self.state == "PREPLAN":
            if len(self.stop_points.poses) ==4:
                self.state = "GO_TO_STOP"
                for i, pose in enumerate(self.stop_points.poses):
                    self.get_logger().info(f"Pose {i+1}: {pose}")
        elif self.state == "GO_TO_STOP":
            stop = self.stop_points.poses[self.current_stop_index]
            if self.arrived(stop):
                self.get_logger().info(f"Reached stop {self.current_stop_index+1}, dwelling…")
                self.state = "PLAN"
                self.stop_start_time = self.get_clock().now()
                self.get_logger().info(f"Planning from stop {self.current_stop_index} to stop {self.current_stop_index + 1}")
                self.get_logger().info(f"Start pose: {self.stop_points.poses[self.current_stop_index - 1]}")
                self.get_logger().info(f"Goal pose: {self.stop_points.poses[self.current_stop_index]}")
        elif self.state == "DWELL":
            # pub goal pose here

            elapsed = (self.get_clock().now() - self.stop_start_time).nanoseconds / 1e9
            if elapsed >= self.dwell_duration:
                self.current_stop_index += 1
                if self.current_stop_index < len(self.stop_points.poses) - 1:
                    self.state = "GO_TO_STOP"
                else:
                    self.state = "GO_TO_GOAL"
                self.get_logger().info(f"Leaving dwell, next state: {self.state}")
        elif self.state == "PLAN":
            stop = self.stop_points.poses[self.current_stop_index]
            
            # Use current position from odometry instead of previous waypoint
            start_msg = PoseWithCovarianceStamped()
            start_msg.header.stamp = self.get_clock().now().to_msg()
            start_msg.header.frame_id = "map"  # or whatever frame is appropriate
            start_msg.pose.pose = self.latest_pose.pose.pose

            stop_stamped = PoseStamped()
            stop_stamped.header.stamp = self.get_clock().now().to_msg()
            stop_stamped.header.frame_id = "map"  # or whatever frame is appropriate
            stop_stamped.pose = stop  # Assign the Pose object to PoseStamped
                        
            self.start_pub.publish(start_msg)
            self.goal_pub.publish(stop_stamped)
            self.state = "DWELL"
            self.get_logger().info(f"Reached Plan, next state: {self.state}")

        elif self.state == "GO_TO_GOAL":
            stop = self.stop_points.poses[self.current_stop_index]
            
            # Use current position from odometry instead of previous waypoint
            start_msg = PoseWithCovarianceStamped()
            start_msg.header.stamp = self.get_clock().now().to_msg()
            start_msg.header.frame_id = "map"  # or whatever frame is appropriate
            start_msg.pose.pose = self.latest_pose.pose.pose
            
            stop_stamped = PoseStamped()
            stop_stamped.header.stamp = self.get_clock().now().to_msg()
            stop_stamped.header.frame_id = "map"  # or whatever frame is appropriate
            stop_stamped.pose = stop  # Assign the Pose object to PoseStamped
            self.start_pub.publish(start_msg)
            self.goal_pub.publish(stop_stamped)
            
            if self.arrived(self.stop):
                self.get_logger().info("Final goal reached, stopping.")
                self.state = "STOPPED"

        # 2) safety overrides (banana, person, red-light) still apply
        if self.banana_detected or self.person_detected or self.red_light_detected:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)
            return

        # 3) otherwise let the planner/follower handle motion
        #    by keeping speed > 0.0 (your pure pursuit node will chase the published /goal_pose)
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = 1.0
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateMachine()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

