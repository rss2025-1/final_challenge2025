#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        
        # States
        self.state = "PATH_PLANNING"  # other possible: "STOPPED"
        
        # Subscribers (these should publish True when detected)
        self.banana_sub = self.create_subscription(Bool, '/banana_detected', self.banana_callback, 10)
        self.person_sub = self.create_subscription(Bool, '/shoe_detected', self.person_callback, 10)
        self.traffic_light_sub = self.create_subscription(Bool, '/red_light_detected', self.traffic_light_callback, 10)
        self.path_planned = self.create_subscription(Bool, '/path_planned', self.path_planned_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/safety", 10)
        # Publisher to path planner (example, could be more complex in reality)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.ackermann_sub = self.create_subscription(
            AckermannDriveStamped,
            '/vesc/low_level/ackermann_cmd',
            self.ackermann_callback,
            10)

        # Timer to keep checking / acting
        self.timer = self.create_timer(0.5, self.run_state_machine)  # 2Hz
        self.steering_angle = 0
        # Detection flags
        self.banana_detected = False
        self.person_detected = False
        self.red_light_detected = False

        # Additional stoping logic
        self.odom_pub = self.create_suscription(Odometry, "/pf/pose/odom", 10)


        self.stop_points = []
        p1 = PoseStamped()
        p1.pose.position.x, p1.pose.position.y = 1.0, 2.0   # example coords
        p2 = PoseStamped()
        p2.pose.position.x, p2.pose.position.y = 3.5, -0.5  # example coords
        self.stop_points = [p1, p2]

        # final goal (you might set this elsewhere, or load as a param too)
        self.final_goal = PoseStamped()
        self.final_goal.pose.position.x, self.final_goal.pose.position.y = 5.0, 5.0

        # state machine bookkeeping
        self.current_stop_index = 0
        self.state = "GO_TO_STOP"    # will cycle: GO_TO_STOP, DWELL, GO_TO_STOP, DWELL, GO_TO_GOAL, STOPPED
        self.stop_threshold = 0.2    # meters
        self.dwell_duration = 3.0    # seconds to wait at each stop
        self.stop_start_time = None



        self.get_logger().info("State Machine Initialized.")

    # helper functions for replanning
    def odom_callback(self, msg: Odometry):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.current_pose = np.array([x, y])

    def arrived(self, target: PoseStamped):
        if self.current_pose is None:
            return False
        dx = target.pose.position.x - self.current_pose[0]
        dy = target.pose.position.y - self.current_pose[1]
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
        if self.state == "GO_TO_STOP":
            goal = self.stop_points[self.current_stop_index]
            self.goal_pub.publish(goal)
            if self.arrived(goal):
                self.get_logger().info(f"Reached stop {self.current_stop_index+1}, dwelling…")
                self.state = "DWELL"
                self.stop_start_time = self.get_clock().now()

        elif self.state == "DWELL":
            elapsed = (self.get_clock().now() - self.stop_start_time).nanoseconds / 1e9
            if elapsed >= self.dwell_duration:
                self.current_stop_index += 1
                if self.current_stop_index < len(self.stop_points):
                    self.state = "GO_TO_STOP"
                else:
                    self.state = "GO_TO_GOAL"
                self.get_logger().info(f"Leaving dwell, next state: {self.state}")

        elif self.state == "GO_TO_GOAL":
            self.goal_pub.publish(self.final_goal)
            if self.arrived(self.final_goal):
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

