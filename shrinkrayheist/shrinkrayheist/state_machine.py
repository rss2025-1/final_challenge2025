#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        
        # States
        self.banana_detected = False
        self.person_detected = False
        self.red_light_detected = False
        
        # Subscribers 
        self.banana_sub = self.create_subscription(Bool, '/banana_detected', self.banana_callback, 10)
        self.person_sub = self.create_subscription(Bool, '/shoe_detected', self.person_callback, 10)
        self.traffic_light_sub = self.create_subscription(Bool, '/red_light_detected', self.traffic_light_callback, 10)

        # Path Planning Subscribers
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_cb, 10)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose",self.goal_cb, 10)
        self.goal_reached_sub = self.create_subscription(Bool, "/goal_reached", self.goal_reached_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.odom_cb, 10)
        
        #Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/safety", 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/current_goal_pose', 10)
        self.start_pub = self.create_publisher(PoseWithCovarianceStamped, '/current_start_pose', 10)

        self.timer = self.create_timer(0.1, self.run_state_machine)  #Timer to keep checking / acting

        # Motion controller-relate states
        self.stop_points = PoseArray()
        self.current_stop_index = 1 # index of the current stop point (starts at 1 since 0 is the start point) 
        # order is 1, 2, 3, 2, 1, 0 (banana 1, banana 2, end, banana 2, banana 1, start)
        self.plan_state = True    # boolean which when true allows the state machine to send new start and goal pose to the planner
        self.current_pose = PoseWithCovarianceStamped() # This is constantly updated by the odom_sub callback and used as the starting point for path planning

        # States for logging
        self.person_log = False 

        self.get_logger().info("State Machine Initialized.")

    # helper functions for replanning
    def pose_cb(self, msg: PoseWithCovarianceStamped):
        pose = Pose()
        pose.position = msg.pose.pose.position
        pose.orientation = msg.pose.pose.orientation
        self.stop_points.poses.append(pose)
        self.get_logger().info(f"There are {len(self.stop_points.poses)} stop points.")

    def goal_cb(self, msg):
        self.stop_points.poses.append(msg.pose)
        self.get_logger().info(f"There are {len(self.stop_points.poses)} stop points.")
        # Add in the reverse path
        if len(self.stop_points.poses) == 4:
            self.stop_points.poses.append(self.stop_points.poses[2])
            self.stop_points.poses.append(self.stop_points.poses[1])
            self.stop_points.poses.append(self.stop_points.poses[0])
            self.get_logger().info("Adding reverse path.")
            self.get_logger().info(f"There are {len(self.stop_points.poses)} stop points.")
            
    def odom_cb(self, msg: Odometry):
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose

    def banana_callback(self, msg: Bool):
        self.banana_detected = msg.data
        if self.banana_detected:
            # self.get_logger().info(f"Banana detected: {msg.data}")
            pass

    def person_callback(self, msg: Bool): #not necessary if using safety controller
        self.person_detected = msg.data
        if msg.data:
            # self.get_logger().info(f"Person detected: {msg.data}")
            pass            

    def traffic_light_callback(self, msg: Bool):
        self.red_light_detected = msg.data
        if msg.data:
            # self.get_logger().info(f"Red Light detected: {msg.data}")
            pass

    def goal_reached_cb(self, msg: Bool):
        self.goal_reached = msg.data
        if msg.data:
            self.get_logger().info(f"Goal reached: {msg.data}")
            self.plan_state = True
            if self.current_stop_index == len(self.stop_points.poses) - 1:
                self.get_logger().info("All stop points reached.")

    def motion_controller(self):
        # If we can plan and we have 4 stop points (start, banana 1, banana 2, end), proceed
        if self.plan_state and (len(self.stop_points.poses) == 7):
            self.plan_state = False
            self.get_logger().info("Planning Path...")
            # Making a U-turn
            if self.current_stop_index == 4:
                self.get_logger().info("Making a U-turn...")
                # U-turn function

            # This publishes the start and goal pose to the planner, which automatically follows the path as well
            self.start_pub.publish(self.current_pose)
            goal_pose = PoseStamped()
            goal_pose.pose = self.stop_points.poses[self.current_stop_index]
            self.goal_pub.publish(goal_pose)

            # Update the stop location going forward
            if self.current_stop_index < len(self.stop_points.poses) - 1:
                self.current_stop_index += 1

    def run_state_machine(self):
        # 1) motion controller
        self.motion_controller()

        # 2) safety overrides (banana, person, red-light) still apply
        if self.banana_detected or self.person_detected or self.red_light_detected:
            if self.person_detected and not self.person_log:
                self.person_log = True
                self.get_logger().info("Person detected. Stopping.")
                
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)
            return

        if self.person_log:
            self.get_logger().info("Person no longer detected. Resuming.")
        self.person_log = False

def main(args=None):
    rclpy.init(args=args)
    node = StateMachine()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

