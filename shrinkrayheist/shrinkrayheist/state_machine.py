#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

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
        # self.ackermann_sub = self.create_subscription(
        #     AckermannDriveStamped,
        #     '/vesc/low_level/ackermann_cmd',
        #     self.ackermann_callback,
        #     10)

        # Timer to keep checking / acting
        self.timer = self.create_timer(0.1, self.run_state_machine)  # 2Hz
        self.steering_angle = 0
        # Detection flags
        self.banana_detected = False
        self.person_detected = False
        self.red_light_detected = False

        self.get_logger().info("State Machine Initialized.")
    # def ackermann_callback(self, msg):
    #     self.steering_angle = msg.drive.steering_angle
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
        drive_msg = AckermannDriveStamped()

        if self.banana_detected or self.person_detected or self.red_light_detected: #or not self.path_planned: 
            if self.person_detected or self.red_light_detected: #stopped state
                self.get_logger().info("In STOPPED state!")
                drive_msg.header.stamp = self.get_clock().now().to_msg()
                drive_msg.drive.speed = 0.0
                # drive_msg.drive.steering_angle = self.steering_angle
                self.drive_pub.publish(drive_msg)
            
            elif self.banana_detected: #wait state
                self.get_logger().info("In WAIT state!")
                drive_msg.header.stamp = self.get_clock().now().to_msg()
                # drive_msg.drive.steering_angle = self.steering_angle
                drive_msg.drive.speed = 0.0  
                self.drive_pub.publish(drive_msg)
        else:
            # drive_msg.drive.speed = 0.0  
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = 1.0
            # drive_msg.drive.steering_angle = self.steering_angle
            self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateMachine()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
