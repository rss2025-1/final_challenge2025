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

        # Timer to keep checking / acting
        self.timer = self.create_timer(0.5, self.run_state_machine)  # 2Hz

        # Detection flags
        self.banana_detected = False
        self.person_detected = False
        self.red_light_detected = False

        self.get_logger().info("State Machine Initialized.")

    def banana_callback(self, msg: Bool):
        self.banana_detected = msg.data
        if msg.data:
            self.get_logger().info(f"Banana detected: {msg.data}")

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

        if self.banana_detected or self.person_detected or self.red_light_detected or not self.path_planned: #stopped state
            # If anything detected or path planning, stop
            if self.person_detected or self.red_light_detected:
                self.get_logger().info("Switching to STOPPED state!")
                drive_msg.drive.speed = 0.0
                self.drive_pub.publish(drive_msg)
            
            elif self.banana_detected:
                self.get_logger().info("Switching to STOPPED state and rerouting path!")
                #TODO: reroute path by re path planning, stop car while path planning from car to banana, and then make it move again, 
  

        else:
            drive_msg.drive.speed = 1.0
            self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateMachine()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
