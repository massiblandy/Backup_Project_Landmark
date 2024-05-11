import rclpy
from rclpy.node import Node
from custom_interfaces.msg import NeckPosition

class MotorSubscriber(Node):
    def __init__(self):
        super().__init__('motor_subscriber')
        self.neck_subscription = self.create_subscription(NeckPosition, '/neck_position', self.topic_callback_neck, 10)

    def topic_callback_neck(self, msg):
        neck_sides = msg.position19
        neck_up = msg.position20
        print(f"Neck Position: Sides {neck_sides}, Up {neck_up}")

def main(args=None):
    rclpy.init(args=args)
    motor_subscriber = MotorSubscriber()
    rclpy.spin(motor_subscriber)
    motor_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
