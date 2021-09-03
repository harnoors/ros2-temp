import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import Float32MultiArray

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(Float32MultiArray,'pose_classification',self.cl_callback, 10)
        self.subscription

    async def cl_callback(self, msg):
        result = int(led_data[0])
        resultPr = int(led_data[1])
        print("the predicted class is: ", int(results),"  with probability:", resultPr)

minimal_subscriber = MinimalSubscriber()   

def main():
    rclpy.spin(minimal_subscriber)

if __name__ == "__main__":
    main()