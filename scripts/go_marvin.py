#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist 

def robot_publisher():
    process = None
    temp_file_name = None
    try:
        pub = rospy.Publisher('/tom/cmd_vel', Twist, latch=True)
        msg = Twist()
        msg.linear.x = 1.0
        pub.publish(msg)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
        rospy.init_node('go_marvin', anonymous=True)
        robot_publisher()
