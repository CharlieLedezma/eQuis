#!/usr/bin/env python3
import rospy
from adam.msg import Angles
from lx16a import *
import time
x=0

LX16A.initialize("/dev/ttyUSB0")

servos = [LX16A(1), LX16A(2), LX16A(3), LX16A(4), LX16A(5), LX16A(6),LX16A(7), LX16A(8), 
          LX16A(9), LX16A(10), LX16A(11), LX16A(13), LX16A(14), LX16A(15), LX16A(16), 
          LX16A(17), LX16A(18)]


def lx16a_servo_callback(lx16a_servo_message):
    global x
    mult=2
    while ((x*mult)+1) < len(lx16a_servo_message.angle):
    	rospy.loginfo("servo[%d] angle received: (%d)", lx16a_servo_message.angle[x*mult], 
    													lx16a_servo_message.angle[(x*mult)+1])
    	servos[lx16a_servo_message.angle[x*mult]].moveTimeWaitWrite(lx16a_servo_message.angle[(x*mult)+1])
    	x = x+1
    LX16A.moveStartAll()
    x=0
    print("")
    
rospy.init_node('servo_angles_subscriber_node', anonymous=True)

rospy.Subscriber("servo_angles_topic", Angles, lx16a_servo_callback)

# spin() simply keeps python from exiting until this node is stopped
rospy.spin()
