#!/usr/bin/env python3
# license removed for brevity
import rospy
from adam.msg import Angles
import time

# pacote de dados [posição do motor no vetor, ang do motor]
home = [0,91, 1,114, 2,159, 3,150, 4,205, 5,134, 6,112, 7,122, 8,118, 9,123,
        10,125, 11,121, 12,120, 13,169, 14,232, 15,72, 16,49]
mov1 = [6,130, 8,113, 9,99, 14,217, 15,128, 16,87]
mov2 = [5,128, 6,121, 8,120, 9,109, 14,229, 16,94]
mov3 = [4,197, 9,122, 13,182, 15,92, 16,79]
Conjunto_Mov = [home, mov1, mov2, mov3]


#create a new publisher. we specify the topic name, then type of message then the queue size
pub = rospy.Publisher('servo_angles_topic', Angles, queue_size=10)

#we need to initialize the node
rospy.init_node('lx16a_servo_publisher_node', anonymous=True)

#set the loop rate
rate = rospy.Rate(1) # 1hz
time.sleep(4)

#keep publishing until a Ctrl-C is pressed
#while not rospy.is_shutdown():
    
for x in range(len(Conjunto_Mov)):
  ang = Angles()
  ang.angle = Conjunto_Mov[x] 
  rospy.loginfo("I publish:")
  rospy.loginfo(ang)
  pub.publish(ang)
  rate.sleep()

