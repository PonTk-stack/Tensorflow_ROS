import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf



class RosTensorFlow():
    def __init__(self):

        a = 1

    def main(self):
        
        counter = tf.Variable(0)
        increment = tf.assign_add(counter, 1)

        sess = tf.InteractiveSession()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

#sess.run(tf.variables_initializer([counter]))
        print(counter.eval())
        increment.eval()
        print(counter.eval())
        increment.eval()
        print(counter.eval())
        sess.close()

if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
    exit()
    rospy.spin()
