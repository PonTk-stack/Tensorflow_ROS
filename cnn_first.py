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
        png = tf.read_file('model/03-02-original.png')
        image = tf.image.decode_png(png, channels=1, dtype=tf.uint8)
        image_float = tf.to_float(image)
# tf.nn.conv2d to 4 tensor
        image_reshape = tf.reshape(image_float, [1, 32, 32, 1])
        print(sess.run(tf.shape(image_float)))

        kernel = tf.constant(
          [
            [ 0, -1, -1, -1,  0],
            [-1,  0,  3,  0, -1],
            [-1,  3,  0,  3, -1],
            [-1,  0,  3,  0, -1],
            [ 0, -1, -1, -1,  0]
          ],
          dtype=tf.float32)
# tf.nn.conv2d is to affect 4 tensor
        #kernel_reshape = tf.reshape(kernel, [5, 5, 1, 1])
        kernel_reshape = tf.reshape(kernel, [-1, 5, 1, 1])
        print(kernel_reshape.eval())

# sraid range for 3 pxel
        strides = [1, 3, 3, 1]

# tatamikomi
        convolution_result = tf.nn.conv2d(
          image_reshape,
          kernel_reshape,
          strides=strides,
          #padding='VALID'
          padding='SAME'
        )

# to 2 tensor
        remaked_image = tf.reshape(convolution_result, [11, 11])

        print(sess.run(tf.shape(remaked_image)))
        print(sess.run((remaked_image)))


# push newral network
        y = tf.constant([[1.1, 3.2, -0.9]])
# softmax function
        p = tf.nn.softmax(y)
        print(p.eval())

if __name__ == '__main__':
    #rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
    exit()
    #rospy.spin()
