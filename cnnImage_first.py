import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
import cv2
import numpy as np



# pip install pillow
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf




class RosTensorFlow():
    def __init__(self):

        a = 1
    def main(self):
        sess = tf.InteractiveSession()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        mnist = input_data.read_data_sets('MNIST_data/')
# reshape first image to 28*28 pixel 
        image_matrix = tf.reshape(mnist.train.images[0], [28, 28])
# convert to 8 bits mutrix (seisuu)

        image_matrix_uint8 = tf.cast(255 * image_matrix, tf.uint8)

# show label
        print(mnist.train.labels[0])
# create gray scale image
        Image.fromarray(image_matrix_uint8.eval(), 'L')

#####################################################################
#Random number seed fixed to ensure reproducibility (any number is acceptable)
        tf.set_random_seed(12345)
# Download and reload mnist data in mnist data Directory
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# input image 
        x = tf.placeholder(tf.float32, name='x')
        print(x)
# convert size
        x_1 = tf.reshape(x, [-1, 28, 28, 1])
        print(x_1)
# tatamikomi
# random kernel 

        k_0 = tf.Variable(tf.truncated_normal([4, 4, 1, 10], mean=0.0, stddev=0.1))
#Convolution
        x_2 = tf.nn.conv2d(x_1, k_0, strides=[1, 3, 3, 1], padding='VALID')
# kasseika function
        x_3 = tf.nn.relu(x_2)
# pooling
        x_4 = tf.nn.max_pool(x_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
# convert size
        x_5 = tf.reshape(x_4, [-1, 160])
# zenketugou
# weight and bias
        w_1 = tf.Variable(tf.zeros([160, 40]))
        b_1 = tf.Variable([0.1] * 40)
# zenketugou
        x_6 = tf.matmul(x_5, w_1) + b_1
# kakkeika func
        x_7 = tf.nn.relu(x_6)
# zenketugou
# weight and bias
        w_2 = tf.Variable(tf.zeros([40, 10]))
        b_2 = tf.Variable([0.1] * 10)
# zenketugou
        x_8 = tf.matmul(x_7, w_2) + b_2
# kakuritu ka
        y = tf.nn.softmax(x_8)
# loss function for minimum
# answer label
        labels = tf.placeholder(tf.float32, name='labels')
# loss function and adam function
        loss = -tf.reduce_sum(labels * tf.log(y))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
# seido kensyou
        prediction_match = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction_match, tf.float32), name='accuracy')

# paramer
#batisize
        BATCH_SIZE = 32
# train count
        NUM_TRAIN = 10000
#Output frequency during training
        OUTPUT_BY = 500

# run training
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        for i in range(NUM_TRAIN):
          batch = mnist.train.next_batch(BATCH_SIZE)
          inout = {x: batch[0], labels: batch[1]}
          if i % OUTPUT_BY == 0:
            train_accuracy = accuracy.eval(feed_dict=inout)
            print('step {:d}, accuracy {:.2f}'.format(i, train_accuracy))
          optimizer.run(feed_dict=inout)

#Accuracy verification by test data
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, labels: mnist.test.labels})
        print('test accuracy {:.2f}'.format(test_accuracy))
        
if __name__ == '__main__':
    #rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
    exit()
    #rospy.spin()
