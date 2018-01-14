import tensorflow as tf
import numpy as np
import io
import picamera
import numpy
import cv2
from scipy import misc

camera = picamera.PiCamera()
camera.hflip = True
camera.vflip = True
camera.resolution = (320, 240)
face_cascade = cv2.CascadeClassifier("home/pi/Desktop/Projects/haarcascade_frontalface_default.xml")

checkpoint_file = "/home/pi/Desktop/Projects/saves/model.ckpt"
test_dir = "/home/pi/Desktop/Projects/Phase3"

imgInput = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, varName):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=varName)

def bias_variable(shape, varName):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=varName)

W_conv1 = weight_variable([3, 3, 3, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")

W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")

W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], "b_fc1")

W_fc2 = weight_variable([1024, 2], "W_fc2")
b_fc2 = bias_variable([2], "b_fc2")

h_conv1 = tf.nn.relu(conv2d(imgInput, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

def detect_face():
	stream = io.BytesIO()

	camera.capture(stream, format="jpeg")

	buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

	image = cv2.imdecode(buff, 1)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("saves/testface.jpeg", gray)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)

	print("Found " + str(len(faces)) + " face(s)")
	print(str(faces))
	
	cropped_image = 0;

	if len(faces) == 1:
		cropped_image = image[faces[0][1]: faces[0][1] + faces[0][3], faces[0][0]: faces[0][0] + faces[0][2]]
	
	return cropped_image

saver = tf.train.Saver()

with tf.Session() as sess:
	#saver.restore(sess, checkpoint_file)
	#cropped_image = 0
	#i = 0
	#while(cropped_image == 0 and i < 1):
	cropped_image = detect_face()
		#i = i + 1

	if (cropped_image != 0):
		test_image = np.array(cropped_image, np.float32)
		#print("length = " + len(cropped_image))
		test_image = test_image.reshape(-1, 28, 28, 3)
	
		prediction = sess.run(y_conv, feed_dict={imgInput: test_image, keep_prob: 1.0})
		print(prediction)
		print("will = [0, 1], other = [1, 0]")
		print("Model restored.")

