import tensorflow as tf
import random
import numpy as np
from scipy import misc
import os

checkpoint_file = "/tmp/model.ckpt"

#sess = tf.InteractiveSession()

imgInput = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape, varName):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=varName)

def bias_variable(shape, varName):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=varName)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([3, 3, 3, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")

h_conv1 = tf.nn.relu(conv2d(imgInput, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], "b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2], "W_fc2")
b_fc2 = bias_variable([2], "b_fc2")

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

#gets 50 random faces and formats them into a matrix
def get_batch(sampleSize):
	images = []
	labels = []
	will_dir = "../Project/will"
	other_dir = "../Project/other"
	will_images = os.listdir(will_dir)
	other_images = os.listdir(other_dir)
	willRandNums = random.sample(range(len(will_images)), sampleSize)
	otherRandNums = random.sample(range(len(other_images)), sampleSize)
	for i in range(sampleSize):
		if (np.random.random() < 0.5):
			images.append(np.array(misc.imread(other_dir + "/" + other_images[otherRandNums[i]])).reshape(28, 28, 3))
			labels.append([1, 0])
		else:
			images.append(np.array(misc.imread(will_dir + "/" + will_images[willRandNums[i]])).reshape(28, 28, 3))
			labels.append([0,1])

	return [np.array(images).reshape(-1, 28, 28, 3), np.array(labels).reshape(-1, 2)]

def get_test():
	images = []
	labels = []
	test_dir = "../Project/test"
	test_images = os.listdir(test_dir)
	
	for i in range(len(test_images)):
		images.append(np.array(misc.imread(test_dir + "/" + test_images[i])).reshape(28, 28, 3))
		if str(test_images[i][0]) == "w":
			labels.append([0, 1])
		else:
			labels.append([1, 0])

	return [np.array(images).reshape(-1, 28, 28, 3), np.array(labels).reshape(-1, 2)]


def train(num_iters, train_batch_size):
	test_images, test_labels = get_test()
	for i in range(num_iters):
		images, labels = get_batch(train_batch_size)
		if i % 2 == 0:
			train_accuracy = accuracy.eval(feed_dict={imgInput: test_images, y_: test_labels, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={imgInput: images, y_: labels, keep_prob: 0.5})

saver = tf.train.Saver()
print("Saving weights...")
with tf.Session() as sess:
	sess.run(init_op)

	#Train the model
	train(1000, 50)

	save_path = saver.save(sess, checkpoint_file)
	print("Model saved in file: %s" % save_path)
