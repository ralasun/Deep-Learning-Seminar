#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages
import _pickle as cPickle

# Import external packages
import numpy as np
import tensorflow as tf

# Hyperparameters
len_input = 224*224*3

learning_rate = 0.0001
num_itr = 500
batch_size = 128
display_step = 10

# Define some functions... for whatever purposes
def read_data(fpath):
	with open(fpath, "rb") as fo:
		data_train = cPickle.load(fo, encoding="bytes")
		np.random.shuffle(data_train)
	return data_train

def reformat_params(dict_lyr):
	params_pre = {}
	for key in dict_lyr:
		params_pre[key + "_W"] = tf.Variable(dict_lyr[key]["weights"], name=key + "_W")
		params_pre[key + "_b"] = tf.Variable(dict_lyr[key]["biases"], name=key + "_b")
	return params_pre

def slice_params_module(name_module, params_pre):
	params_module = {}
	keys = [key for key in params_pre]
	for key in keys:
		if name_module in key:
			params_module[key.replace(name_module,"")] = params_pre[key]
	return params_module

def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv2ser(x, params_fc):
	# Serialise
	shape_ftmap_end = x.get_shape()
	len_ftmap_end = int(shape_ftmap_end[1]*shape_ftmap_end[2]*shape_ftmap_end[3])
	print(params_fc.get_shape().as_list()[0])
	return tf.reshape(x, [-1, params_fc.get_shape().as_list()[0]])

def fc1d(x, W, b, bn=True):
	# FC layer wrapper
	fc = tf.add(tf.matmul(x, W), b)
	if bn == True: fc = tf.contrib.layers.batch_norm(fc)
	return tf.nn.relu(fc)
	
def inception_module(tsr_X, name_module, params_pre):
	params_module = slice_params_module(name_module, params_pre)
	
	# convolutions
	inception_1x1 = conv2d(tsr_X, params_module["1x1_W"], params_module["1x1_b"])
	inception_3x3r = conv2d(tsr_X, params_module["3x3_reduce_W"], params_module["3x3_reduce_b"])
	inception_3x3 = conv2d(inception_3x3r, params_module["3x3_W"], params_module["3x3_b"])
	inception_5x5r = conv2d(tsr_X, params_module["5x5_reduce_W"], params_module["5x5_reduce_b"])
	inception_5x5 = conv2d(inception_5x5r, params_module["5x5_W"], params_module["5x5_b"])
	inception_pool_proj = conv2d(tsr_X, params_module["pool_proj_W"], params_module["pool_proj_b"])

	# Concatenation
	inception_concat = tf.concat([inception_1x1, inception_3x3, inception_5x5, inception_pool_proj], axis=-1)
	return inception_concat


def arxt(X, params_pre, params):
	# Wrapper of all
	X_reshaped = tf.reshape(X, shape=[-1, 224, 224, 3])

	# Convolution and max pooling(down-sampling) Layers
	# Convolution parameters are from pretrained data
	conv1_7x7_s2 = conv2d(X_reshaped, params_pre['conv1_7x7_s2_W'], params_pre['conv1_7x7_s2_b'], strides=2)
	conv1_7x7p_s2 = maxpool2d(conv1_7x7_s2, k=2)

	conv2_3x3 = conv2d(conv1_7x7p_s2, params_pre['conv2_3x3_W'], params_pre['conv2_3x3_b'])
	conv2p_3x3 = maxpool2d(conv2_3x3, k=2)

	# Modules
	inception_3a = inception_module(conv2p_3x3, "inception_3a_", params_pre)
	inception_3b = inception_module(inception_3a, "inception_3b_", params_pre)
	inception_3bp = maxpool2d(inception_3b, k=2)

	inception_4a = inception_module(inception_3bp, "inception_4a_", params_pre)
	inception_4b = inception_module(inception_4a, "inception_4b_", params_pre)
	inception_4c = inception_module(inception_4b, "inception_4c_", params_pre)
	inception_4d = inception_module(inception_4c, "inception_4d_", params_pre)
	inception_4e = inception_module(inception_4d, "inception_4e_", params_pre)
	inception_4ep = maxpool2d(inception_4e, k=2)

	inception_5a = inception_module(inception_4ep, "inception_5a_", params_pre)
	inception_5b = inception_module(inception_5a, "inception_5b_", params_pre)
	inception_5ap = tf.nn.avg_pool(inception_5b, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')

	inception_5ap_1d = conv2ser(inception_5ap, params["fc6_W"])
	fc6 = fc1d(inception_5ap_1d, params["fc6_W"], params["fc6_b"])
	fc7 = fc1d(fc6, params["fc7_W"], params["fc7_b"])
	return fc1d(fc7, params["fc8_W"], params["fc8_b"])


dict_lyr = np.load("/home/fdalab/Desktop/KU_Project/X_Raycode_by_sanghoon/googlenet.npy", encoding = 'latin1').item()
params_pre = reformat_params(dict_lyr)

data_saved = {'save_point': tf.Variable(0)}
params = {
	# len_ftmap_end = int(shape_ftmap_end[1]*shape_ftmap_end[2]*shape_ftmap_end[3])
	'fc6_W': tf.Variable(tf.random_normal([1024, 4096]), name='fc6_W'),
	'fc6_b': tf.Variable(tf.random_normal([4096]), name='fc6_b'),

	'fc7_W': tf.Variable(tf.random_normal([4096, 4096]), name='fc7_W'),
	'fc7_b': tf.Variable(tf.random_normal([4096]), name='fc7_b'),
	
	'fc8_W': tf.Variable(tf.random_normal([4096, 2]), name='fc8_W'),
	'fc8_b': tf.Variable(tf.random_normal([2]), name='fc8_b'),
}

# Graph placeholders
X = tf.placeholder(tf.float32, [None, len_input])
y = tf.placeholder(tf.float32, [None, 2])
pred = arxt(X, params_pre, params)

def feed_dict(data, batch_size):
	batch = data[np.random.choice(data.shape[0], size=batch_size,  replace=True)]
	print(batch.shape)
	return {X: batch[:,:len_input], y: batch[:,len_input:]}

# BUILDING THE COMPUTATIONAL GRAPH


# Evaluation modules
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(crossEntropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Integrate tf summaries
tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# Load pretrained data
saver = tf.train.Saver()

# RUNNING THE COMPUTATIONAL GRAPH
with tf.Session() as sess:
	data_train = read_data("/home/fdalab/Desktop/KU_Project/X_Raycode_by_sanghoon/pickledata_train.pickle")
	data_test = read_data("/home/fdalab/Desktop/KU_Project/X_Raycode_by_sanghoon/pickledata_train.pickle")

	summaries_dir = './logs'
	train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(summaries_dir + '/test')
	
	# Initialise the variables and run
	init = tf.global_variables_initializer()
	sess.run(init)
	
	# with tf.device("/cpu:0"):
	with tf.device("/gpu:0"):
		# For train
		try:
			saver.restore(sess, './modelckpt/inception.ckpt')
			print('Model restored')
			epoch_saved = data_saved['save_point'].eval()
		except tf.errors.NotFoundError:
			print('No saved model found')
			epoch_saved = 1
		except tf.errors.InvalidArgumentError:
			print('Model structure has change. Rebuild model')
			epoch_saved = 1

		# Training cycle
		for epoch in range(epoch_saved, epoch_saved + num_itr + 1):
			# Run optimization op (backprop)
			summary, acc_train, loss_train, _ = sess.run([merged, accuracy, cost, optimizer], feed_dict=feed_dict(data_train, batch_size))
			train_writer.add_summary(summary, epoch)
			
			summary, acc_test = sess.run([merged, accuracy], feed_dict=feed_dict(data_test, batch_size))
			test_writer.add_summary(summary, epoch)
			print("Accuracy at step {0}: {1}".format(epoch, acc_test))

			if epoch % display_step == 0:
				print("Epoch {0}, Minibatch Loss= {1:.6f}, Train Accuracy= {2:.5f}".format(epoch, loss_train, acc_train))
	
		print("Optimisation Finished!")