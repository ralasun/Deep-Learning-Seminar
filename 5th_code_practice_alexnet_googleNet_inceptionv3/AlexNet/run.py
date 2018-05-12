import tensorflow as tf 
import pickle
import numpy as np
import os
from datetime import datetime
import argparse
import urllib.request
import tarfile
import sys
from pprint import pprint

NCLASS=10
DATA_URL="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def maybe_download_or_extract(model_dir):
	"""
	downloading cifar-10 via data_url
	args: 
		- model_dir : final destination directory of tarfile"""

	dest_dir=model_dir
	if not os.path.exists(dest_dir):
		os.makedirs(dest_dir)
	filename=DATA_URL.split('/')[-1]
	filepath=os.path.join(dest_dir, filename)
	if not os.path.exists(filepath):

		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>>Downloading %s %.1f%%'
				%(filename, float(count*block_size)/float(total_size)*100))
			sys.stdout.flush()

		filepath, _=urllib.request.urlretrieve(DATA_URL, filename, _progress)
		statinfo=os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	members=tarfile.open(filepath, 'r:gz').getnames()
	sub_dir=os.path.join(model_dir,members[0])
	# print(sub_dir)
	sub_filenames=[os.path.join(sub_dir, filename) for filename in os.listdir(sub_dir)]
	# pprint(sub_filenames)
	#make directory of train data and test data
	train_dir=os.path.join(model_dir, 'train')
	test_dir=os.path.join(model_dir, 'test')

	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
	train_file_glob=os.path.join(sub_dir, 'data_batch_*')
	train_lists=tf.gfile.Glob(train_file_glob)
	
	for old_train_path in train_lists:
		new_train_path=os.path.join(train_dir, os.path.basename(old_train_path))
		if not os.path.exists(new_train_path):
			tf.gfile.Copy(old_train_path, new_train_path)

	if not os.path.exists(test_dir):
		os.makedirs(test_dir)
	test_file_glob=os.path.join(sub_dir, 'test_batch')

	old_test_path=test_file_glob
	new_test_path=os.path.join(test_dir, os.path.basename(new_train_path))
	if not os.path.exists(new_test_path):
		tf.gfile.Copy(old_test_path, new_test_path)

	print('Successfully forming the directory of train and test')

	return train_dir, test_dir



def one_hot_label(pre_label, indices):
	"""
	basically, cifar-10 model does not privde one-hot encoded label. 
	so, we should perform one-hot encoding on label from cifar-10 in advance.
	args:
		- pre_label : label basically provided by cifar-10. 
					  label that should be one-hot encoded.
		- indices : numpy array. np.array([0,1,2,3,4,5,6,7,8,9]) 
		            the number of elements in indices should be equal to the number of classes.
	"""
	one_label=np.zeros((1,NCLASS))
	index=np.where(indices==pre_label)[0][0]
	one_label[:,index]=1.0
	return one_label


def load_batch_data(batch_path):
	"""
	from cifar-10 page, data is provided as form of pickle file.
	so, we should unpickle individual train and test file using this function.
	args:
		- batch_path : each individually data_batch pickle file to be unpickled.
	"""
	with open(batch_path, 'rb') as fo:
		batch_data=pickle.load(fo, encoding='bytes')
	data=batch_data[b'data']
	labels=batch_data[b'labels']

	indices=np.arange(NCLASS)
	one_labels=np.zeros([data.shape[0], NCLASS])
	for i,label in enumerate(labels):
		one_label=one_hot_label(label, indices)
		one_labels[i]=one_label

	return data, one_labels


def load_data(batch_dir):
	"""
	load all the train data and test data
	args:
		- batch_dir : the upper directory of test and train directory 
	"""
	batch_lists=os.listdir(batch_dir)
	batch_paths=[os.path.join(batch_dir, batch_list) for batch_list in batch_lists]
	
	if len(batch_paths) > 1:
		full_data=[]
		full_labels=[]
		for batch_path in batch_paths:
			data, one_labels=load_batch_data(batch_path)
			full_data.append(data)
			full_labels.append(one_labels)

		full_data=np.asarray(full_data).reshape(-1, 32*32*3)
		full_labels=np.asarray(full_labels).reshape(-1, 10)

		return full_data, full_labels
	elif len(batch_paths) == 1:
		data, labels=load_batch_data(batch_paths[0])
		return data, labels

def next_batch(data, one_labels, train_batch_size):	
	"""
	form the batch set of the data, which will be fed into Session part.
	args:
		data - train or test data
		one_labels - one-hot encoded labels
		train_batch_size - hyperparameter. batch size
	"""
	index_in_batch=np.random.choice(data.shape[0], size=train_batch_size, replace=False)
	X_batch=data[index_in_batch]
	Y_batch=one_labels[index_in_batch]
	return X_batch, Y_batch

def conv2d_with_norm(inputs, filter_size, num_filters, prefix, stride=1, padding='SAME', regularize=True):
	"""
		function of convolution, relu, and local response normalization, which is the partial layer of alexnet.
		alexnet consists of 5 convolutional layers and 3 fully connected layers. And also, 5 convolutional layers
		consist of 2 local response layers and 3 non-local response layers. This formulates the local-response layer.
	args:
		inputs - input of function
		filter_size - size of receptive field
		num_filters - the number of filter
		prefix - the name of this function, which is the function os sub-functions of this function.
		stride - stride
		regularize - if True, weights are rugularized with l2-norm. if False, weights are not regularized.
	"""
	he_init=tf.contrib.layers.variance_scaling_initializer() # he initialization
	if regularize==True:
		w_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)
	elif regularize==False:
		w_regularizer=None

	strides=(stride, stride)

	conv=tf.layers.conv2d(inputs, num_filters, filter_size, strides, padding, activation=tf.nn.relu, 
							kernel_initializer=he_init, 
							kernel_regularizer=w_regularizer, 
							name=prefix+'_conv')
	local_norm=tf.nn.local_response_normalization(conv, name=prefix+'_lrn')
	return local_norm

def conv2d(inputs, filter_size, num_filters, prefix, stride=1, padding='SAME', regularize=True):
	"""
	this consists of non-local response layer. the remainder intrunction is same as the function right above. 
	Please refer to that function.
	"""	
	he_init=tf.contrib.layers.variance_scaling_initializer()
	if regularize==True:
		w_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)
	elif regularize==False:
		w_regularizer=None

	strides=(stride, stride)
	conv=tf.layers.conv2d(inputs, num_filters, filter_size, strides, padding, activation=tf.nn.relu,
							kernel_initializer=he_init, 
							kernel_regularizer=w_regularizer,
							name=prefix+'_conv')
	return conv

def maxpool(inputs, pool_size, strides, prefix, padding='SAME'):
	"""
	maxpooling layer.
	args
		inputs - input of maxpooling layer
		pool_size - receptive field of pooling layer
		strides - stride
		prefix - name of this maxpooling layer
		padding - if 'SAME', output tensor has the same size of the input tensor. if 'valid', its not.
	"""
	maxpool=tf.layers.max_pooling2d(inputs, pool_size, strides, padding, name=prefix+'_maxpool')
	return maxpool


def AlexNet():
	"""
	This is the alexNet model part. The most of the construction follows the model suggested by the alexnet paper.
	However, instead of constructing the 2 local response layers, add an one more local response layer. 
	In conclusion, the overall model consists of 3 local response layers and 2 non-local response layers( both are 
	convolutional layers) and the rest of 2 layers are fully-connected layers.
	"""
	X=tf.placeholder(dtype=tf.float32, shape=[None, 32*32*3], name='X') #placeholder if input data
	X_reshaped=tf.reshape(X, shape=[-1, 32, 32, 3])
	keep_prob=0.7 # drop out probability in fully-connected layer

	with tf.name_scope('conv1') as scope:
		conv_n_norm1=conv2d_with_norm(X_reshaped, 5, 64, 'conv1', stride=1)
		pool1=maxpool(conv_n_norm1,3, 2,'conv1')

	with tf.name_scope('conv2') as scope:
		conv_n_norm2=conv2d_with_norm(pool1, 5, 64, 'conv2', stride=1)
		pool2=maxpool(conv_n_norm2, 3, 2, 'conv2')

	with tf.name_scope('conv3') as scope:
		conv3=conv2d(pool2, 3, 128, 'conv3')

	with tf.name_scope('conv4') as scope:
		conv4=conv2d(conv3, 3, 128,'conv4')
	with tf.name_scope('conv5') as scope:
		conv_n_norm5=conv2d_with_norm(conv4, 3, 128, 'conv5')
		pool3=maxpool(conv_n_norm5, 3, 2,'conv5')

	_, w, h, chs=pool3.get_shape().as_list()
	pool3_reshaped=tf.reshape(pool3, [-1, w*h*chs])

	with tf.name_scope('fully') as scope:
		flc1=tf.layers.dense(pool3_reshaped, 384, activation=tf.nn.relu, 
							kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='flc1')
		flc1=tf.nn.dropout(flc1, keep_prob)

		flc2=tf.layers.dense(flc1, 192, activation=tf.nn.relu, 
							kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='flc2')
		flc2=tf.nn.dropout(flc2, keep_prob)

	with tf.name_scope('output') as scope:
		logits=tf.layers.dense(flc2, units=10, name='logits')

	print(logits.get_shape())
	return X, logits

def training_op(logits, learning_rate):
	### training part.. ###
	Y=tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')

	with tf.name_scope('cross_entropy'):
		cross_entropy=tf.nn.softmax_cross_entropy_with_logits(
			labels=Y, logits=logits)
		cross_entropy_mean=tf.reduce_mean(cross_entropy)
	tf.summary.scalar('cross_entropy', cross_entropy_mean)

	with tf.name_scope('train'):
		optimizer=tf.train.AdamOptimizer(learning_rate)
		train_step=optimizer.minimize(cross_entropy_mean)

	return train_step, cross_entropy_mean, logits, Y


def evaluation_op(result_tensor, ground_truth_tensor):
	"""evaluation part"""
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			prediction=tf.argmax(result_tensor, 1)
			correct_prediction=tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))
		with tf.name_scope('accuracy'):
			accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	return  accuracy, prediction

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for Tensorboard visualization)."""

	with tf.name_scope('summaries'):
		mean=tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.summary.scalar('stddev'):
			stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_mean(var))
		tf.summary.histogram('histogram', var)

def train(model_dir):

	train_dir, test_dir=maybe_download_or_extract(model_dir)

	learning_rate=0.001
	training_steps=1000
	batch_size=256
	display_step=5
	##loading the data
	train_data, train_one_labels=load_data(train_dir)
	test_data, test_one_labels=load_data(test_dir)

	#Building the computational graph
	X, logits=AlexNet()
	train_step, cross_entropy_mean, logits, Y=training_op(logits, learning_rate)
	accuracy, prediction=evaluation_op(logits, Y)

	with tf.Session() as sess:

		summaries_dir='./tf_logs'

		if tf.gfile.Exists(summaries_dir):
			tf.gfile.DeleteRecursively(summaries_dir)
		tf.gfile.MakeDirs(summaries_dir)

		merged=tf.summary.merge_all()
		train_writer=tf.summary.FileWriter(summaries_dir+'/train', tf.get_default_graph())
		test_writer=tf.summary.FileWriter(summaries_dir+'/test')

		init=tf.global_variables_initializer()
		sess.run(init)

		step=0
		for epoch in range(training_steps):

			for num in range(int(train_data.shape[0]/batch_size)):

				X_batch, Y_batch=next_batch(train_data, train_one_labels, batch_size)
				_, train_summary, train_accuracy, train_loss=sess.run([train_step, merged, accuracy, 
															cross_entropy_mean], 
															feed_dict={X:X_batch, Y:Y_batch})
				train_writer.add_summary(train_summary, step)

				X_test_batch, Y_test_batch=next_batch(test_data, test_one_labels, batch_size)
				test_summary, test_accuracy=sess.run([merged, accuracy], feed_dict={X:X_test_batch, Y:Y_test_batch})
				test_writer.add_summary(test_summary, step)

				step+=1

				if step % display_step == 0:
					print('Epoch %d: Step %d: Train accuracy=%1.f%%, Train loss=%1.3f' %(epoch, step, train_accuracy*100, train_loss))
					print('Epoch %d: Step %d: Test accuracy=%1.f%%' %(epoch, step, test_accuracy*100))

		print('Optimization is finished!')






	

if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument(
		'--model_dir',
		type=str,
		default='',
		help="""directory of test data and train data""")
	args=parser.parse_args()
	model_dir=args.model_dir
	train(model_dir)