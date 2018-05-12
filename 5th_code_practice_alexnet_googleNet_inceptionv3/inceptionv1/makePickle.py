#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os, subprocess, sys, shutil
import _pickle as cPickle
import pickle as pickle

# Import 3rd party packages
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

# Import built-in packages

# Define constants (mostly paths of directories)
DIR_DATA_LABELLED_C00_TRAIN = "/home/fdalab/Desktop/dl_cv_tensorflow_10weeks-master/week2/inceptionv1/x_ray/abnormal/"
DIR_DATA_LABELLED_CNN_TRAIN = "/home/fdalab/Desktop/dl_cv_tensorflow_10weeks-master/week2/inceptionv1/x_ray/normal/"
DIR_DATA_PICKLE = "./"

len_h = 224#*4
len_w = 224#*4
len_c = 3
n_class = 2 # Normal & abnormal

def dir_to_pickle(dir_src, resolution, vec_class):
	len_h, len_w, len_c = resolution

	seq_fpath = os.listdir(dir_src)

	seq_rec = np.zeros(shape=(len(seq_fpath), len_h*len_w*len_c + n_class), dtype=np.float32)
	print(seq_rec.shape)
	for i, fpath in enumerate(seq_fpath):
		print(fpath)
		img = Image.open(dir_src + fpath).convert('RGB')
		# img = raw_img.convert('RGB')
		size_img = img.size
		
		min_side = min(size_img)
		padding_h, padding_v= (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
		
		img_crop = img.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
		img_resize = img_crop.resize((len_h, len_w))

		arr_img = np.asarray(img_resize)
		arr1d_img = arr_img.reshape(len_h*len_w*len_c)
		seq_rec[i] = np.append(arr1d_img, vec_class) # [1, 0] for normal

	return seq_rec

seq_rec_train_C00 = dir_to_pickle(DIR_DATA_LABELLED_C00_TRAIN, (len_h, len_w, len_c), [1, 0])
seq_rec_train_CNN = dir_to_pickle(DIR_DATA_LABELLED_CNN_TRAIN, (len_h, len_w, len_c), [0, 1])
seq_rec_train = np.concatenate([seq_rec_train_C00, seq_rec_train_CNN])



print(seq_rec_train_C00.shape)
print(seq_rec_train_CNN.shape)
print(seq_rec_train.shape)
print(sys.getsizeof(seq_rec_train))
print(seq_rec_test_C00.shape)
print(seq_rec_test_CNN.shape)
print(seq_rec_test.shape)
print(sys.getsizeof(seq_rec_test))

with open(DIR_DATA_PICKLE + "data_train.pickle", 'wb') as handle:
    pickle.dump(seq_rec_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(DIR_DATA_PICKLE + "data_test.pickle", 'wb') as handle:
    pickle.dump(seq_rec_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
