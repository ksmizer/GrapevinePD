import os    
os.environ['THEANO_FLAGS'] = "device=gpu"   

import sys
sys.path.insert(0, '../convnets-keras')

from keras import backend as K
from theano import tensor as T
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, Lambda, Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D


def mean_subtract(img):   
    img = T.set_subtensor(img[:,0,:,:],img[:,0,:,:] - 123.68)
    img = T.set_subtensor(img[:,1,:,:],img[:,1,:,:] - 116.779)
    img = T.set_subtensor(img[:,2,:,:],img[:,2,:,:] - 103.939)

    return img / 255.0

def get_alexnet(input_shape,nb_classes,mean_flag): 
	# code adapted from https://github.com/heuritech/convnets-keras

	input_1 = Input(shape=input_shape)

	if mean_flag:
		mean_subtraction = Lambda(mean_subtract, name='mean_subtraction')(input_1)
		conv_1 = Conv2D(96, (11, 11),strides=(4,4),
                        kernel_initializer='he_normal',activation='relu',
                        name='conv_1')(mean_subtraction)
	else:
		conv_1 = Conv2D(96, (11, 11),strides=(4,4),
                        kernel_initializer='he_normal',activation='relu',
                        name='conv_1')(input_1)

	conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
	conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)
	conv_2 = merge([
	    Conv2D(128,(5,5),kernel_initializer='he_normal',activation="relu",
                name='conv_2_'+str(i+1))
                (splittensor(ratio_split=2,id_split=i)(conv_2)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1,1))(conv_3)
	conv_3 = Conv2D(384,(3,3),kernel_initializer='he_normal',activation='relu',
                name='conv_3')(conv_3)

	conv_4 = ZeroPadding2D((1,1))(conv_3)
	conv_4 = merge([
	    Conv2D(192,(3,3),kernel_initializer='he_normal',activation="relu",
                name='conv_4_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_4)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

	conv_5 = ZeroPadding2D((1,1))(conv_4)
	conv_5 = merge([
	    Conv2D(128,(3,3),kernel_initializer='he_normal',activation="relu",
                name='conv_5_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_5)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

	dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)
	dense_1 = Flatten(name="flatten")(dense_1)
	dense_1 = Dense(4096,kernel_initializer='he_normal',activation='relu',
                name='dense_1')(dense_1)

	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(4096,kernel_initializer='he_normal',activation='relu',
                name='dense_2')(dense_2)

	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(nb_classes,kernel_initializer='he_normal',
                name='dense_3_new')(dense_3)

	prediction = Activation("softmax",name="softmax")(dense_3)

	alexnet = Model(inputs=input_1, outputs=prediction)
    
	return alexnet

