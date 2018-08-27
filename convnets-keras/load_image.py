from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import scipy.ndimage
import json

from alexnet_base import *
from utils import *
from datetime import datetime

batch_size = 16
input_size = (3,227,227)
nb_classes = 6
mean_flag = True # if False, then the mean subtraction layer is not prepended

#code ported from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

alexnet = get_alexnet(input_size,nb_classes,mean_flag)
alexnet.load_weights('weights/alexnet_weights.h5', by_name=True)

# uncomment to view model summary
# print alexnet.summary()

# use predict?
# example|
#   filename = sys.argv[1]
#   img_height, img_width = 150, 150
#   channels = 3
#   img = image.load_img(filename, target_size=(img_height, img_width))
#   x = image.img_to_array(img)
#   x = np.expand_dims(x, axis=0)
# grabbed from https://github.com/aidiary/keras-examples/blob/master/vgg16/dogs_vs_cats/predict.py 
# - need to load in image or access users directory & images
# - example only gives predicted probability of each instance belonging to one
# class, need to research into how to give probs for all classes.
# - probably need to create a "master" program to run at all times and create
# threads to call instances of this function for each user. Maybe a queue system

_folder = "../Data/Users/Test/Untested/"

# predicting images
#height, width, channels = scipy.ndimage.imread(_folder+'test1.jpg').shape
height, width = 227, 227
img = image.load_img(_folder+'test1.jpg', target_size=(width, height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = alexnet.predict(images)
print classes

# predicting multiple images at once
img = image.load_img(_folder+'test2.jpg', target_size=(width, height))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = alexnet.predict(images)

# print the classes, the images belong to
print classes
print classes[0]
print classes[0][0]

# generate json file
cust={'cust': 001, 'name': 'Test Name', 'time': datetime.now().time(), 'classes': [classes]}

print cust
