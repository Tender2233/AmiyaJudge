from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import os
from datasetreader import *

model = load_model('amiya.h5')
x_data, y_data=getdata("tfrecord_train.tfrecords")
model.evaluate(x_data, y_data, batch_size=128)