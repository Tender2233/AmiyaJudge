import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import to_categorical

def parse_image_example(serial_example):
    features = tf.parse_single_example(
        serial_example,
        features={
            'img_raw' : tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        }
    )
    return features['img_raw'], features['label']


def getdata(filename):
    print("start reading")
    dataset = tf.data.TFRecordDataset(filename).map(parse_image_example)
    iterator = dataset.make_one_shot_iterator()
    img,label = iterator.get_next()
    image = tf.decode_raw(img, tf.uint8)
    label = tf.cast(label, tf.int32)
    x_data = []
    y_data = []
    #image = tf.decode_raw(img_raw, tf.uint8)
    i=0
    print("start while")
    with tf.Session() as sess:
        while 1:
            try:
                i+=1
                print(i)
                # _x_data, _y_data = sess.run(next_element)
                # _x_data=_x_data.reshape(100,100,1)
                
                
                #print(sess.run(image))
                # image,label=sess.run(image,label)
                # image=sess.run(image)
                x_data.append(sess.run(image))
                y_data.append(to_categorical(sess.run(label), 3))
            except tf.errors.OutOfRangeError:
                break


    x_train_data = np.array(x_data).reshape(-1,100,100,1)/255
    y_train_data = np.array(y_data)
    

    return x_train_data, y_train_data


if __name__ == '__main__':
    x_train_data, y_train_data=getdata("tfrecord_train.tfrecords")
    print(x_train_data)