import os 
import tensorflow as tf 
from PIL import Image  
import numpy as np

def fetData(filename):

    with tf.Session() as sess:
    
        filename_queue = tf.train.string_input_producer([filename]) #读入流中
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        #filename_queue = tf.train.string_input_producer([filename_queue])#生成一个queue队列

         #sess.run()

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)#返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'img_raw' : tf.FixedLenFeature([], tf.string),
                                        })#将image数据和label取出来
 
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        label = features['label']
        img = tf.reshape(img, [100, 100, 3])/255  #reshape为128*128的3通道图片
    
    
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)

        imagelist=[]
        labellist=[]

        for i in range(115):
            init_op = tf.initialize_all_variables()
            sess.run(init_op)

            imagelist.append(img)
            labellist.append(sess.run(label))

            # print(img.shape)
            # print(label)

        #print(img)
        #img = tf.cast(img, tf.float32) * (1. / 255)#在流中抛出img张量
        # label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
        # img,label=sess.run(img,label)
        #print(img.shape)
        #print(label)
        coord.request_stop()
        coord.join(threads)
    return imagelist,labellist


if __name__=="__main__":
    img,label=fetData("tfrecord_train.tfrecords")
    