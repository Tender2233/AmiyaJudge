import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
#import matplotlib.pyplot as plt
import numpy as np

# #将TFrecord文件转化为JPG格式图片


def getdata():
    x_data = []
    y_data = []
    filename_queue = tf.train.string_input_producer(
        ["tfrecord_train.tfrecords"])  #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label':
                                           tf.FixedLenFeature([], tf.int64),
                                           'img_raw':
                                           tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [100, 100, 3])
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:  #开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(265):
            example, l = sess.run([image, label])  #在会话中取出image和label
            # img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
            # img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
            print(example, l)
            x_data.append(example)
            y_data.append(l)
        coord.request_stop()
        coord.join(threads)
        y_data = sess.run(tf.one_hot(y_data, 3))
        return x_data, y_data


if __name__ == '__main__':
    x, y = getdata()
    print(x.shape)
    print(y.shape)