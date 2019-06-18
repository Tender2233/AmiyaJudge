import os 
import tensorflow as tf 
from PIL import Image 
import numpy as np


 #将图片转化为TFrecord格式二进制文件


cwd='Original/picture/'
classes={'donkey','horse','scorpion'} #人为 设定 3 类
writer= tf.python_io.TFRecordWriter("tfrecord_train.tfrecords") #要生成的文件


if __name__=="__main__":
    
    for index,name in enumerate(classes):
        class_path=cwd+name+'/'
        class_save='Original/'+name+'/'
        for img_name in os.listdir(class_path): 
            img_path=class_path+img_name #每一个图片的地址
            img_save=class_save+img_name
            # img = tf.gfile.FastGFile(img_path,"rb").read()
            # img=tf.image.resize_images(img,(100,100),0)

            img=Image.open(img_path)
            img= img.resize((100,100))
            img=img.convert('1')
            img.save(img_save)
          
