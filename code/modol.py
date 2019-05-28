from keras.optimizers import Adam
from keras import Model, Input
from keras.layers import *
import tensorflow as tf
import time
import os
#hyper things
train_epochs = 10
batch_size = 10
learning_rate = 0.01
test_ratio = 0.1

#the picture size
imginput = Input(shape=(100, 100, 3))

#imgtube1
imgtube1 = Conv2D(filters=32, kernel_size=3, padding='same',
                  activation='relu')(imginput)
imgtube1 = BatchNormalization()(imgtube1)
imgtube1 = AveragePooling2D(padding="valid",
                            pool_size=2,
                            data_format="channels_last",
                            strides=1)(imgtube1)
imgtube1 = Flatten()(imgtube1)

#imgtube2
imgtube2 = Conv2D(filters=32, kernel_size=5, padding='same',
                  activation='relu')(imginput)
imgtube2 = BatchNormalization()(imgtube2)
imgtube2 = Conv2D(filters=32, kernel_size=3, padding='same',
                  activation='relu')(imgtube2)
imgtube2 = BatchNormalization()(imgtube2)
imgtube2 = Conv2D(filters=32, kernel_size=5, padding='same',
                  activation='relu')(imgtube2)
imgtube2 = BatchNormalization()(imgtube2)
imgtube2 = AveragePooling2D(padding="valid",
                            pool_size=2,
                            data_format="channels_last",
                            strides=1)(imgtube2)
imgtube2 = Flatten()(imgtube2)

#imgtube3
imgtube3 = Conv2D(filters=32, kernel_size=5, padding='same',
                  activation='relu')(imginput)
imgtube3 = BatchNormalization()(imgtube3)
imgtube3 = Conv2D(filters=32, kernel_size=5, padding='same',
                  activation='relu')(imgtube3)
imgtube3 = BatchNormalization()(imgtube3)
imgtube3 = Conv2D(filters=32, kernel_size=3, padding='same',
                  activation='relu')(imgtube3)
imgtube3 = BatchNormalization()(imgtube3)
imgtube3 = Conv2D(filters=32, kernel_size=3, padding='same',
                  activation='relu')(imgtube3)
imgtube3 = BatchNormalization()(imgtube3)
imgtube3 = Conv2D(filters=32, kernel_size=1, padding='same',
                  activation='relu')(imgtube3)
imgtube3 = BatchNormalization()(imgtube3)
imgtube3 = AveragePooling2D(padding="valid",
                            pool_size=2,
                            data_format="channels_last",
                            strides=1)(imgtube3)
imgtube3 = Flatten()(imgtube3)

#combine network
comnet = concatenate([imgtube1, imgtube2, imgtube3])
comnet = Dense(64, activation='relu')(comnet)
output = Dense(3, activation='softmax')(comnet)

#difine the modol
model = Model(inputs=[imginput], outputs=[output])
model.summary()

model.compile(optimizer=Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#TODO get the data , train the modol
