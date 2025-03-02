import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist #importing the dataset images
(x_train, y_train), (x_test, y_test) = mnist.load_data() #splitting into training and testing data
# x is the values and y is the digit
x_train = tf.keras.utils.normalize(x_train, axis=1) #the values of the x in training and testing data are made between 0-1
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #turns grid dataset into one linear dataset
model.add(tf.keras.layers.Dense(128,activation='relu')) #adding the hidden layers
model.add(tf.keras.layers.Dense(128,activation='relu')) #128 is the number of neurons in the layer
model.add(tf.keras.layers.Dense(10,activation='softmax')) #softmax make sure that the value (probability) of all the 10 neurons add upto one

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3) #training the model, epochs is the number of times the model is going through same data

model.save('handwritten.model.keras')