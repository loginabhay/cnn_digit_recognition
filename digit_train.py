from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape, y_train.shape , y_test.shape)
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.figure()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(np.argmax(np.round(x_train[0])))
x_train = np.array(x_train).reshape(-1,28,28,1).astype('float32')
x_test = np.array(x_test).reshape(-1,28,28,1).astype('float32')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save('digit.h5py')
