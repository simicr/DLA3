from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import datasets as tfd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle


N_CLASSES = 20
INPUT_SIZE = (32, 32, 1)
LOSS = tf.losses.CategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']
CALLBACKS = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)] # Correct?
BATCH_SIZE = 32
EPOCHS = 10

train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
(x_train, y_train), (x_test, y_test) = train_data, test_data
x_train, x_test = np.mean(x_train, axis=3), np.mean(x_test, axis=3)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=N_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=N_CLASSES)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

TRAIN_SIZE, _ , _ , _ = x_train.shape
INPUT_IMG = tf.keras.Input(shape=INPUT_SIZE)
y_pred = None





def M1():
    # CNN from the practicals, only for testing if the code works.
    global y_pred

    h1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(INPUT_IMG)
    h1 = tf.keras.layers.MaxPooling2D((2, 2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="valid")(h1)
    h2 = tf.keras.layers.MaxPooling2D((2, 2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="valid")(h2)
    h3 = tf.keras.layers.MaxPooling2D((2, 2))(h3)

    h4 = tf.keras.layers.Flatten()(h3)
    h5 = tf.keras.layers.Dense(units=100, activation="relu")(h4)
    h6 = tf.keras.layers.Dense(units=20)(h5)
    
    y_pred = tf.keras.layers.Dense(units=N_CLASSES)(h6)


M1()
model = tf.keras.Model(INPUT_IMG, y_pred)
model.summary()

# Data augmentation prob, needs changing, this is taken from the practicals. 
sampler = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                                          width_shift_range=0.1, height_shift_range=0.1).flow(x_train, y_train, batch_size=BATCH_SIZE)


model.compile(optimizer=tf.keras.optimizers.Adam(), loss=LOSS, metrics=['accuracy'])
model.fit(sampler, epochs=EPOCHS, steps_per_epoch=TRAIN_SIZE//BATCH_SIZE, callbacks=CALLBACKS, validation_data=(x_val, y_val))
test_loss, test_acc = model.evaluate(x_val, y_val)



