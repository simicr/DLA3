from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import datasets as tfd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle


N_CLASSES = 20
INPUT_SIZE = (32, 32, 1)
LOSS = 'categorical_crossentropy'
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
input_img = tf.keras.Input(shape=INPUT_SIZE)
y_pred = None

def task1():
    fig, axs = plt.subplots(4, 5)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow(x_train[i])
        print(y_train[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def M1():
    global y_pred

    h1 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    h1 = tf.keras.layers.MaxPooling2D((2, 2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='valid')(h1)
    h2 = tf.keras.layers.MaxPooling2D((2, 2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(h2)
    h3 = tf.keras.layers.MaxPooling2D((2, 2))(h3)

    fh1 = tf.keras.layers.Flatten()(h3)
    fh2 = tf.keras.layers.Dense(units=128, activation='relu')(fh1)
    fh3 = tf.keras.layers.Dense(units=32, activation='relu')(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

def M2():
    global y_pred

    h1 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding='same')(input_img)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(h1)
    h2 = tf.keras.layers.MaxPooling2D((2, 2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(h2)
    h3 = tf.keras.layers.MaxPooling2D((2, 2))(h3)

    h4 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding='same')(h3)
    h4 = tf.keras.layers.MaxPool2D((2,2))(h3)

    fh1 = tf.keras.layers.Flatten()(h4)
    fh2 = tf.keras.layers.Dense(units=128, activation='relu')(fh1)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu')(fh2)
    fh4 = tf.keras.layers.Dense(units=32, activation='relu')(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh4)

def M3():
    global y_pred




M2()
model = tf.keras.Model(input_img, y_pred)
model.summary()

sampler = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                                          width_shift_range=0.1, height_shift_range=0.1).flow(x_train, y_train, batch_size=BATCH_SIZE)


model.compile(optimizer=tf.keras.optimizers.Adam(), loss=LOSS, metrics=['accuracy'])
model.fit(sampler, epochs=EPOCHS, steps_per_epoch=TRAIN_SIZE//BATCH_SIZE, callbacks=CALLBACKS, validation_data=(x_val, y_val))
test_loss, test_acc = model.evaluate(x_val, y_val)



