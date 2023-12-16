from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import datasets as tfd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd #for saving results

def scheduler(epoch, lr):
   if epoch < 10:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

N_CLASSES = 20
INPUT_SIZE = (32, 32, 1)
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
CALLBACKS = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3), tf.keras.callbacks.LearningRateScheduler(scheduler)] # Correct?
BATCH_SIZE = 32
EPOCHS = 15

train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
(x_train, y_train), (x_test, y_test) = train_data, test_data
x_train, x_test = np.mean(x_train, axis=3), np.mean(x_test, axis=3)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=N_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=N_CLASSES)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# normalize input data
x_train_n = (x_train.astype('float32')) / 255
x_val_n = (x_val.astype('float32')) / 255

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

    h0 = tf.keras.layers.Conv2D(filters=4, kernel_size=(1,1))(input_img)
    # h0 = tf.keras.layers.MaxPool2D((1,1))(h0) #this does nothing

    h1 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding='same')(h0)
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

    h0 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1))(input_img)
    h0 = tf.keras.layers.MaxPool2D((1,1))(h0) #this does nothing

    h1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(5,5), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=5, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), padding='same')(h2)
    h3 = tf.keras.layers.MaxPool2D((2,2))(h3)

    h4 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), padding='same')(h3)
    h4 = tf.keras.layers.MaxPool2D((2,2))(h4)

    fh1 = tf.keras.layers.Flatten()(h4)
    fh2 = tf.keras.layers.Dense(units=128, activation='relu')(fh1)
    fh3 = tf.keras.layers.Dense(units=64, activation='relu')(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

def M4():
    global y_pred

    h0 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5))(input_img)
    
 
    h1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.AvgPool2D((3,3))(h1)

    h2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')(h2)
    h3 = tf.keras.layers.MaxPool2D((2,2))(h3)


    fh1 = tf.keras.layers.Flatten()(h3)
    fh2 = tf.keras.layers.Dense(units=128, activation='relu')(fh1)
    fh3 = tf.keras.layers.Dense(units=64, activation='relu')(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)


def M5(l1_value, l2_value):
    global y_pred
    # Double number of filters on each iteration - generic first layers, specific later layers 
    reg = tf.keras.regularizers.L1L2(l1=l1_value, l2=l2_value)
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=reg)(input_img)
    h0 = tf.keras.layers.MaxPool2D((2, 2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=reg)(h0)
    h1 = tf.keras.layers.MaxPool2D((2, 2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=reg)(h1)
    h2 = tf.keras.layers.MaxPool2D((2, 2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=reg)(fh1)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=reg)(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)




#l1, l2 parameters - this model gets 0.412 accuracy on normal training of 10 epochs
M5(0, 0)
model_name = 'M5-regularized'
#added model name to save statistics and keep track of it at least a bit - convenient for converting to latex later too - can delete for submission
model_name = 'M5'
model = tf.keras.Model(input_img, y_pred)
model.summary()

sampler = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                                          width_shift_range=0.1, height_shift_range=0.1).flow(x_train_n, y_train, batch_size=BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
history = model.fit(sampler, epochs=EPOCHS, steps_per_epoch=TRAIN_SIZE//BATCH_SIZE, callbacks=CALLBACKS, validation_data=(x_val_n, y_val))

#csv QoL
history_df = pd.DataFrame(history.history)
history_df.to_csv(f'{model_name}_training_history.csv', index=False)

test_loss, test_acc = model.evaluate(x_val_n, y_val)

#csv QoL
evaluation_df = pd.DataFrame({'Test Loss': [test_loss], 'Test Accuracy': [test_acc]})
evaluation_df.to_csv(f'{model_name}_evaluation_results.csv', index=False)



