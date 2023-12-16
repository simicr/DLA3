from sklearn.model_selection import train_test_split
from keras import datasets as tfd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def scheduler(epoch, lr):
   if epoch < 10:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

# Define constants
N_CLASSES = 20
INPUT_SIZE = (32, 32, 1)
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
CALLBACKS = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), tf.keras.callbacks.LearningRateScheduler(scheduler)]
BATCH_SIZE = 32
EPOCHS = 20

# Preparing the data
train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
(x_train, y_train), (x_test, y_test) = train_data, test_data
x_train, x_test = np.mean(x_train, axis=3), np.mean(x_test, axis=3)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=N_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=N_CLASSES)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)


# Define some more variables.
TRAIN_SIZE, _ , _ , _ = x_train.shape
input_img = tf.keras.Input(shape=INPUT_SIZE)
y_pred = None


#
# TASKA A
#
def task_a():
    fig, axs = plt.subplots(4, 5)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow(x_train[i])
        print(y_train[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


#
# TASK B
#

def M1():
    global y_pred

    h1 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    h1 = tf.keras.layers.MaxPooling2D((2, 2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='valid')(h1)
    h2 = tf.keras.layers.MaxPooling2D((2, 2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(h2)
    h3 = tf.keras.layers.MaxPooling2D((2, 2))(h3)

    fh1 = tf.keras.layers.Flatten()(h3)
    fh2 = tf.keras.layers.Dense(units=128, activation='relu')(fh1)
    fh3 = tf.keras.layers.Dense(units=32, activation='relu')(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

def M2():
    global y_pred

    h0 = tf.keras.layers.Conv2D(filters=4, kernel_size=(1,1))(input_img)
    h0 = tf.keras.layers.MaxPool2D((1,1))(h0)

    h1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(h1)
    h2 = tf.keras.layers.MaxPooling2D((2, 2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(h2)
    h3 = tf.keras.layers.MaxPooling2D((2, 2))(h3)

    h4 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(h3)
    h4 = tf.keras.layers.MaxPool2D((2,2))(h3)

    fh1 = tf.keras.layers.Flatten()(h4)
    fh2 = tf.keras.layers.Dense(units=128, activation='relu')(fh1)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu')(fh2)
    fh4 = tf.keras.layers.Dense(units=32, activation='relu')(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh4)

def M3():
    global y_pred

    h0 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1,1))(input_img)
    h0 = tf.keras.layers.MaxPool2D((1,1))(h0) 

    h1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(5,5), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='same')(h2)
    h3 = tf.keras.layers.MaxPool2D((2,2))(h3)

    h4 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='same')(h3)
    h4 = tf.keras.layers.MaxPool2D((2,2))(h4)

    fh1 = tf.keras.layers.Flatten()(h4)
    fh2 = tf.keras.layers.Dense(units=128, activation='relu')(fh1)
    fh3 = tf.keras.layers.Dense(units=64, activation='relu')(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)


def M5():
    global y_pred
    #double number of filters on each iteration - generic first layers, specific later layers 
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)



    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu')(fh1)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu')(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

#
# TASK C
#
    
# Adding L2 reg with lambda=0.1 on the FC part
def C1():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh1)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu',  kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

# Adding L2 reg with lambda=0.1 on the FC part
def C2():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.1))(fh1)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu',  kernel_regularizer=tf.keras.regularizers.L2(0.1))(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

# Adding L2 reg with lambda=0.001 in the FC part
def C3():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(fh1)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu',  kernel_regularizer=tf.keras.regularizers.L2(0.001))(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

# Adding dropout with rate 0.5 in the FC part.
def C4():
    global y_pred
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu')(fh1)
    fh2 = tf.keras.layers.Dropout(0.5)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu')(fh2)
    fh3 = tf.keras.layers.Dropout(0.5)(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

# Adding dropout with rate 0.1 in the FC part.
def C5():
    global y_pred
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu')(fh1)
    fh2 = tf.keras.layers.Dropout(0.1)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu')(fh2)
    fh3 = tf.keras.layers.Dropout(0.1)(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)


#very sparse dropout layer with two regularizations, about same performance as c2 and c3 but for some reason reaches it way faster than 
#c2 and is slighlty more accurate on training data than c3
def C6():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(fh1)
    fh2 = tf.keras.layers.Dropout(0.01)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu',  kernel_regularizer=tf.keras.regularizers.L2(0.001))(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)



#very shit, too agressive model, stays at same. I thought single regularization might give good results when combined with dropout
def C7():
    global y_pred
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(fh1)
    fh2 = tf.keras.layers.Dropout(0.1)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu')(fh2)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)
#
# TASK E
#

sampler = tf.keras.preprocessing.image.ImageDataGenerator().flow(x_train, y_train, batch_size=BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()    
C7()
model_name = 'C7'

model = tf.keras.Model(input_img, y_pred)
model.summary()


model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
history = model.fit(sampler, epochs=EPOCHS, steps_per_epoch=TRAIN_SIZE//BATCH_SIZE, callbacks=CALLBACKS, validation_data=(x_val, y_val))
test_loss, test_acc = model.evaluate(x_val, y_val)

# Write down results in a csv file.
history_df = pd.DataFrame(history.history)
history_df.to_csv(f'{model_name}_training_history.csv', index=False)
evaluation_df = pd.DataFrame({'Test Loss': [test_loss], 'Test Accuracy': [test_acc]})
evaluation_df.to_csv(f'{model_name}_evaluation_results.csv', index=False)



