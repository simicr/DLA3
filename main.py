from sklearn.model_selection import train_test_split
from keras import datasets as tfd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os


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

dict = pickle.load(open("cifar20_perturb_test.pkl", "rb"))
x_perturb, y_perturb = dict['x_perturb'], dict['y_perturb']
x_perturb = np.mean(x_perturb, axis=3)
x_perturb = np.expand_dims(x_perturb, axis=-1)
y_perturb = tf.keras.utils.to_categorical(y_perturb, num_classes=N_CLASSES)


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

    class_counts_train = np.sum(y_train, axis=0)
    class_counts_val = np.sum(y_train, axis=0)
    class_proportions_train = class_counts_train / len(y_train)
    class_proportions_val = class_counts_val / len(y_val)

    # Print the proportions for each class
    for class_label, proportion in enumerate(class_proportions_train):
        print(f"Class {class_label}: Proportion - {proportion:.4f}")
    for class_label, proportion in enumerate(class_proportions_val):
        print(f"Class {class_label}: Proportion - {proportion:.4f}")


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


# Addinng dropout with 0.1 and L2 reg with lambda = 0.001 
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
    fh2 = tf.keras.layers.Dropout(0.1)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu',  kernel_regularizer=tf.keras.regularizers.L2(0.001))(fh2)
    fh3 = tf.keras.layers.Dropout(0.1)(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

# Addinng dropout with 0.1 and L2 reg with lambda = 0.01
def C7():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh1)
    fh2 = tf.keras.layers.Dropout(0.1)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh2)
    fh3 = tf.keras.layers.Dropout(0.1)(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

# Adding dropout with 0.1 with L2 reg with lambda 0.1
def C8():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.1))(fh1)
    fh2 = tf.keras.layers.Dropout(0.1)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.1))(fh2)
    fh3 = tf.keras.layers.Dropout(0.1)(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)

#
# TASK D
# 
    
x_full, y_full = train_data
x_full = np.mean(x_full, axis=3)
x_full = np.expand_dims(x_full, axis=-1) 
y_full = tf.keras.utils.to_categorical(y_full, num_classes=N_CLASSES)


#
# TASK E - C7 seems to be the best performing model so the models used here stem from it
#
    
#no batch norm - same as C7 Actually - the idea is just to test it for data augmentation
def E1():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh1)
    fh2 = tf.keras.layers.Dropout(0.1)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh2)
    fh3 = tf.keras.layers.Dropout(0.1)(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)





#batch norm on all layers 
def E2():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.BatchNormalization()(h0)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh1)
    fh2 = tf.keras.layers.BatchNormalization()(fh2)
    fh2 = tf.keras.layers.Dropout(0.1)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh2)
    fh3 = tf.keras.layers.BatchNormalization()(fh3)
    fh3 = tf.keras.layers.Dropout(0.1)(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)



#batch norm on dense layers only
def E3():
    global y_pred
    h0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(input_img)
    h0 = tf.keras.layers.MaxPool2D((2,2))(h0)

    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')(h0)
    h1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(h1)

    h2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    fh1 = tf.keras.layers.Flatten()(h2)
    fh2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh1)
    fh2 = tf.keras.layers.BatchNormalization()(fh2)
    fh2 = tf.keras.layers.Dropout(0.1)(fh2)
    fh3 = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(fh2)
    fh3 = tf.keras.layers.BatchNormalization()(fh3)
    fh3 = tf.keras.layers.Dropout(0.1)(fh3)
    y_pred = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(fh3)




# function to train and reload model - utilitity - does the same thing as before but more convenienty
#if you wanna test on the perturbation dataset, set perturbation to true
def train_and_evaluate_model(y_pred, sampler, model_name, retrain=False, perturbation=False):
    model_save_path = f"results/{model_name}_model.h5"
    history_csv_path = f'results/{model_name}_training_history.csv'
    evaluation_csv_path = f'results/{model_name}_evaluation_results.csv'
    perturbation_csv_path = f'results/{model_name}_perturbed_results.csv'

    if not retrain and os.path.exists(model_save_path):
        # Load the existing model
        model = tf.keras.models.load_model(model_save_path)
        print(f"Loaded pre-trained model '{model_name}'.")

    else:
        model = tf.keras.Model(input_img, y_pred)
        print(model_name)

        optimizer = tf.keras.optimizers.Adam()

        model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
        history = model.fit(sampler, epochs=EPOCHS, steps_per_epoch=TRAIN_SIZE // BATCH_SIZE, callbacks=CALLBACKS, validation_data=(x_val, y_val))
        
        model.save(model_save_path)


    test_loss, test_acc = model.evaluate(x_val, y_val)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv_path, index=False)

    evaluation_df = pd.DataFrame({'Test Loss': [test_loss], 'Test Accuracy': [test_acc]})
    evaluation_df.to_csv(evaluation_csv_path, index=False)

    if perturbation:
        perturbation_loss, perturbation_acc = model.evaluate(x_perturb, y_perturb)
        perturbation_df = pd.DataFrame({'Perturbation Loss': [perturbation_loss], 'Perturbation Accuracy': [perturbation_acc]})
        perturbation_df.to_csv(perturbation_csv_path, index=False)

    

    return model

data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=0, horizontal_flip=True, vertical_flip=True,
                                                          width_shift_range=0.1, height_shift_range=0.1,
                                                          validation_split=0.2, fill_mode='reflect')

# model.summary()
sampler = tf.keras.preprocessing.image.ImageDataGenerator().flow(x_train, y_train, batch_size=BATCH_SIZE)
sampler_augmented = data_augmenter.flow(x_train, y_train, batch_size=BATCH_SIZE)

C7()
#please if you use data augmentation, say so in the model name and swap out the the sampler
model_name = 'D1-Full'
train_and_evaluate_model(y_pred, sampler, model_name, perturbation=True)

