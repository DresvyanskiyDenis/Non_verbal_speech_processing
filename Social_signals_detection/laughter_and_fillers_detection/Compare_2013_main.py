import os

import numpy as np
import pandas as pd
from scipy.io import wavfile
import tensorflow as tf

from Social_signals_detection.laughter_and_fillers_detection.utils import Database

def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, filters=64, kernel_size=10, strides=1, activation='relu',
                                     padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.AvgPool1D(pool_size=2))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='softmax')))
    print(model.summary())

    return model

if __name__ == "__main__":
    # params
    path_to_labels = 'C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\ComParE_2013_Vocalization\\ComParE2013_Voc\\lab\\train.mlf'
    path_to_data = 'C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\ComParE_2013_Vocalization\\ComParE2013_Voc\\wav\\'
    window_size=6
    window_step=3
    database=Database(path_to_data, path_to_labels)
    database.load_all_data_and_labels()
    database.cut_all_instances(window_size, window_step)
    train_data, train_labels=database.get_all_concatenated_cutted_data_and_labels()
    # permutate training data
    permutation=np.random.permutation(train_data.shape[0])
    train_data, train_labels= train_data[permutation], train_labels[permutation]
    # transform categorical labels to probabilistic (with keras)
    # create for data one additional dimension
    train_data=train_data[..., np.newaxis]
    train_labels=tf.keras.utils.to_categorical(train_labels)
    # create and configure the model
    input_shape=(train_data.shape[1], 1)
    model=create_model(input_shape)
    model.summary()
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    model.fit(train_data, train_labels)


