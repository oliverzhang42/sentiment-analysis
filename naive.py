import h5py
import numpy as np

def to_one_hot(vec, classes):
    return np.eye(classes)[vec]

x = np.load("bert_train.npy", allow_pickle=True)
y = np.load("train_labels.npy")
y = to_one_hot(y, 5)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, LeakyReLU

model_lstm = Sequential()
model_lstm.add(LSTM(256, input_dim=300, input_length=200, return_sequences=True))
model_lstm.add(LeakyReLU())
model_lstm.add(LSTM(256))
model_lstm.add(LeakyReLU())
model_lstm.add(Dense(256, activation = 'relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(5, activation = 'softmax'))

opt = keras.optimizers.Adam(lr=0.001)
model_lstm.compile(opt, loss="categorical_crossentropy", metrics=['accuracy'])

train = x[0:20000]
y_train = y[0:20000]

val = x[20000:]
y_val = y[20000:]

model_lstm.fit(train, y_train, validation_data=(val, y_val), epochs=3)

model_lstm.save_weights("model_file")


#guesses = model_list.predict(test)
np.save("guesses", guesses)
