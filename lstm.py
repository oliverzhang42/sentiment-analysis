import h5py
import numpy as np
from keras.utils import Sequence

class BertSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)


def to_one_hot(vec, classes):
    return np.eye(classes)[vec]

filename = 'bert_train.h5'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    x = list(f[a_group_key])

y = np.load("train_labels.npy")
y = to_one_hot(y, 5)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, LeakyReLU

model_lstm = Sequential()
model_lstm.add(LSTM(256, input_dim=768, input_length=150, return_sequences=True))

model_lstm.add(LeakyReLU())
model_lstm.add(LSTM(256))
model_lstm.add(LeakyReLU())
model_lstm.add(Dense(256, activation = 'relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(5, activation = 'softmax'))

opt = keras.optimizers.Adam(lr=0.001)
model_lstm.compile(opt, loss="categorical_crossentropy", metrics=['accuracy'])

train = BertSequence(x[0:20000], y[0:20000], 64)
val = BertSequence(x[20000:], y[20000:], 64)

model_lstm.fit_generator(train, validation_data=val, epochs=3)
model_lstm.save_weights("model_file")

