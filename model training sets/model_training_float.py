#model training for floats

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model
from math import sqrt

# Data
x_train = np.array([
                    [sqrt(1),sqrt(2),sqrt(3),sqrt(4)], [sqrt(5),sqrt(6),sqrt(7),sqrt(8)], [sqrt(12),sqrt(13),sqrt(14),sqrt(15)],
            [sqrt(17),sqrt(19),sqrt(20),sqrt(21)], [sqrt(11),sqrt(12),sqrt(13),sqrt(14)],[sqrt(44),sqrt(45),sqrt(46),sqrt(47)],
            [sqrt(37),sqrt(38),sqrt(39),sqrt(40)],[sqrt(9),sqrt(10),sqrt(11),sqrt(12)],
            [sqrt(5),sqrt(6),sqrt(7),sqrt(8)],[sqrt(4),sqrt(5),sqrt(6),sqrt(7)],
            [sqrt(2),sqrt(3),sqrt(4),sqrt(5)],
            [sqrt(9),sqrt(10),sqrt(12),sqrt(13)]])
y_train = np.array([sqrt(5),sqrt(9),sqrt(16),sqrt(22),sqrt(15),sqrt(48),sqrt(41),sqrt(13),
                    sqrt(9),sqrt(8),sqrt(6),sqrt(14)])

x_val = np.array([
                  [sqrt(4), sqrt(5), sqrt(6), sqrt(7)], [sqrt(71), sqrt(72), sqrt(73), sqrt(74)],
                  [sqrt(16), sqrt(17), sqrt(18), sqrt(19)], [sqrt(2), sqrt(3), sqrt(4), sqrt(5)],
                  [sqrt(25), sqrt(26), sqrt(27), sqrt(28)], [sqrt(80), sqrt(81), sqrt(82), sqrt(83)],
                  [sqrt(36), sqrt(37), sqrt(38), sqrt(39)], [sqrt(50), sqrt(51), sqrt(52), sqrt(53)],
                  [sqrt(1),sqrt(2),sqrt(3),sqrt(4)]])

y_val = np.array([sqrt(8), sqrt(75), sqrt(20), sqrt(6),
                 sqrt(29), sqrt(84), 13, 9, sqrt(40), sqrt(54),sqrt(5)])

# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(4,))) # Input shape is (4,) for your data
model.add(Dense(64, activation='relu'))  
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for predicting a single value

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Determine the batch size
batch_size = 2

# Calculate steps per epoch
steps_per_epoch = len(x_train) // batch_size

# Train the model
model.fit(x_train.reshape((x_train.shape[0], x_train.shape[1], 1)), y_train,
          epochs=150, verbose=1, validation_data=(x_val.reshape((x_val.shape[0], x_val.shape[1], 1)), y_val),
          steps_per_epoch=steps_per_epoch)


model.save("models/model_int_v2.h5")

