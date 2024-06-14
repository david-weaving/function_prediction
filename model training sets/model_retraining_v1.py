
# for retraining the model

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from math import sqrt

model = load_model("models/my_model_v3.h5")

x_train = np.array([])
y_train = np.array([])

x_val = np.array([])
y_val = np.array([])

# Determine the batch size
batch_size = 2

# Calculate steps per epoch
steps_per_epoch = len(x_train) // batch_size

model.fit(x_train.reshape((x_train.shape[0], x_train.shape[1], 1)), y_train,
          epochs=125, verbose=1, validation_data=(x_val.reshape((x_val.shape[0], x_val.shape[1], 1)), y_val),
          steps_per_epoch=steps_per_epoch)

model.save("models/model_float_v1.h5")