from tensorflow.keras.models import load_model
import numpy as np
from math import sqrt

model = load_model("models/my_model_v3.h5")

x_train = [[[4,2,4,2],[25,36,49,64],[sqrt(1),sqrt(2),sqrt(3),sqrt(4)], [57,58,59,60], [sqrt(5),sqrt(6),sqrt(7),sqrt(8)], [49,64,81,100], [71, 5, 71, 5], [543,542,541,540],[196,225,256,289], [sqrt(12),sqrt(13),sqrt(14),sqrt(15)],
            [5,6,7,8], [sqrt(17),sqrt(19),sqrt(20),sqrt(21)], [104,51,104,51], [sqrt(11),sqrt(12),sqrt(13),sqrt(14)],[100,101,102,103],[8,7,6,5],[92,300,92,300]]]
y_train = [4,81,sqrt(5),61,sqrt(9),121, 71, 539,324,sqrt(16),9,sqrt(22), 104, sqrt(15),104,4,92]

x_val = [[]]
y_val = []

# Determine the batch size
batch_size = 2

# Calculate steps per epoch
steps_per_epoch = len(x_train) // batch_size

model.fit(x_train.reshape((x_train.shape[0], x_train.shape[1], 1)), y_train,
          epochs=150, verbose=1, validation_data=(x_val.reshape((x_val.shape[0], x_val.shape[1], 1)), y_val),
          steps_per_epoch=steps_per_epoch)

model.save("models/my_model_v3_1.h5")