import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import x_train, y_train

print(np.size(x_train,0))
print(np.size(y_train))
p=0
k=0
j=0
l = 0
for i in y_train:
    if i == "polynomial":
        p=p+1
    if i == "exponential":
        k=k+1
    if i == "sine":
        j=j+1
    if i == "ln":
        l=l+1
print(f'Exponentials: {k}, Poly: {p}, Sine: {j}, Natural Log: {l}')
exit()


# Encode labels
label_map = {"ln": 0, "polynomial": 1, "exponential": 2, "sine": 3}
y_train_encoded = np.array([label_map[label[0]] for label in y_train])


# Reshape x_train to fit the model input
x_train_reshaped = x_train.reshape((len(y_train), 6, 2))

# Split data into training and validation sets
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_reshaped, y_train_encoded, test_size=0.2, random_state=22) # 10 is good

# l=0
# p=0
# e=0
# s=0

# for i in y_val_split:
#     if i == 0:
#         l = l + 1
#     if i == 1:
#         p=p+1
#     if i == 2:
#         e=e+1
#     if i == 3:
#         s=s+1

# print(f"Logs: {l}, Poly: {p}, Exp: {e}, Sine: {s}")

# exit()

# Define the model with some changes
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(6, 2)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
history = model.fit(x_train_reshaped, y_train_encoded, 
                    validation_data=(x_val_split, y_val_split),
                    epochs=1000, batch_size=32,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate model performance on validation data
val_loss, val_acc = model.evaluate(x_val_split, y_val_split)
print(f"Validation accuracy: {val_acc}")

# Save the model
model.save("C:/Users/Administrator/func pred/function_prediction/models/model_V1_8.h5")
