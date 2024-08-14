import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import x_train_degree, y_train_degree

print(np.size(x_train_degree,0))
print(np.size(y_train_degree))
p=0
k=0
j=0
l=0
b=0
for i in y_train_degree:
    if i == "2":
        p=p+1
    if i == "1":
        k=k+1
    if i == "3":
        j=j+1
    if i == "4":
        l=l+1
    if i == "5":
        b=b+1
        
print(f'Number of FIRST degrees: {k}, Number of SECOND degrees: {p}, Number of THIRD degrees: {j}, Number of FOURTH degrees: {l}, Number of FIFTH degrees: {b}')
exit()


# Encode labels
label_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
y_train_encoded = np.array([label_map[label[0]] for label in y_train_degree])


# Reshape x_train to fit the model input
x_train_reshaped = x_train_degree.reshape((len(y_train_degree), 6, 2))

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
    tf.keras.layers.Dense(5, activation='softmax')
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
                    epochs=150, batch_size=32,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate model performance on validation data
val_loss, val_acc = model.evaluate(x_val_split, y_val_split)
print(f"Validation accuracy: {val_acc}")

# Save the model
model.save("C:/Users/Administrator/func pred/function_prediction/models/model_degree_V1.h5")
