import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -------------------------------------THIS FUNCTION ROUNDS THE PREDICTION TO THE NEAREST WHOLE INTEGER--------------------------------------

# note: change the function to return floating point numbers when the data recieved from the user is also a floating point number

def make_custom_prediction(model, input_data):             
    
    input_data = np.array(input_data).reshape((1, len(input_data), 1))
    
    prediction = model.predict(input_data)
    
    prediction_rounded = int(np.round(prediction))
    
    return prediction_rounded

# ---------------------------------------------------------------------------

#def make_custom_prediction(model, input_data):
    
    #input_data = np.array(input_data).reshape((1, len(input_data), 1))
    
    #prediction = model.predict(input_data)
    
    #return prediction[0][0]  # Return the raw predicted value


model = load_model("C:/Users/Administrator/func pred/function_prediction/models/my_model_v3.h5")

x = 0
original_points = a,b,c,d= 1,2,1,2 # these are the points the user sends in, given that the first point is 1
data_plot = [a,b,c,d]
input_data = [a,b,c,d]
x_values = [1,2,3,4]

while x != 50:

    data_plot.append(make_custom_prediction(model,input_data))
    input_data = [data_plot[x+1], data_plot[x+2], data_plot[x+3], data_plot[x+4]]
    x = x + 1
    x_values.append(x+4)

    
print(data_plot) #predicted data
x_values_original = [1,2,3,4]

#user input graph
plt.plot(x_values_original, original_points, marker='o', linestyle='-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Graph')
plt.show()

#predicted graph
plt.plot(x_values, data_plot, marker='o', linestyle='-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted Graph')
plt.show()
