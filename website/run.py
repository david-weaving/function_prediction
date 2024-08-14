import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

from flask import Flask, request, jsonify, send_from_directory
import func_web
import tensorflow as tf

# Ensure TensorFlow does not use any GPU devices
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

model = tf.keras.models.load_model("C:/Users/Administrator/func pred/function_prediction/models/model_V1_8.h5")
model_degree =tf.keras.models.load_model("C:/Users/Administrator/func pred/function_prediction/models/model_degree_V1.h5")

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

@app.route('/info')
def serve_info():
    return send_from_directory('.', 'info.html')

@app.route('/index')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/user_predict')
def serve_user():
    return send_from_directory('.', 'user_predict.html')


@app.route('/predict_poly', methods=['POST'])
def poly_p():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        degree = int(data.get('degree', 3))  # Default to degree 3 if not provided
        x_common, y_fit, e_function = func_web.poly_average(x, y, degree)

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "function": e_function
            }
        })
    except Exception as e:
        print(f"Error in process: {e}")
        return jsonify({"error": str(e)})


@app.route('/predict_sine', methods=['POST'])
def sine_p():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        x_common, y_fit, e_function = func_web.sine_average(x, y)

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "function": e_function
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_ln', methods=['POST'])
def ln_p():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        x_common, y_fit, e_function = func_web.ln_average(x, y)

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "function": e_function
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_exp', methods=['POST'])
def exp_p():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        x_common, y_fit, e_function = func_web.exp_average(x, y)

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "function": e_function
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        x = data['x']
        y = data['y']
        predicted_function = func_web.predict_function(x, y, model)
        x_common, y_fit, e_function = [], [], ""

        if predicted_function == "polynomial":
            degree = func_web.predict_degree(x,y, model_degree)
            x_common, y_fit, e_function = func_web.poly_average(x, y, degree)  # example degree
        elif predicted_function == "sine":
            x_common, y_fit, e_function = func_web.sine_average(x, y)
        elif predicted_function == "exponential":
            x_common, y_fit, e_function = func_web.exp_average(x, y)
        elif predicted_function == "ln":
            x_common, y_fit, e_function = func_web.ln_average(x, y)
        else:
            return jsonify({"error": "Unknown function type"})

        return jsonify({
            "result": {
                "x_common": x_common,
                "y_fit": y_fit,
                "predicted_function": predicted_function,
                "function": e_function
            }
        })
    except Exception as e:
        print(f"Error in process: {e}")
        return jsonify({"error": str(e)})

# Ensure this block is commented out or removed for production
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
