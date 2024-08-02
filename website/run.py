from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import func_web

app = Flask(__name__)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        x = data['x']
        y = data['y']

        predicted_function = func_web.predict_function(x, y)

        x_common, y_fit, e_function = [], [], ""

        if predicted_function == "polynomial":
            x_common, y_fit, e_function = func_web.poly_average(x, y, 3)  # example degree
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

if __name__ == '__main__':
    app.run(debug=True)

