# app.py

from flask import Flask, send_from_directory, jsonify
from main_for_website import run_main

app = Flask(__name__)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')  # Serve the HTML file

@app.route('/process', methods=['GET'])
def process():
    try:
        result = run_main()  # Call the function from main.py
        return jsonify(result=result)  # Return result as JSON response
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
