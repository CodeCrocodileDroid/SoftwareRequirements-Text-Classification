# app.py
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open('requirement_classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # ... prediction logic ...
    return jsonify({'type': predicted_type})

if __name__ == '__main__':
    app.run(debug=True)