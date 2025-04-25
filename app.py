from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import joblib
import pandas as pd
import numpy as np
from utils.predict import predict_grade, process_input_data, batch_predict

app = Flask(__name__)

# Load the model at startup
MODEL_PATH = 'model.joblib'
model = None #defaults to none incase model loading fails
try:
    with open(MODEL_PATH, 'rb') as f:
        model = joblib.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    """Render the main input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process prediction requests - handles both single and batch predictions"""
    if request.is_json:
        data = request.json
        
        # Check if it's a batch prediction (list of dictionaries)
        if isinstance(data, list):
            # Limit batch size to 100
            if len(data) > 100:
                return jsonify({
                    "error": "Batch size exceeds maximum limit of 100 records"
                }), 400
                
            # Process batch prediction
            try:
                results = batch_predict(data, model)
                return jsonify(results)
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        
        # Single prediction
        else:
            try:
                result = predict_grade(data, model)
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 400
    else:
        # Handle form submission
        try:
            form_data = {k: v for k, v in request.form.items()}
            result = predict_grade(form_data, model)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/example.json')
def example_json():
    """Provide an example JSON array for testing with 2 objects and current feature names"""
    examples = [
        {
            "age": 21,
            "gender": "Male",
            "attendance": 88,
            "midterm_score": 75,
            "final_score": 82,
            "assignments_avg": 78,
            "quizzes_avg": 72,
            "participation_score": 8,
            "projects_score": 85,
            "total_score": 78.8,
            "study_hours_per_week": 10,
            "stress_level": 5,
            "sleep_hours_per_night": 7,
            "department": "CS",
            "extracurricular_activities": "Yes",
            "internet_access_at_home": "Yes",
            "parent_education_level": "Bachelor's",
            "family_income_level": "Medium"
        },
        {
            "age": 19,
            "gender": "Female",
            "attendance": 92,
            "midterm_score": 81,
            "final_score": 77,
            "assignments_avg": 85,
            "quizzes_avg": 88,
            "participation_score": 9,
            "projects_score": 90,
            "total_score": 81.7,
            "study_hours_per_week": 14,
            "stress_level": 4,
            "sleep_hours_per_night": 8,
            "department": "Mathematics",
            "extracurricular_activities": "No",
            "internet_access_at_home": "No",
            "parent_education_level": "Master's",
            "family_income_level": "Low"
        }
    ]
    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
