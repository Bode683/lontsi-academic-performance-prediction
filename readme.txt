# Student Performance Prediction System

A web application that uses machine learning to predict student grades based on various academic and socio-economic factors. This system provides a REST API for predictions and a simple web interface for user interaction.

## Features

- Predict student grades (A-F) based on input features
- REST API for programmatic access
- Web interface for manual data entry
- Support for both single and batch predictions (up to 100 records)
- Confidence scores for each grade prediction
- JSON file upload capability

## Tech Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Deployment**: Ready for PythonAnywhere, Heroku, or AWS

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Local Setup

1. Clone the repository (or download and extract the ZIP file)

2. Create and activate a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Generate a dummy model (for testing purposes)
   ```bash
   python generate_model.py
   ```

5. Start the Flask application
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://localhost:5000`

### Deployment

#### PythonAnywhere

1. Sign up for a PythonAnywhere account
2. Create a new web app (choose Flask)
3. Upload your code or clone from Git
4. Install requirements using the PythonAnywhere console
5. Configure the WSGI file to point to your `app.py`

#### Heroku

1. Create a `Procfile` with the following content:
   ```
   web: gunicorn app:app
   ```

2. Deploy using the Heroku CLI:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

## API Documentation

### Predict Endpoint

**URL**: `/predict`
**Method**: `POST`
**Content-Type**: `application/json`

#### Request Format

Single prediction:
```json
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
  "study_hours_per_week": 10,
  "extracurricular_activities": "Yes",
  "internet_access_at_home": "Yes",
  "parent_education_level": "Bachelor's",
  "family_income_level": "Medium",
  "stress_level": 5,
  "sleep_hours_per_night": 7
}
```

Batch prediction (array of student records):
```json
[
  {
    "age": 21,
    "gender": "Male",
    // other fields...
  },
  {
    "age": 19,
    "gender": "Female",
    // other fields...
  }
]
```

#### Response Format

Single prediction:
```json
{
  "predicted_grade": "B",
  "confidence": {
    "A": 0.10,
    "B": 0.72,
    "C": 0.12,
    "D": 0.05,
    "F": 0.01
  }
}
```

Batch prediction:
```json
[
  {
    "predicted_grade": "B",
    "confidence": {/* ... */}
  },
  {
    "predicted_grade": "A",
    "confidence": {/* ... */}
  }
]
```

## Project Structure

```
project/
├── app.py                # Flask entry point
├── model.pkl             # Trained ML model
├── generate_model.py     # Script to generate a dummy model
├── templates/
│   └── index.html        # Form interface
├── static/
│   ├── style.css         # UI styles
│   └── script.js         # Client-side functionality
├── utils/
│   └── predict.py        # Feature processing and prediction logic
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Future Enhancements

- User authentication system
- Integration with learning management systems (LMS)
- Feature importance visualization
- Admin dashboard for model monitoring
- Support for uploading and retraining models
- Student feedback tracking

## License

MIT License
