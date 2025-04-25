import pandas as pd
import numpy as np
import logging
import sys
import joblib

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def process_input_data(data):
    """
    Process the input data to prepare it for the model.
    - Convert categorical values to appropriate numeric representation
    - Ensure all required fields are present
    - Handle type conversions
    - Map input feature names to match those used during model training

    Args:
        data (dict): Input data dictionary with student features

    Returns:
        pandas.DataFrame: Processed data ready for model prediction
    """
    required_fields = [
        'age', 'gender', 'attendance', 'midterm_score', 'final_score',
        'assignments_avg', 'quizzes_avg', 'participation_score', 'projects_score',
        'study_hours_per_week', 'extracurricular_activities', 'internet_access_at_home',
        'parent_education_level', 'family_income_level', 'stress_level', 'sleep_hours_per_night'
    ]

    # Check for missing fields
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Create a copy to avoid modifying the original
    processed = data.copy()

    # Convert numeric fields (handle string inputs from forms)
    numeric_fields = [
        'age', 'attendance', 'midterm_score', 'final_score', 'assignments_avg',
        'quizzes_avg', 'participation_score', 'projects_score', 'study_hours_per_week',
        'stress_level', 'sleep_hours_per_night'
    ]

    for field in numeric_fields:
        try:
            processed[field] = float(processed[field])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {field}: {processed[field]}. Must be a number.")

    # Create DataFrame with the right order of features
    df = pd.DataFrame([processed])

    # Map numeric feature names to match the model's expected input features
    numeric_feature_mapping = {
        'age': 'Age',
        'attendance': 'Attendance (%)',
        'midterm_score': 'Midterm_Score',
        'final_score': 'Final_Score',
        'assignments_avg': 'Assignments_Avg',
        'quizzes_avg': 'Quizzes_Avg',
        'participation_score': 'Participation_Score',
        'projects_score': 'Projects_Score',
        'study_hours_per_week': 'Study_Hours_per_Week',
        'stress_level': 'Stress_Level (1-10)',
        'sleep_hours_per_night': 'Sleep_Hours_per_Night'
    }

    # Calculate Total_Score based on the input data (weighted average)
    # This ensures we have the Total_Score feature that was present during training
    total_score = (float(processed['midterm_score']) * 0.3 +
                  float(processed['final_score']) * 0.4 +
                  float(processed['assignments_avg']) * 0.15 +
                  float(processed['quizzes_avg']) * 0.1 +
                  float(processed['projects_score']) * 0.05)

    processed['total_score'] = total_score
    numeric_feature_mapping['total_score'] = 'Total_Score'

    # Rename the numeric columns
    df = df.rename(columns=numeric_feature_mapping)

    # Ensure Total_Score is in the dataframe
    if 'Total_Score' not in df.columns:
        df['Total_Score'] = total_score

    # Create binary encoded columns for categorical variables as they were during training

    # Gender (Male=1, Female=0)
    gender_value = 1 if str(processed['gender']).lower() in ['male', 'm', '1'] else 0
    df['Gender_Male'] = gender_value

    # Department - create one-hot encoded columns as in training
    # Default all to 0
    df['Department_CS'] = 0
    df['Department_Engineering'] = 0
    df['Department_Mathematics'] = 0

    # Set the appropriate department to 1 based on input
    department = str(processed.get('department', 'CS')).lower()
    if 'cs' in department or 'computer' in department:
        df['Department_CS'] = 1
    elif 'eng' in department or 'engineering' in department:
        df['Department_Engineering'] = 1
    elif 'math' in department or 'mathematics' in department:
        df['Department_Mathematics'] = 1
    else:
        # Default to CS if no match
        df['Department_CS'] = 1

    # Extracurricular Activities (Yes=1, No=0)
    extracurricular = 1 if str(processed['extracurricular_activities']).lower() in ['yes', 'y', 'true', '1'] else 0
    df['Extracurricular_Activities_Yes'] = extracurricular

    # Internet Access (Yes=1, No=0)
    internet_access = 1 if str(processed['internet_access_at_home']).lower() in ['yes', 'y', 'true', '1'] else 0
    df['Internet_Access_at_Home_Yes'] = internet_access

    # Parent Education Level
    education_levels = ['High School', "Master's", 'PhD']
    education = str(processed['parent_education_level']).lower()

    for level in education_levels:
        column_name = f'Parent_Education_Level_{level}'
        df[column_name] = 0

    # Set the appropriate education level to 1
    if 'high' in education or 'secondary' in education:
        df['Parent_Education_Level_High School'] = 1
    elif 'master' in education:
        df["Parent_Education_Level_Master's"] = 1
    elif 'phd' in education or 'doctorate' in education:
        df['Parent_Education_Level_PhD'] = 1
    else:
        # Unknown education level
        df['Parent_Education_Level_Unknown'] = 1

    # Family Income Level
    income_levels = ['Low', 'Medium']
    income = str(processed['family_income_level']).lower()

    for level in income_levels:
        column_name = f'Family_Income_Level_{level}'
        df[column_name] = 0

    # Set the appropriate income level to 1
    if 'low' in income:
        df['Family_Income_Level_Low'] = 1
    elif 'medium' in income or 'mid' in income:
        df['Family_Income_Level_Medium'] = 1
    # High income is represented by both Low and Medium being 0

    # Drop the original categorical columns that were used to create the encoded ones
    if 'gender' in df.columns:
        df = df.drop('gender', axis=1)
    if 'department' in df.columns:
        df = df.drop('department', axis=1)
    if 'extracurricular_activities' in df.columns:
        df = df.drop('extracurricular_activities', axis=1)
    if 'internet_access_at_home' in df.columns:
        df = df.drop('internet_access_at_home', axis=1)
    if 'parent_education_level' in df.columns:
        df = df.drop('parent_education_level', axis=1)
    if 'family_income_level' in df.columns:
        df = df.drop('family_income_level', axis=1)

    # Ensure features are in the exact same order as during training
    # This is critical to avoid the "Feature names must be in the same order as they were in fit" warning
    # Order extracted from model.feature_names_in_ in the logs
    expected_feature_order = [
        'Age',
        'Attendance (%)',
        'Midterm_Score',
        'Final_Score',
        'Assignments_Avg',
        'Quizzes_Avg',
        'Participation_Score',
        'Projects_Score',
        'Total_Score',
        'Study_Hours_per_Week',
        'Stress_Level (1-10)',
        'Sleep_Hours_per_Night',
        'Gender_Male',
        'Department_CS',
        'Department_Engineering',
        'Department_Mathematics',
        'Extracurricular_Activities_Yes',
        'Internet_Access_at_Home_Yes',
        'Parent_Education_Level_High School',
        "Parent_Education_Level_Master's",
        'Parent_Education_Level_PhD',
        'Parent_Education_Level_Unknown',
        'Family_Income_Level_Low',
        'Family_Income_Level_Medium'
    ]

    # Reorder processed_data to match expected_feature_order
    df = df.reindex(columns=expected_feature_order, fill_value=0)

    # --- StandardScaler integration ---
    try:
        scaler = joblib.load('scaler.joblib')
        scaler_features = [
            'Age',
            'Attendance (%)',
            'Midterm_Score',
            'Final_Score',
            'Assignments_Avg',
            'Quizzes_Avg',
            'Participation_Score',
            'Projects_Score',
            'Total_Score',
            'Study_Hours_per_Week',
            'Stress_Level (1-10)',
            'Sleep_Hours_per_Night',
            'Gender_Male',
            'Department_CS',
            'Department_Engineering',
            'Department_Mathematics',
            'Extracurricular_Activities_Yes',
            'Internet_Access_at_Home_Yes',
            'Parent_Education_Level_High School',
            "Parent_Education_Level_Master's",
            'Parent_Education_Level_PhD',
            'Parent_Education_Level_Unknown',
            'Family_Income_Level_Low',
            'Family_Income_Level_Medium'
        ]
        # Check which features from scaler_features are actually in df
        available_features = [col for col in scaler_features if col in df.columns]

        if len(available_features) > 0:
            # Only scale features that are available
            logging.debug("Before scaling:\n%s", df[available_features].head())

            df.loc[:, available_features] = scaler.transform(df[available_features])
            logging.debug("After scaling:\n%s", scaler.transform(df[available_features][:5]))
            # logging.debug("Applied StandardScaler to available features: %s", available_features)

            # Log any missing features
            missing = [col for col in scaler_features if col not in df.columns]
            if missing:
                logging.warning("Some features were not available for scaling: %s", missing)
        else:
            logging.error("No matching features available for scaling")

    except Exception as e:
        logging.error("Scaler could not be loaded or applied: %s", str(e))

    logging.debug("Final feature order: %s", list(df.columns))
    logging.debug("Processed data features: %s", list(df.columns))
    logging.debug("Processed data shape: %s", df.shape)

    return df

def predict_grade(data, model):
    """
    Generate grade prediction for a single student

    Args:
        data (dict): Student data in dictionary format
        model: Trained ML model with predict and predict_proba methods

    Returns:
        dict: Prediction result with predicted grade and confidence scores
    """
    if model is None:
        raise ValueError("Model not loaded correctly")

    logging.debug("Input data: %s", data)

    # Process input data
    processed_data = process_input_data(data)

    # Log the processed data features
    logging.debug("Processed data features: %s", list(processed_data.columns))
    logging.debug("Processed data shape: %s", processed_data.shape)

    # Log model feature names if available
    if hasattr(model, 'feature_names_in_'):
        logging.debug("Model feature names: %s", list(model.feature_names_in_))

    # Compare features
    if hasattr(model, 'feature_names_in_'):
        model_features = set(model.feature_names_in_)
        data_features = set(processed_data.columns)
        missing_features = model_features - data_features
        extra_features = data_features - model_features

        if missing_features:
            logging.warning("Missing features: %s", missing_features)
        if extra_features:
            logging.warning("Extra features: %s", extra_features)

    # Get predictions
    try:
        grade_index = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        logging.debug("Prediction successful. Grade index: %s", grade_index)
    except Exception as e:
        logging.error("Prediction failed: %s", str(e))
        return {"error": str(e)}
    # Map grade index to letter grade
    grade_mapping = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'F'
    }

    predicted_grade = grade_mapping.get(grade_index, 'Unknown')

    # Create confidence dictionary
    confidence = {
        grade_mapping[i]: round(float(prob), 2)  # Convert numpy float to Python float for JSON serialization
        for i, prob in enumerate(probabilities)
    }

    return {
        "predicted_grade": predicted_grade,
        "confidence": confidence
    }

def batch_predict(data_list, model):
    """
    Process a batch of student records

    Args:
        data_list (list): List of student data dictionaries
        model: Trained ML model

    Returns:
        list: List of prediction results
    """
    logging.info("Starting batch prediction with %d records", len(data_list))
    results = []

    for i, student_data in enumerate(data_list):
        logging.debug("Processing record %d/%d", i+1, len(data_list))
        try:
            result = predict_grade(student_data, model)
            results.append(result)
            logging.debug("Successfully processed record %d", i+1)
        except Exception as e:
            logging.error("Error processing record %d: %s", i+1, str(e))
            # Add error information to results
            results.append({"error": str(e)})

    logging.info("Batch prediction completed. Processed %d records", len(results))
    return results