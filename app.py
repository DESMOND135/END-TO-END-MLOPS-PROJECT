from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flashing messages

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')  # Load the form page

# Route to handle prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return redirect(url_for('index'))  # Redirect GET requests to home

    try:
        # Collect form data safely
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score', 0)),
            writing_score=float(request.form.get('writing_score', 0))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        app.logger.info(f"Data for Prediction:\n{pred_df}")

        # Make prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        app.logger.info(f"Prediction Result: {prediction}")

        # Pass prediction to template
        return render_template('home.html', results=prediction[0])

    except ValueError:
        flash("Please enter valid numeric values for scores.", "danger")
        return redirect(url_for('index'))

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        flash(f"An error occurred: {e}", "danger")
        return redirect(url_for('index'))

# Run the app locally
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
