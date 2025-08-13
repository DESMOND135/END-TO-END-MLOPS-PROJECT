from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly
import json

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    age = int(request.form["age"])
    sex = 1 if request.form["sex"] == "Male" else 0
    bmi = float(request.form["bmi"])
    smoker = 1 if request.form["smoker"] == "Yes" else 0
    region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
    region = region_map[request.form["region"]]
    children = int(request.form["children"])

    # Prepare and scale input
    input_data = np.array([[age, sex, bmi, smoker, region, children]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    # Plot 1: Charges by Age (fixed smoker=0, sex=0, region=0, children=1, bmi=25)
    ages = np.linspace(18, 100, 100)
    input_matrix = np.column_stack([ages, np.zeros(100), np.ones(100)*25, np.zeros(100), np.zeros(100), np.ones(100)])
    charges = model.predict(scaler.transform(input_matrix))

    df = pd.DataFrame({'Age': ages, 'Charges': charges})
    fig1 = px.line(df, x="Age", y="Charges", title="Insurance Charges by Age")
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # Plot 2: Charges by Age and Smoker
    smoker_values = [0, 1]
    smoker_charges = [model.predict(scaler.transform(np.column_stack([
        ages, np.zeros(100), np.ones(100)*25, np.ones(100)*s, np.zeros(100), np.ones(100)]))) for s in smoker_values]

    smoker_charges_df = pd.DataFrame({
        'Age': np.tile(ages, 2),
        'Charges': np.concatenate(smoker_charges),
        'Smoker': ['No']*100 + ['Yes']*100
    })

    fig2 = px.line(smoker_charges_df, x="Age", y="Charges", color="Smoker", title="Charges by Age and Smoker Status")
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("result.html", prediction=round(prediction, 2), plot1=graphJSON1, plot2=graphJSON2)

if __name__ == "__main__":
    app.run(debug=True)
