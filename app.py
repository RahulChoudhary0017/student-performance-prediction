from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = joblib.load("models/student_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🎯 Getting data from index.html form
        attendance = float(request.form['attendance'])
        study_hours = float(request.form['study_hours'])
        assignment = float(request.form['assignment'])
        midterm = float(request.form['midterm'])
        previous_marks = float(request.form['previous_marks'])

        # 🔢 Convert into model input format
        features = np.array([[
            attendance,
            study_hours,
            assignment,
            midterm,
            previous_marks
        ]])

        # 🤖 Prediction
        prediction = model.predict(features)[0]

        # 🎯 Send result to result.html
        return render_template(
            "result.html",
            prediction=prediction,
            attendance=attendance,
            study_hours=study_hours,
            assignment=assignment,
            midterm=midterm,
            previous_marks=previous_marks
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Error Occurred",
            attendance="N/A",
            study_hours="N/A",
            assignment="N/A",
            midterm="N/A",
            previous_marks="N/A",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)