from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__, template_folder='Flask/tempelate')

# Load model
MODEL_PATH = "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except:
    print("❌ Model Not Found")
    model = None

# Feature names (IMPORTANT: same as training)
FEATURE_NAMES = [
    'baseline value',
    'accelerations',
    'fetal_movement',
    'uterine_contractions',
    'light_decelerations',
    'severe_decelerations',
    'prolongued_decelerations',
    'abnormal_short_term_variability',
    'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability',
    'histogram_width',
    'histogram_min',
    'histogram_max',
    'histogram_number_of_peaks',
    'histogram_number_of_zeroes',
    'histogram_mode',
    'histogram_mean',
    'histogram_median',
    'histogram_variance',
    'histogram_tendency'
]

# Class labels
CLASS_LABELS = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological"
}

# Home page (form)
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return render_template("output.html", error="Model not loaded")

        data = []

        # Get form data
        for feature in FEATURE_NAMES:
            key = feature.replace(" ", "_")
            value = request.form.get(key)

            if value == "" or value is None:
                return render_template("output.html", error=f"Missing: {feature}")

            data.append(float(value))

        # Convert to DataFrame
        input_df = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Prediction
        prediction = model.predict(input_df)[0]

        # Confidence
        try:
            prob = model.predict_proba(input_df)[0]
            confidence = round(max(prob) * 100, 2)
        except:
            confidence = 0

        label = CLASS_LABELS.get(int(prediction), "Unknown")

        return render_template(
            "output.html",
            prediction=prediction,
            prediction_text=label, 
            confidence=confidence
        )

    except Exception as e:
        return render_template("output.html", error=str(e))


# Inspect page
@app.route("/inspect")
def inspect():
    return render_template("inspect.html")


# Run app
if __name__ == "__main__":
    app.run(debug=True)