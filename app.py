from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load ANN model, scaler, and feature columns
model = load_model("model/breast_cancer_ann_model.h5")
scaler = joblib.load("model/breast_cancer_scaler.joblib")
model_columns = joblib.load("model/breast_cancer_columns.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        pred_prob = model.predict(input_scaled)
        pred_class = (pred_prob > 0.5).astype(int)[0][0]
        result = "Malignant" if pred_class == 1 else "Benign"

        return jsonify({
            "Prediction": result,
            "Probability": float(pred_prob[0][0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False)
