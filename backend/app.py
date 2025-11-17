# project/backend/app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Paths ---
TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates')
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'model_Artifacts1')

MODEL_PATH = os.path.join(ARTIFACT_DIR, 'best_model.pkl')
METADATA_PATH = os.path.join(ARTIFACT_DIR, 'model_metadata.pkl')
ENCODERS_PATH = os.path.join(ARTIFACT_DIR, 'label_encoders.pkl')
FEATURES_PATH = os.path.join(ARTIFACT_DIR, 'feature_columns.pkl')

app = Flask(__name__, template_folder=TEMPLATES_PATH)
CORS(app)  # Allow frontend to call API

# --- Load model artifacts ---
if not all(os.path.exists(p) for p in [MODEL_PATH, METADATA_PATH, ENCODERS_PATH, FEATURES_PATH]):
    raise RuntimeError(
        f"Put the artifact files (best_model.pkl, model_metadata.pkl, label_encoders.pkl, feature_columns.pkl) inside {ARTIFACT_DIR}"
    )

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
encoders = joblib.load(ENCODERS_PATH)          # dict: {col_name: LabelEncoder()}
feature_columns = joblib.load(FEATURES_PATH)   # list of feature names in training order

# Mapping prediction → human-friendly label
rating_map = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/form')
def index():
    """Show input form (index.html)."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle both API (JSON) and HTML form submissions."""
    # --- Get input ---
    if request.is_json:  
        payload = request.get_json(force=True)
    else:  
        payload = request.form.to_dict()

    # --- Build row with defaults ---
    row, fallback_used = {}, []
    for col in feature_columns:
        if col in payload:
            row[col] = payload[col]
        else:
            if col in encoders:  # categorical
                row[col] = encoders[col].classes_[0]
            else:  # numeric
                row[col] = 0
            fallback_used.append(col)

    df = pd.DataFrame([row])

    # --- Numeric columns ---
    for col in df.columns:
        if col not in encoders:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Encode categoricals ---
    warnings = {}
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            val = df.at[0, col]
            if val not in le.classes_:
                warnings[col] = {
                    "received": val,
                    "replaced_with": le.classes_[0],
                    "allowed": list(le.classes_[:10])
                }
                df.at[0, col] = le.classes_[0]
            df[col] = le.transform(df[col])

    # --- Keep correct column order ---
    df = df[feature_columns]

    # --- Predict ---
    pred = model.predict(df)
    if metadata.get('needs_label_shift', False):
        pred = pred + 1
    pred_int = int(pred[0])
    label = rating_map.get(pred_int, str(pred_int))

    # --- If API call → return JSON ---
    if request.is_json:
        response = {"prediction": pred_int, "label": label}
        if warnings:
            response['warnings'] = warnings
        if fallback_used:
            response.setdefault('warnings', {})['defaults_used'] = fallback_used
        return jsonify(response)

    # --- If form submission → render result.html ---
    return render_template(
        "result.html",
        prediction=label,
        confidence=round(metadata.get("accuracy", 0.96) * 100, 2)
    )
@app.route('/result')
def result_page():
    prediction = request.args.get('prediction')
    label = request.args.get('label')
    return render_template('result.html', prediction=prediction, label=label)




if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
