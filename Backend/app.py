# if you want to use BERT model then run this file 
# command :
#       windows : python app.py
#       linux   : python3 app.py

import os
import re
import json
import hashlib
import numpy as np
import torch
import joblib
try:
    import xgboost
except ImportError:
    pass # Will be handled later if model fails to load
from scipy.sparse import hstack
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_DIR = os.path.join(BASE_DIR, "bert_fakejob_model")
XGBOOST_MODEL_DIR = os.path.join(BASE_DIR, "xgboost_models")
LOGISTIC_MODEL_DIR = os.path.join(BASE_DIR, "models")
USER_FILE = os.path.join(BASE_DIR, "users.json")            # user data stored here

# XGBoost Paths
XGB_VECTORIZER_PATH = os.path.join(XGBOOST_MODEL_DIR, "tfidf_vectorizer.pkl")
XGB_MODEL_PATH = os.path.join(XGBOOST_MODEL_DIR, "xgboost_fakejob_model.pkl")
XGB_KEYWORDS_PATH = os.path.join(XGBOOST_MODEL_DIR, "suspicious_keywords.pkl")

# Logistic Paths
LOG_VECTORIZER_PATH = os.path.join(LOGISTIC_MODEL_DIR, "tfidf_vectorizer_logestic.pkl")
LOG_MODEL_PATH = os.path.join(LOGISTIC_MODEL_DIR, "fake_job_detector_model_logestic.pkl")

SUSPICIOUS_KEYWORDS_LIST = [
    'earn', 'money', 'income', 'cash', 'hurry', 'limited', 'urgent',
    'apply now', 'no experience', 'work from home', 'easy job', 'quick money',
    'investment', 'bitcoin', 'crypto', 'telegram', 'whatsapp', 'pay to apply',
    'registration fee', 'send details', 'bank account', 'click here', 'link'
]

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load Models
# -----------------------------
models = {}

# Load BERT
try:
    print("⏳ Loading BERT model for inference ...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()
    models["bert"] = {"model": bert_model, "tokenizer": tokenizer, "device": device}
    print("✅ BERT model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load BERT model: {e}")

# Load XGBoost
try:
    print("⏳ Loading XGBoost model ...")
    xgb_vectorizer = joblib.load(XGB_VECTORIZER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_keywords = joblib.load(XGB_KEYWORDS_PATH)
    models["xgboost"] = {
        "model": xgb_model, 
        "vectorizer": xgb_vectorizer, 
        "keywords": xgb_keywords
    }
    print("✅ XGBoost model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load XGBoost model: {e}")

# Load Logistic Regression
try:
    print("⏳ Loading Logistic Regression model ...")
    log_vectorizer = joblib.load(LOG_VECTORIZER_PATH)
    log_model = joblib.load(LOG_MODEL_PATH)
    models["logistic"] = {
        "model": log_model, 
        "vectorizer": log_vectorizer, 
        "keywords": SUSPICIOUS_KEYWORDS_LIST
    }
    print("✅ Logistic Regression model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load Logistic Regression model: {e}")


# -----------------------------
# Inference Utility
# -----------------------------

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    return " ".join(text.lower().split())

def build_features(text: str, vectorizer, keywords):
    """Build combined TF-IDF and numeric features for ML models"""
    text_clean = clean_text(text)
    X_tfidf = vectorizer.transform([text_clean])
    text_len = np.log1p(len(text_clean))
    num_suspicious = sum(word in text_clean for word in keywords)
    X_extra = np.array([[text_len, num_suspicious]])
    X_final = hstack([X_tfidf, X_extra])

    return X_final, {
        "text_len_log1p": float(text_len),
        "num_suspicious_keywords": int(num_suspicious)
    }

def predict_fake_job_bert(text, model_data):
    """Predict using BERT model"""
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    device = model_data["device"]
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    label = "🟥 FAKE JOB POSTING" if pred == 1 else "🟩 REAL JOB POSTING"
    confidence = round(float(np.max(probs)) * 100, 2)

    return {
        "label": label,
        "pred": pred,
        "confidence_percent": confidence,
        "probabilities": {"real": float(probs[0]), "fake": float(probs[1])},
        "model_used": "BERT"
    }

def predict_fake_job_ml(text, model_data, model_name="ML"):
    """Predict using XGBoost or Logistic Regression"""
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    keywords = model_data["keywords"]
    
    X, meta = build_features(text, vectorizer, keywords)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()

    label = "🟥 FAKE JOB POSTING" if pred == 1 else "🟩 REAL JOB POSTING"
    confidence = round(float(max(proba)) * 100, 2)

    return {
        "label": label,
        "pred": pred,
        "confidence_percent": confidence,
        "probabilities": {"real": float(proba[0]), "fake": float(proba[1])},
        "features": meta,
        "model_used": model_name
    }

# -----------------------------
# Helper Functions for Users
# -----------------------------
def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump([], f)
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# Routes
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "models_loaded": list(models.keys())
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    model_type = data.get("model", "bert").lower() # default to "bert"
    
    # Handle multiple texts support (optional, if needed by frontend, but mostly single)
    if not text or not isinstance(text, str):
         if "texts" in data and isinstance(data["texts"], list):
             # Simplify for now: just use BERT for batch or loop
             # But request asks for model selection
             return jsonify({"error": "Batch prediction not fully supported with model selection in this update."}), 400
         return jsonify({"error": "Provide 'text' (string)."}), 400

    if model_type not in models:
        return jsonify({"error": f"Model '{model_type}' not found or failed to load. Available: {list(models.keys())}"}), 400
    
    try:
        if model_type == "bert":
            result = predict_fake_job_bert(text, models["bert"])
        elif model_type == "xgboost":
            result = predict_fake_job_ml(text, models["xgboost"], "XGBoost")
        elif model_type == "logistic":
            result = predict_fake_job_ml(text, models["logistic"], "Logistic Regression")
        else:
            return jsonify({"error": "Unknown model type"}), 400
            
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    confirm_password = data.get("confirmPassword", "")

    if not name or not email or not password or not confirm_password:
        return jsonify({"error": "All fields are required."}), 400

    if password != confirm_password:
        return jsonify({"error": "Passwords do not match."}), 400

    users = load_users()

    if any(u["email"] == email for u in users):
        return jsonify({"error": "Email already registered."}), 400

    hashed_pw = hash_password(password)
    users.append({"name": name, "email": email, "password": hashed_pw})
    save_users(users)

    return jsonify({"message": "Signup successful!"}), 200


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    users = load_users()
    hashed_pw = hash_password(password)
    user = next((u for u in users if u["email"] == email and u["password"] == hashed_pw), None)

    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    return jsonify({"message": "Login successful!", "user": {"name": user["name"], "email": user["email"]}}), 200


# -----------------------------
# Dev Entry
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
