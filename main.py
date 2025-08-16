from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins like ["http://localhost:3000", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Include OPTIONS for preflight
    allow_headers=["*"],  # In production, be more specific like ["Content-Type", "Authorization"]
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# --- Load model and tokenizer ---
model_dir = "./onnx"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
ort_session = ort.InferenceSession(f"{model_dir}/model.onnx")
label_names = ["story_rant", "sports", "entertainment", "news"]
label_encoder = LabelEncoder()
label_encoder.fit(label_names)

# --- Helper function ---
def classify_text(text: str):
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=128)
    ort_inputs = {k: v for k, v in inputs.items()}
    
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pred_idx = np.argmax(probs, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[0][pred_idx])
    return {
        "label": predicted_label,
        "confidence": confidence,
        "confidence_scores": {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probs[0])
        }
    }

# --- Request model for POST ---
class TextInput(BaseModel):
    text: str

# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "Text classification API is running 🚀"}

@app.post("/classify")
def classify_post(input: TextInput):
    return classify_text(input.text)

@app.get("/classify")
def classify_get(text: str = Query(..., min_length=1)):
    return classify_text(text)

# Optional: Explicit OPTIONS handler (usually not needed with CORSMiddleware)
@app.options("/classify")
def options_classify():
    return {"message": "OK"}