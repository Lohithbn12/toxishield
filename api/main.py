from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
pipe = joblib.load(MODEL_PATH)

app = FastAPI(title="ToxiShield API", version="1.0")

class TextIn(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(payload: TextIn):
    proba = float(pipe.predict_proba([payload.text])[0,1])
    pred = int(proba >= 0.5)
    return {"toxic_proba": proba, "toxic_pred": pred}

@app.get("/")
def index():
    return {"status":"ok", "message":"ToxiShield API running", "try": ["/health", "/docs"]}
