# ToxiShield — Toxic Comment Moderation Portal (End-to-End)

**What:** Binary toxicity detector (toxic vs clean) trained on Davidson et al. (2017).  
**Stack:** scikit-learn → FastAPI → Streamlit → Docker → Render.

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/train.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
# In another terminal:
streamlit run ui/app.py
```

## Docker
```bash
docker build -t toxishield:latest .
docker run -p 8000:8000 toxishield:latest
```

## Render
API (Docker): connect repo, deploy.  
UI (Python): build `pip install -r requirements.txt`, start `streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0`, set secret `API_URL=https://<your-api>.onrender.com/predict`.
