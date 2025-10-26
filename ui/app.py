import streamlit as st, requests, html

st.set_page_config(page_title="ToxiShield — Moderation Portal", page_icon="🛡️", layout="centered")
st.title("🛡️ ToxiShield — Moderation Portal")
st.caption("Detect toxic content using a ML model (TF‑IDF + Logistic Regression).")

API_URL = st.secrets.get("API_URL", "http://localhost:8000/predict")

sample = "I totally disagree with you."
text = st.text_area("Enter text to check", value=sample, height=160)

col1, col2 = st.columns([1,1])
with col1:
    threshold = st.slider("Toxic threshold", 0.1, 0.9, 0.5, 0.05)
with col2:
    btn = st.button("Analyze", use_container_width=True)

def call_api(s: str):
    r = requests.post(API_URL, json={"text": s}, timeout=15)
    r.raise_for_status()
    return r.json()

def color_score(p):
    if p >= 0.8: return "red"
    if p >= 0.6: return "orange"
    if p >= 0.4: return "gold"
    return "green"

if btn:
    try:
        res = call_api(text)
        p = res.get("toxic_proba", 0.0)
        pred = int(p >= threshold)
        st.markdown(f"**Probability (toxic):** `{p:.3f}`  —  **Prediction:** {'🚨 Toxic' if pred else '✅ Clean'}")
        st.progress(min(max(p,0.0),1.0))
        st.markdown(f"<div style='padding:10px;border-radius:8px;border:1px solid #ddd'>"
                    f"<b>Preview:</b> <span style='color:{color_score(p)}'>{html.escape(text)}</span></div>",
                    unsafe_allow_html=True)
    except Exception as e:
        st.error(f"API error: {e}")
        st.info("Check API_URL secret and that the FastAPI service is live.")
else:
    st.info("Enter some text and click **Analyze**.")

st.markdown("---")
st.caption("Model: TF‑IDF + Logistic Regression trained on Davidson et al. (2017) hate/offensive language dataset.")
