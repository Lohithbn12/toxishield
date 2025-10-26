import os, html, requests
import streamlit as st

st.set_page_config(page_title="ToxiShield — Moderation Portal", page_icon="🛡️", layout="centered")
st.title("🛡️ ToxiShield — Moderation Portal")
st.caption("Detect toxic content using a ML model (TF-IDF + Logistic Regression).")

# ← We read API_URL from environment (set this in Render UI service → Settings → Environment)
API_URL = os.environ.get("API_URL", "https://toxishield.onrender.com/predict")

# Initialize session state to remember the last probability & text
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

sample = "I totally disagree with you."
text = st.text_area("Enter text to check", value=sample if not st.session_state.last_text else st.session_state.last_text, height=160)

col1, col2 = st.columns([1,1])
with col1:
    threshold = st.slider("Toxic threshold", 0.1, 0.9, 0.5, 0.05)
with col2:
    btn = st.button("Analyze", use_container_width=True)

def call_api(s: str):
    r = requests.post(API_URL, json={"text": s}, timeout=15)
    r.raise_for_status()
    return r.json()

def color_for(p):
    if p >= 0.8: return "red"
    if p >= 0.6: return "orange"
    if p >= 0.4: return "gold"
    return "green"

# 1) Only call API when the Analyze button is clicked
if btn:
    try:
        res = call_api(text)
        st.session_state.last_proba = float(res.get("toxic_proba", 0.0))
        st.session_state.last_text = text
    except Exception as e:
        st.error(f"API error: {e}")
        st.info("Check the API_URL environment variable and that the FastAPI service is live at /health.")

# 2) On every rerun (e.g., moving the threshold), if we have a saved probability, recompute the label and render
if st.session_state.last_proba is not None:
    p = st.session_state.last_proba
    pred = int(p >= threshold)

    st.markdown(f"**Probability (toxic):** `{p:.3f}`  —  **Prediction:** {'🚨 Toxic' if pred else '✅ Clean'}  \n"
                f"_Threshold:_ `{threshold:.2f}`  (move the slider to see label change)")
    st.progress(min(max(p,0.0),1.0))
    st.markdown(
        f"<div style='padding:10px;border-radius:8px;border:1px solid #ddd'>"
        f"<b>Preview:</b> <span style='color:{color_for(p)}'>{html.escape(st.session_state.last_text)}</span></div>",
        unsafe_allow_html=True
    )
else:
    st.info("Enter some text and click **Analyze**.")

st.markdown("---")
st.caption("Model: TF-IDF + Logistic Regression trained on Davidson et al. (2017) hate/offensive language dataset.")
st.caption("Build: v2")

