import os, json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pipeline import build_pipeline

DATA = "data/labeled_data.csv"

def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"{DATA} not found. Run: python src/download_data.py")
    df = pd.read_csv(DATA)
    df["toxic"] = (df["class"] != 2).astype(int)
    X = df["tweet"].astype(str)
    y = df["toxic"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline(); pipe.fit(X_tr, y_tr)
    y_proba = pipe.predict_proba(X_te)[:,1]; y_hat = (y_proba >= 0.5).astype(int)
    report = classification_report(y_te, y_hat, output_dict=True)
    try: auc = roc_auc_score(y_te, y_proba)
    except Exception: auc = None
    os.makedirs("models", exist_ok=True); joblib.dump(pipe, "models/model.joblib")
    with open("models/metrics.json","w") as f: json.dump({"test_report": report, "test_auc": auc}, f, indent=2)
    print("Saved models/model.joblib")
    if auc is not None: print(f"Test AUC: {auc:.3f}")
if __name__ == "__main__": main()
