import os, requests
URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
OUT = "data/labeled_data.csv"
def main():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        print(f"{OUT} already exists; skipping download."); return
    r = requests.get(URL, timeout=30); r.raise_for_status()
    with open(OUT, "wb") as f: f.write(r.content)
    print(f"Downloaded to {OUT} ({len(r.content)} bytes).")
if __name__ == "__main__": main()
