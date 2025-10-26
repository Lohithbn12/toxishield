from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1,2),
        min_df=2,
        max_features=200000
    )
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    return Pipeline([("tfidf", vec), ("lr", clf)])
