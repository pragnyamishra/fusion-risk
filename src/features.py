"""Multimodal ColumnTransformer shared by every model."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C


def build_preprocessor() -> ColumnTransformer:
    """Financial + behavioral (scaled) + categorical (OHE) + text (TF-IDF)."""
    return ColumnTransformer(
        transformers=[
            ("financial", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), C.FINANCIAL_COLS),
            ("behavioral", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), C.BEHAVIORAL_COLS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", max_categories=20),
             C.CATEGORICAL_COLS),
            ("text", TfidfVectorizer(max_features=1500, min_df=3, max_df=0.8,
                                     ngram_range=(1, 2)), C.TEXT_COL),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer):
    """Best-effort expanded feature names for explainability."""
    names = []
    names += C.FINANCIAL_COLS
    names += C.BEHAVIORAL_COLS
    try:
        ohe = preprocessor.named_transformers_["categorical"]
        names += list(ohe.get_feature_names_out(C.CATEGORICAL_COLS))
    except Exception:
        names += C.CATEGORICAL_COLS
    try:
        tfidf = preprocessor.named_transformers_["text"]
        names += [f"tfidf__{t}" for t in tfidf.get_feature_names_out()]
    except Exception:
        pass
    return names
