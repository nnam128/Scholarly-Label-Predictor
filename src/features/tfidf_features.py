from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFeatureExtractor:
    """
    Advanced TF-IDF feature extractor for scientific paper classification.
    Includes:
    - Title TF-IDF (keyword signal)
    - Abstract TF-IDF (semantic topic signal)
    - Authors TF-IDF (research community signal)
    """
    def __init__(
        self,
        max_title_features: int = 3000,
        max_abstract_features: int = 8000,
        max_author_features: int = 1500
    ):
        #title
        self.title_vectorizer = TfidfVectorizer(
            max_features=max_title_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2, #appear 2 time
            max_df=0.9 #appear in less than 90% documents
        )
        
        #abstract
        self.abstract_vectorizer = TfidfVectorizer(
            max_features=max_abstract_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        
        #author
        self.author_vectorizer = TfidfVectorizer(
            max_features=max_author_features,
            ngram_range=(1, 1),
            min_df=1
        )
        self.is_fitted = False
        
    def _normalize_authors(self, authors: str) -> str:
        """
        Convert:
        "A. Dovier, T. Dreossi"
        ->
        "dovier dreossi"
        """
        if not authors or not isinstance(authors, str):
            return ""
        authors = authors.lower()
        parts = [a.strip() for a in authors.split(",") if a.strip()]
        
        cleaned = []
        for p in parts:
            tokens = p.split()
            if len(tokens) == 0:
                continue
            cleaned.append(tokens[-1])
        return " ".join(cleaned)
    
    def _safe_text(self, x) -> str:
        if x is None:
            return ""
        return str(x)
        
    def fit(self, df: pd.DataFrame):
        """
        Fit TF-IDF vectorizers on TRAIN only.
        """
        titles = df["title"].fillna("").map(self._safe_text)
        abstracts = df["abstract"].fillna("").map(self._safe_text)
        authors = df["authors"].fillna("").map(self._normalize_authors)
        self.title_vectorizer.fit(titles)
        self.abstract_vectorizer.fit(abstracts)
        self.author_vectorizer.fit(authors)
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> Dict:
        titles = df["title"].fillna("").map(self._safe_text)
        abstracts = df["abstract"].fillna("").map(self._safe_text)
        authors = df["authors"].fillna("").map(self._normalize_authors)
        
        title_tfidf = self.title_vectorizer.transform(titles)
        abstract_tfidf = self.abstract_vectorizer.transform(abstracts)
        author_tfidf = self.author_vectorizer.transform(authors)
        
        return {
            "title_tfidf": title_tfidf,
            "abstract_tfidf": abstract_tfidf,
            "author_tfidf": author_tfidf
        }
        
    def transform_concat(self, df: pd.DataFrame):
        """
        Return single concatenated sparse matrix.
        Useful for XGBoost / Logistic Regression.
        """
        from scipy.sparse import hstack
        
        features = self.transform(df)
        return hstack([
            features["title_tfidf"],
            features["abstract_tfidf"],
            features["author_tfidf"]
        ])