from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

class MetadataFeatureExtractor:
    """
    Extract structured metadata features from paper data
    """
    def __init__(self):
        self.venue_encoder = LabelEncoder()
        self.is_fitted = False
        self.author_freq = defaultdict(int) # author statistics
        
    def fit(self, df: pd.DataFrame):
        """
        Fit venue encoder on training data only
        """
        if "venue" in df.columns:
            self.venue_encoder.fit(df["venue"].fillna("unknown"))
        if "authors" in df.columns:
            for authors in df["authors"].fillna(""):
                for a in authors.split(","):
                    a = a.strip()
                    if a:
                        self.author_freq[a] += 1
        self.is_fitted = True
        return self
    
    #author feature
    def _author_features(self, authors: str) -> Dict:
        if not isinstance(authors, str):
            authors = ""
        authors = authors.strip()
        if not authors:
            return {
                "author_count": 0,
                "author_freq_mean": 0,
                "author_freq_max": 0,
                "author_freq_min": 0,
            }
        author_list = [a.strip() for a in authors.split(",") if a.strip()]
        if not author_list:
            return {
                "author_count": 0,
                "author_freq_mean": 0,
                "author_freq_max": 0,
                "author_freq_min": 0,
            }
        freqs = [self.author_freq.get(a, 0) for a in author_list]
        return {
            "author_count": len(author_list),
            "author_freq_mean": float(np.mean(freqs)),
            "author_freq_max": int(np.max(freqs)),
            "author_freq_min": int(np.min(freqs)),
        }
    
    def transform_row(self, row: Dict) -> Dict:
        """
        Convert a single paper into metadata features.
        """
        venue = row.get("venue", "unknown")
        year = row.get("year", 0)
        authors = row.get("authors", "")
        if not isinstance(authors, str):
            authors = ""
        
        #venue
        if self.is_fitted:
            try:
                venue_encoded = self.venue_encoder.transform([venue])[0]
            except:
                venue_encoded = -1
        else:
            venue_encoded = -1
        #year
        try:
            year = int(year)
        except:
            year = 0
        year_norm = (year - 2000) / 30  #scaling
        
        #authors
        author_feats = self._author_features(authors)
        
        #flag missing
        has_abstract = 1 if row.get("abstract") else 0
        has_authors = 1 if author_feats["author_count"] > 0 else 0
        
        return {
            "venue_encoded": venue_encoded,
            "year_norm": year_norm,
            
            "author_count": author_feats["author_count"],
            "author_freq_mean": author_feats["author_freq_mean"],
            "author_freq_max": author_feats["author_freq_max"],
            "author_freq_min": author_feats["author_freq_min"],
            
            "has_abstract": has_abstract,
            "has_authors": has_authors
        }
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert full dataframe into metadata feature dataframe.
        """
        features = df.apply(self.transform_row, axis=1, result_type="expand")
        return features