from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict

class MetadataFeatureExtractor:
    """
    Extract and Scale structured metadata features from paper data.
    """
    def __init__(self):
        self.venue_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.author_freq = defaultdict(int)
        # Danh sách các cột cần được Scale (loại trừ các cột nhị phân has_...)
        self.cols_to_scale = [
            "venue_encoded", "year_norm", "author_count", 
            "author_freq_mean", "author_freq_max", "author_freq_min"
        ]
        
    def fit(self, df: pd.DataFrame):
        """
        Fit encoders and scaler on training data.
        """
        # 1. Fit Venue
        if "venue" in df.columns:
            self.venue_encoder.fit(df["venue"].fillna("unknown"))
            
        # 2. Fit Author Frequencies
        if "authors" in df.columns:
            for authors in df["authors"].fillna(""):
                for a in str(authors).split(","):
                    a = a.strip()
                    if a:
                        self.author_freq[a] += 1
        
        # 3. Fit Scaler
        # Chúng ta cần chạy transform tạm thời để lấy dữ liệu thô phục vụ việc fit scaler
        raw_features = self._extract_raw_features(df)
        self.scaler.fit(raw_features[self.cols_to_scale])
        
        self.is_fitted = True
        return self
    
    def _author_features(self, authors: str) -> Dict:
        if not isinstance(authors, str) or not authors.strip():
            return {"author_count": 0, "author_freq_mean": 0, "author_freq_max": 0, "author_freq_min": 0}
        
        author_list = [a.strip() for a in authors.split(",") if a.strip()]
        freqs = [self.author_freq.get(a, 0) for a in author_list]
        
        return {
            "author_count": len(author_list),
            "author_freq_mean": float(np.mean(freqs)) if freqs else 0,
            "author_freq_max": int(np.max(freqs)) if freqs else 0,
            "author_freq_min": int(np.min(freqs)) if freqs else 0,
        }

    def _extract_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hàm nội bộ để trích xuất feature chưa qua chuẩn hóa.
        """
        def process_row(row):
            venue = row.get("venue", "unknown")
            year = row.get("year", 0)
            authors = str(row.get("authors", ""))
            
            # Venue encoding
            try:
                v_enc = self.venue_encoder.transform([venue])[0] if self.is_fitted else -1
            except:
                v_enc = -1
                
            # Year normalization (tạm thời giữ nguyên logic cũ của bạn)
            try:
                year_val = int(year)
            except:
                year_val = 2000
            y_norm = (year_val - 2000) / 30
            
            auth_feats = self._author_features(authors)
            
            return {
                "venue_encoded": v_enc,
                "year_norm": y_norm,
                "author_count": auth_feats["author_count"],
                "author_freq_mean": auth_feats["author_freq_mean"],
                "author_freq_max": auth_feats["author_freq_max"],
                "author_freq_min": auth_feats["author_freq_min"],
                "has_abstract": 1 if str(row.get("abstract", "")).strip() else 0,
                "has_authors": 1 if auth_feats["author_count"] > 0 else 0
            }
        
        return df.apply(process_row, axis=1, result_type="expand")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features and apply StandardScaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Extractor must be fitted before transform.")
            
        features_df = self._extract_raw_features(df)
        
        # Áp dụng chuẩn hóa cho các cột số
        features_df[self.cols_to_scale] = self.scaler.transform(features_df[self.cols_to_scale])
        
        return features_df