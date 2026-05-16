#Embedding semantic bằng Transformer(SBERT)
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class SBERTFeatureExtractor:
    def __init__(self, model_name='all-mpnet-base-v2', device=None):
        """
        Sử dụng SBERT để biến đổi Title + Abstract thành vector ngữ nghĩa.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Loaded SBERT model: {model_name} on {self.device}")

    def _prepare_text(self, df: pd.DataFrame):
        # Kết hợp Title và Abstract để có ngữ cảnh đầy đủ
        # Xử lý trường hợp abstract bị thiếu (NaN)
        titles = df['title'].fillna("").astype(str)
        abstracts = df['abstract'].fillna("").astype(str)
        
        # Format: "Title: ... Abstract: ..."
        combined_text = "Title: " + titles + " [SEP] Abstract: " + abstracts
        return combined_text.tolist()

    def transform(self, df: pd.DataFrame):
        texts = self._prepare_text(df)
        # Tiến hành encoding văn bản sang vector (thường là 384 hoặc 768 chiều)
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        return embeddings