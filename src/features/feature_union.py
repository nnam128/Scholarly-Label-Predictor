from scipy.sparse import hstack, csr_matrix, issparse
import numpy as np
import pandas as pd

class FeatureUnion:
    """
    Kết hợp các đặc trưng Text + Metadata.
    Hỗ trợ 2 chế độ: TF-IDF/Sparse và Transformer/Dense.
    """
    def __init__(self, is_transformers: bool = False):
        self.is_transformers = is_transformers

    def _combine(self, text_features, metadata_df: pd.DataFrame):
        """
        Hàm nội bộ để trộn đặc trưng dựa trên loại dữ liệu.
        """
        metadata_values = metadata_df.values
        
        if self.is_transformers:
            # Trường hợp SBERT: Cả hai đều là dense (dày)
            text_dense = text_features if isinstance(text_features, np.ndarray) else text_features.toarray()
            X = np.concatenate([text_dense, metadata_values], axis=1)
            return X
        else:
            # Trường hợp TF-IDF: text_features là sparse (thưa)
            metadata_sparse = csr_matrix(metadata_values)
            X = hstack([text_features, metadata_sparse]).tocsr()
            return X

    def fit_transform(self, text_extractor, metadata_extractor, df: pd.DataFrame):
        # Trích xuất Text Features
        if hasattr(text_extractor, 'fit'):
            text_extractor.fit(df)
            
        # Giả sử text_extractor có hàm transform_concat (TF-IDF) hoặc transform (SBERT)
        if hasattr(text_extractor, 'transform_concat'):
            text_features = text_extractor.transform_concat(df)
        else:
            text_features = text_extractor.transform(df)
        
        metadata_extractor.fit(df)
        metadata_df = metadata_extractor.transform(df)

        X = self._combine(text_features, metadata_df)
        
        return X, metadata_df
    
    def transform(self, text_extractor, metadata_extractor, df: pd.DataFrame):
        if hasattr(text_extractor, 'transform_concat'):
            text_features = text_extractor.transform_concat(df)
        else:
            text_features = text_extractor.transform(df)
        
        metadata_df = metadata_extractor.transform(df)
        
        X = self._combine(text_features, metadata_df)
        
        return X