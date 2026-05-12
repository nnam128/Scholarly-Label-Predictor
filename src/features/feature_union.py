#trộn tất cả feature lại thành 1 vector duy nhất

from scipy.sparse import hstack, csr_matrix
import numpy as np
import pandas as pd


class FeatureUnion:
    """
    Combine TF-IDF features + Metadata features into a single feature matrix.
    Designed for SVM / Logistic Regression baseline.
    """
    def __init__(self):
        pass
    
    def fit_transform(self, tfidf_extractor, metadata_extractor, df: pd.DataFrame):
        # TF-IDF FEATURES (SPARSE)
        tfidf_extractor.fit(df)
        tfidf_features = tfidf_extractor.transform_concat(df)
        
        # METADATA FEATURES (DENSE)
        metadata_extractor.fit(df)
        metadata_df = metadata_extractor.transform(df)

        metadata_matrix = csr_matrix(metadata_df.values)
        
        # CONCAT
        X = hstack([tfidf_features, metadata_matrix]).tocsr()
        
        return X, metadata_df
    
    def transform(self, tfidf_extractor, metadata_extractor, df: pd.DataFrame):
        tfidf_features = tfidf_extractor.transform_concat(df)
        
        metadata_df = metadata_extractor.transform(df)
        metadata_matrix = csr_matrix(metadata_df.values)
        
        X = hstack([tfidf_features, metadata_matrix]).tocsr()
        
        return X