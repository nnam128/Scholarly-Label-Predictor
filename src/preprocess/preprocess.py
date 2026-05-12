from pathlib import Path
import pandas as pd
import numpy as np
import re


class DataMerger:
    def transform(self, base_df: pd.DataFrame, crawl_df: pd.DataFrame) -> pd.DataFrame:
        df = base_df.merge(
            crawl_df,
            on="id",
            how="left",
            suffixes=("", "_crawl")
        )
        df["abstract"] = df["abstract"].replace(r"^\s*$", np.nan, regex=True)
        if "abstract_crawl" in df.columns:
            df["abstract"] = df["abstract"].fillna(df["abstract_crawl"])
            
        if "authors_crawl" in df.columns:
            mask = df["authors"].isna() | (df["authors"].fillna("").str.strip() == "")
            df.loc[mask, "authors"] = df.loc[mask, "authors_crawl"]
            
        df.drop(
            columns=[c for c in df.columns if "_crawl" in c],
            inplace=True,
            errors="ignore"
        )
        
        return df
    
class MissingHandler:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ensure required columns exist
        for col in ["title", "abstract", "authors"]:
            if col not in df.columns:
                df[col] = ""
                
        df["title"] = df["title"].fillna("").astype(str)
        df["abstract"] = df["abstract"].fillna("").astype(str)
        df["authors"] = df["authors"].fillna("").astype(str)
        
        if "venue" in df.columns:
            df["venue"] = df["venue"].fillna("unknown")
            
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
            
        return df
    
class AuthorNormalizer:
    def _clean(self, a: str) -> str:
        a = a.lower().strip()
        a = re.sub(r"\s+", " ", a)
        a = re.sub(r"[^a-z\s]", "", a)
        return a.strip()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        def process(authors):
            if not isinstance(authors, str) or not authors.strip():
                return ""
            parts = [self._clean(a) for a in authors.split(",")]
            parts = [p for p in parts if p]
            return ", ".join(parts)
        
        df["authors"] = df["authors"].apply(process)
        return df
    
class TextCleaner:
    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s\-\:\&\/]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["title"] = df["title"].apply(self.clean)
        if "abstract" in df.columns:
            df["abstract"] = df["abstract"].apply(self.clean)
            
        return df