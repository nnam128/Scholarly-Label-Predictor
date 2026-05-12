from typing import List, Optional
from keybert import KeyBERT


class KeywordExtractor:
    """
    Extract keywords from scientific text using KeyBERT.
    Uses SBERT embeddings internally (no LLM API needed).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = KeyBERT(model=model_name)

    def extract_keywords(
        self,
        text: str,
        top_k: int = 5,
        stop_words: str = "english"
    ) -> List[str]:
        """
        Extract keywords from a text.
        """
        if not text or not isinstance(text, str):
            return []

        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=stop_words,
            top_n=top_k
        )

        return [kw[0] for kw in keywords]

    def extract_from_row(
        self,
        title: str,
        abstract: Optional[str] = None,
        top_k: int = 5
    ) -> List[str]:
        """
        Combine title + abstract for better keyword extraction.
        """
        text = title or ""

        if abstract:
            text = f"{title}. {abstract}"

        return self.extract_keywords(text, top_k=top_k)