from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseCrawler(ABC):
    """
    Abstract base class for metadata crawling from DOI.
    Each crawler implementation (Semantic Scholar, CrossRef,
    OpenAlex, etc.) should inherit from this class.
    Expected output format:
    {
        "id": int,
        "doi": str,
        "abstract": str,
        "crawled_authors": str
    }
    """
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        
    @abstractmethod
    def fetch_by_doi(
        self,
        doi: str,
        paper_id: int
    ) -> Optional[Dict]:
        """
        Fetch metadata using DOI.
        Parameters
        ----------
        doi : str
            Paper DOI.
        paper_id : int
            Unique paper ID from dataset.
        Returns
        -------
        Optional[Dict]
            Dictionary containing crawled metadata.
            Return None if crawling fails.
        """
        pass
    
    def normalize_doi(self, doi: str) -> str:
        """
        Normalize DOI string.
        Examples
        --------
        https://doi.org/10.xxx
        DOI:10.xxx
        10.xxx
        -> 10.xxx
        """
        if not doi:
            return ""
        doi = str(doi).strip()
        doi = doi.replace("https://doi.org/", "")
        doi = doi.replace("http://doi.org/", "")
        doi = doi.replace("DOI:", "")
        doi = doi.replace("doi:", "")
        return doi.strip()
    
    def format_authors(self, authors_list) -> str:
        """
        Convert author list to a clean comma-separated string.
        Parameters
        ----------
        authors_list : list
        Returns
        -------
        str
        """
        if not authors_list:
            return ""
        
        cleaned = [str(author).strip() for author in authors_list if author]
        return ", ".join(cleaned)
    
    def build_result(
        self,
        doi: str,
        paper_id: int,
        abstract: str = "",
        authors: str = ""
    ) -> Dict:
        """
        Standardize crawler output.
        """
        return {
            "id": paper_id,
            "doi": self.normalize_doi(doi),
            "abstract": abstract if abstract else "",
            "crawled_authors": authors if authors else ""
        }
