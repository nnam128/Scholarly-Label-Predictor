from typing import Optional, Dict, List
from bs4 import BeautifulSoup
import requests

from src.crawler.crawler import BaseCrawler


class DOIResolverCrawler(BaseCrawler):
    """
        Final fallback using DOI resolver.
        This redirects to publisher pages and tries a very light
        HTML extraction for abstract-related meta tags.
        This is intentionally simple because publisher pages vary a lot.
        This just focus on common meta tags for abstracts and does not attempt to extract authors
    """
    BASE_URL = "https://doi.org/"
    def __init__(self, timeout: int = 30):
        super().__init__(timeout=timeout)
        
    def fetch_by_doi(
        self,
        doi: str,
        paper_id: int
    ) -> Optional[Dict]:
        
        doi = self.normalize_doi(doi)
        if not doi:
            return None
        
        try:
            url = self.BASE_URL + doi
            headers = {
                "User-Agent": "Mozilla/5.0"
            }
            
            response = requests.get(
                url,
                headers=headers,
                timeout=20,
                allow_redirects=True
            )
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, "html.parser")
            abstract = ""
            meta_candidates = [
                "citation_abstract",
                "description",
                "dc.description"
            ]
            for meta_name in meta_candidates:
                tag = soup.find("meta", attrs={"name": meta_name})
                if tag and tag.get("content"):
                    abstract = tag.get("content").strip()
                    break
                
            return self.build_result(
                doi=doi,
                abstract=abstract,
                authors="",
                paper_id=paper_id
            )
            
        except Exception:
            return None
