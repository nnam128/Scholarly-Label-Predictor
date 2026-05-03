import requests
from typing import Optional, Dict

from src.crawler.crawler import BaseCrawler


class SemanticScholarCrawler(BaseCrawler):
    """
    Crawl metadata from Semantic Scholar API using DOI.
    Endpoint:
    https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}
    Returned fields:
    - abstract
    - authors
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:"
    FIELDS = "title,abstract,authors"
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
        url = self.BASE_URL + doi
        params = {"fields": self.FIELDS}
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout
            )
            if response.status_code != 200:
                #print(f"[SemanticScholar] Failed DOI: {doi} | "f"Status: {response.status_code}")
                return None
            data = response.json()
            abstract = self._extract_abstract(data)
            authors = self._extract_authors(data)
            return self.build_result(
                doi=doi,
                abstract=abstract,
                authors=authors,
                paper_id=paper_id
            )
        except Exception as e:
            print(f"[SemanticScholar] Error for DOI {doi}: {str(e)}")
            return None
        
    def _extract_authors(self, data: Dict) -> str:
        author_list = data.get("authors", [])
        authors = []
        for author in author_list:
            name = author.get("name", "").strip()
            if name:
                authors.append(name)
        return self.format_authors(authors)
    
    def _extract_abstract(self, data: Dict) -> str:
        abstract = data.get("abstract", "")
        if not abstract:
            return ""
        
        return str(abstract).strip()
