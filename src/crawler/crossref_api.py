import requests
from typing import Optional, Dict

from src.crawler.crawler import BaseCrawler


class CrossRefCrawler(BaseCrawler):
    """
    Crawl metadata from CrossRef API using DOI.
    CrossRef endpoint:
    https://api.crossref.org/works/{doi}
    Returned fields:
    - abstract
    - authors
    """
    
    BASE_URL = "https://api.crossref.org/works/"
    
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
        
        try:
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code != 200:
                #print(f"[CrossRef] Failed DOI: {doi} | Status: {response.status_code}")
                return None
            
            data = response.json()
            message = data.get("message", {})
            
            abstract = self._extract_abstract(message)
            authors = self._extract_authors(message)
            
            return self.build_result(
                doi=doi,
                abstract=abstract,
                authors=authors,
                paper_id=paper_id
            )
            
        except Exception as e:
            print(f"[CrossRef] Error for DOI {doi}: {str(e)}")
            return None
        
    def _extract_authors(self, message: Dict) -> str:
        author_list = message.get("author", [])
        
        authors = []
        for author in author_list:
            given = author.get("given", "")
            family = author.get("family", "")
            
            full_name = f"{given} {family}".strip()
            if full_name:
                authors.append(full_name)
                
        return self.format_authors(authors)
    
    def _extract_abstract(self, message: Dict) -> str:
        abstract = message.get("abstract", "")
        
        if not abstract:
            return ""
        
        abstract = str(abstract)
        
        abstract = abstract.replace("<jats:p>", "")
        abstract = abstract.replace("</jats:p>", "")
        abstract = abstract.replace("<p>", "")
        abstract = abstract.replace("</p>", "")
        abstract = abstract.strip()
        
        return abstract
