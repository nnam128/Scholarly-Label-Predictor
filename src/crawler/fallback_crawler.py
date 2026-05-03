from typing import Optional, Dict, List
from bs4 import BeautifulSoup
import requests


from src.crawler.semantic_scholar_api import SemanticScholarCrawler
from src.crawler.crossref_api import CrossRefCrawler
from src.crawler.openalex_api import OpenAlexCrawler
from src.crawler.doi_resolver_api import DOIResolverCrawler


class FallbackCrawler:
    """
    Multi-source fallback crawler.
    Strategy:
    1. Semantic Scholar
    2. CrossRef
    3. OpenAlex
    4. DOI Resolver
    Stop immediately when a crawler successfully returns useful metadata.
    This helps maximize coverage while keeping the pipeline robust against API failures, 
    missing records, and incomplete metadata.
    """
    
    def __init__(self, timeout: int = 30):
        self.crawlers: List = [
            SemanticScholarCrawler(timeout=timeout),
            CrossRefCrawler(timeout=timeout),
            OpenAlexCrawler(timeout=timeout),
            DOIResolverCrawler(timeout=timeout)
        ]
        
    def fetch_by_doi(
        self,
        doi: str,
        paper_id: int
    ) -> Optional[Dict]:
        if not doi:
            return None
        
        for crawler in self.crawlers:
            crawler_name = crawler.__class__.__name__
            
            try:
                result = crawler.fetch_by_doi(
                    doi=doi,
                    paper_id=paper_id
                )
                
                if self._is_valid_result(result):
                    #print(f"[SUCCESS] {crawler_name} -> "f"DOI: {doi}")
                    result["source"] = crawler_name
                    return result
                
                #print(f"[EMPTY] {crawler_name} -> "f"DOI: {doi}")
                
            except Exception as e:
                #print(f"[ERROR] {crawler_name} -> "f"DOI: {doi} | {str(e)}")
                pass
                
        print(f"[FAILED] API crawlers failed for DOI: {doi}")
        return None
    
    def _is_valid_result(self, result: Optional[Dict]) -> bool:
        
        if not result:
            return False
        
        abstract = str(result.get("abstract", "")).strip()
        
        return len(abstract) > 30
    