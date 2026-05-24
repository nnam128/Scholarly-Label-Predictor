from typing import Optional, Dict, List

from src.crawler.semantic_scholar_paperid_crawler import SemanticScholarPaperIDCrawler
from src.crawler.semantic_scholar_api import SemanticScholarCrawler
from src.crawler.crossref_api import CrossRefCrawler
from src.crawler.openalex_api import OpenAlexCrawler
from src.crawler.doi_resolver_api import DOIResolverCrawler


class FallbackCrawler:
    """
    Multi-source fallback crawler.
    Strategy:
    1. DOI-based crawlers
    2. If metadata incomplete:
       fallback to title-based recovery
    """
    def __init__(self, timeout: int = 30):
        self.s2_paperid_crawler = SemanticScholarPaperIDCrawler(
            timeout=timeout
        )
        self.fallback_crawlers: List = [
            SemanticScholarCrawler(timeout=timeout),
            CrossRefCrawler(timeout=timeout),
            OpenAlexCrawler(timeout=timeout),
            DOIResolverCrawler(timeout=timeout),
        ]
    def fetch_by_doi(
        self,
        doi: str,
        title: str,
        paper_id: int,
    ) -> Optional[Dict]:
        best_result = None
        
        # STEP 1: NORMAL DOI CRAWLERS
        if doi:
            for crawler in self.fallback_crawlers:
                crawler_name = crawler.__class__.__name__
                try:
                    result = crawler.fetch_by_doi(
                        doi=doi,
                        paper_id=paper_id
                    )
                    if result:
                        result["source"] = crawler_name
                        # save best partial result
                        if not best_result:
                            best_result = result
                        # perfect result -> return immediately
                        if self._is_complete_result(result):
                            return result
                except Exception:
                    pass
                
        # STEP 2: TITLE-BASED RECOVERY
        try:
            result = self.s2_paperid_crawler.fetch_by_doi(
                doi=doi,
                title=title,
                paper_id=paper_id
            )
            if result:
                result["source"] = "SemanticScholarPaperIDCrawler"
                # merge missing fields
                if best_result:
                    if not best_result.get("abstract"):
                        best_result["abstract"] = result.get("abstract")
                    if not best_result.get("authors"):
                        best_result["authors"] = result.get("authors")
                    # if merged result now good enough
                    if self._is_complete_result(best_result):
                        return best_result
                else:
                    if self._is_valid_result(result):
                        return result
        except Exception:
            pass
        # =====================================================
        # RETURN PARTIAL RESULT
        # =====================================================
        if best_result:
            return best_result
        print(f"[FAILED] API crawlers failed for DOI: {doi}")
        return None
    
    def _is_valid_result(
        self,
        result: Optional[Dict]
    ) -> bool:
        if not result:
            return False
        abstract = str(
            result.get("abstract", "")
        ).strip()
        return len(abstract) > 30
    def _is_complete_result(
        self,
        result: Optional[Dict]
    ) -> bool:
        if not result:
            return False
        abstract = str(
            result.get("abstract", "")
        ).strip()
        authors = result.get("authors")
        has_authors = (
            authors is not None and len(authors) > 0
        )
        return len(abstract) > 30 and has_authors