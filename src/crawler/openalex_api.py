import requests
from typing import Optional, Dict

from src.crawler.crawler import BaseCrawler


class OpenAlexCrawler(BaseCrawler):
    """
    Crawl metadata from OpenAlex using DOI.
    OpenAlex endpoint:
    https://api.openalex.org/works/https://doi.org/{doi}
    Returned fields:
    - abstract
    - authors
    """
    BASE_URL = "https://api.openalex.org/works/https://doi.org/"
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
                #print(f"[OpenAlex] Failed DOI: {doi} | Status: {response.status_code}")
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
            print(f"[OpenAlex] Error for DOI {doi}: {str(e)}")
            return None
    def _extract_authors(self, data: Dict) -> str:
        authorships = data.get("authorships", [])
        authors = []
        for item in authorships:
            author_info = item.get("author", {})
            name = author_info.get("display_name", "").strip()
            if name:
                authors.append(name)
        return self.format_authors(authors)
    
    def _extract_abstract(self, data: Dict) -> str:
        inverted_index = data.get("abstract_inverted_index", {})
        if not inverted_index:
            return ""
        position_to_word = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_to_word[pos] = word
        if not position_to_word:
            return ""
        ordered_words = [
            position_to_word[pos]
            for pos in sorted(position_to_word.keys())
        ]
        return " ".join(ordered_words).strip()
