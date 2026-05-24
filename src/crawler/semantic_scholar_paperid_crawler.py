import requests
from typing import Optional, Dict, List

from src.crawler.crawler import BaseCrawler


class SemanticScholarPaperIDCrawler(BaseCrawler):
    """
    Last-resort recovery crawler.
    Search strategy:
    1. OpenAlex
    2. Semantic Scholar
    3. Crossref
    Uses TITLE search to recover:
    - abstract
    - authors
    - DOI
    """
    OPENALEX_URL = "https://api.openalex.org/works"
    S2_SEARCH_URL = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
    )
    CROSSREF_URL = "https://api.crossref.org/works"
    S2_FIELDS = "title,abstract,authors,externalIds"
    API_KEY = "YOUR_API_KEY"
    def __init__(self, timeout: int = 30):
        super().__init__(timeout=timeout)
        self.s2_headers = {
            "x-api-key": self.API_KEY
        }
    # ────────────────────────────────────────────────────────
    def fetch_by_doi(
        self,
        doi: str,
        title: str,
        paper_id: int
    ) -> Optional[Dict]:
        if not title:
            return None
        # ====================================================
        # 1. OpenAlex
        # ====================================================
        data = self._search_openalex(title)
        if data:
            abstract = self._extract_openalex_abstract(data)
            authors  = self._extract_openalex_authors(data)
            if abstract:
                recovered_doi = data.get("doi")
                return self.build_result(
                    doi=recovered_doi or doi,
                    paper_id=paper_id,
                    abstract=abstract,
                    authors=authors,
                )
        # ====================================================
        # 2. Semantic Scholar
        # ====================================================
        data = self._search_semantic_scholar(title)
        if data:
            abstract = self._extract_s2_abstract(data)
            authors  = self._extract_s2_authors(data)
            if abstract:
                external_ids = data.get(
                    "externalIds",
                    {}
                )
                recovered_doi = external_ids.get("DOI")
                return self.build_result(
                    doi=recovered_doi or doi,
                    paper_id=paper_id,
                    abstract=abstract,
                    authors=authors,
                )
        # ====================================================
        # 3. Crossref
        # ====================================================
        data = self._search_crossref(title)
        if data:
            abstract = self._extract_crossref_abstract(data)
            authors  = self._extract_crossref_authors(data)
            if abstract:
                recovered_doi = data.get("DOI")
                return self.build_result(
                    doi=recovered_doi or doi,
                    paper_id=paper_id,
                    abstract=abstract,
                    authors=authors,
                )
        return None
    # ────────────────────────────────────────────────────────
    def fetch_batch(
        self,
        rows: List[Dict]
    ) -> List[Optional[Dict]]:
        output = []
        for row in rows:
            try:
                result = self.fetch_by_doi(
                    doi=row.get("doi", ""),
                    title=row.get("title", ""),
                    paper_id=row.get("id"),
                )
                output.append(result)
            except Exception:
                output.append(None)
        return outpu
    # ────────────────────────────────────────────────────────
    # SEARCHERS
    # ────────────────────────────────────────────────────────
    def _search_openalex(
        self,
        title: str
    ) -> Optional[Dict]:
        try:
            response = requests.get(
                self.OPENALEX_URL,
                params={
                    "search": title,
                    "per-page": 1,
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return None
            results = response.json().get(
                "results",
                []
            )
            if not results:
                return None
            return results[0]
        except Exception:
            return None
    def _search_semantic_scholar(
        self,
        title: str
    ) -> Optional[Dict]:
        try:
            response = requests.get(
                self.S2_SEARCH_URL,
                headers=self.s2_headers,
                params={
                    "query": title,
                    "limit": 1,
                    "fields": self.S2_FIELDS,
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return None
            data = response.json()
            papers = data.get("data", [])
            if not papers:
                return None
            return papers[0]
        except Exception:
            return None
    def _search_crossref(
        self,
        title: str
    ) -> Optional[Dict]:
        try:
            response = requests.get(
                self.CROSSREF_URL,
                params={
                    "query.title": title,
                    "rows": 1,
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return None
            items = response.json()["message"]["items"]
            if not items:
                return None
            return items[0]
        except Exception:
            return None
    # ────────────────────────────────────────────────────────
    # OPENALEX EXTRACTORS
    # ────────────────────────────────────────────────────────
    def _extract_openalex_abstract(
        self,
        data: Dict
    ) -> str:
        inverted = data.get(
            "abstract_inverted_index"
        )
        if not inverted:
            return ""
        words = {}
        for word, positions in inverted.items():
            for pos in positions:
                words[pos] = word
        abstract = " ".join(
            words[i]
            for i in sorted(words.keys())
        )
        return abstract.strip()
    def _extract_openalex_authors(
        self,
        data: Dict
    ) -> str:
        authorships = data.get(
            "authorships",
            []
        )
        authors = []
        for author_data in authorships:
            author = author_data.get(
                "author",
                {}
            )
            name = author.get(
                "display_name",
                ""
            ).strip()
            if name:
                authors.append(name)
        return self.format_authors(authors)
    # ────────────────────────────────────────────────────────
    # SEMANTIC SCHOLAR EXTRACTORS
    # ────────────────────────────────────────────────────────
    def _extract_s2_abstract(
        self,
        data: Dict
    ) -> str:
        abstract = data.get("abstract", "")
        if not abstract:
            return ""
        return str(abstract).strip()
    def _extract_s2_authors(
        self,
        data: Dict
    ) -> str:
        author_list = data.get(
            "authors",
            []
        )
        authors = []
        for author in author_list:
            name = author.get(
                "name",
                ""
            ).strip()
            if name:
                authors.append(name)
        return self.format_authors(authors)
    # ────────────────────────────────────────────────────────
    # CROSSREF EXTRACTORS
    # ────────────────────────────────────────────────────────
    def _extract_crossref_abstract(
        self,
        data: Dict
    ) -> str:
        abstract = data.get("abstract", "")
        if not abstract:
            return ""
        return (
            str(abstract)
            .replace("<jats:p>", "")
            .replace("</jats:p>", "")
            .strip()
        )
    def _extract_crossref_authors(
        self,
        data: Dict
    ) -> str:
        author_list = data.get(
            "author",
            []
        )
        authors = []
        for author in author_list:
            given = author.get("given", "").strip()
            family = author.get("family", "").strip()
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)
        return self.format_authors(authors)