from pathlib import Path
import requests

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "LING-144 n-gram language model (educational use)"


def fetch_wikipedia_article(title: str, *, timeout: float = 30) -> str:
    """Return the plain-text extract for a Wikipedia article."""
    if not title.strip():
        raise ValueError("Wikipedia title cannot be empty")
    response = requests.get(
        WIKIPEDIA_API_URL,
        params={"action": "query", "format": "json", "titles": title,
                "prop": "extracts", "explaintext": True},
        headers={"User-Agent": USER_AGENT}, timeout=timeout,
    )
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"Wikipedia article not found: {title!r}")
    return page["extract"]


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
