import requests
from bs4 import BeautifulSoup


def ingest_from_url(url: str):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = " ".join(paragraphs)

        with open("data/web_learned.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n\n")

        return "Learned from URL successfully"

    except Exception as e:
        return str(e)