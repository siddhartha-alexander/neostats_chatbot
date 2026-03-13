# utils/web_search.py
from duckduckgo_search import DDGS


def web_search(query):
    try:
        results_text = []

        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)

            for r in results:
                results_text.append(r["title"] + ": " + r["body"])

        return "\n".join(results_text)

    except Exception as e:
        return f"Web search failed: {str(e)}"