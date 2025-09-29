from ddgs import DDGS

def search_web(query: str):
  """Searches the web for the given query using DuckDuckGo."""
  with DDGS() as ddgs:
    # Corrected line: Pass the 'query' as a positional argument.
    results = [r for r in ddgs.text(query, max_results=5)][:5]
    return str(results)