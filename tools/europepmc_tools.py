from langchain_core.tools import tool
from query_europepmc import get_publication_by_pmid, search_publications, get_open_access_full_text


@tool
def europe_pmc_search(query: str, cursor_mark: str = '*') -> dict:
    """Search up to 20 literature records with abstracts/full-text links, relevance first and no implicit date restriction. Query is the Plan's explicit Europe PMC search expression. Hits are candidates, not verified negative/safety/mechanism evidence. Continue only when the Plan/follow-up calls for another page."""
    return search_publications(query, cursor_mark)


@tool
def get_publication_full_text_tool(pmcid: str) -> dict:
    """Retrieve available open-access full-text XML by PMCID only when needed; unavailable text is not negative evidence."""
    return get_open_access_full_text(pmcid)


@tool
def get_publication_by_pmid_tool(pmid: str) -> dict:
    """Fetch one Europe PMC publication by PMID, with native metadata/abstract and provenance, not full text."""
    return get_publication_by_pmid(pmid).model_dump(mode='json')
