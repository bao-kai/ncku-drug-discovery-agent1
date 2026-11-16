"""Exact PMID retrieval from Europe PMC; preserves source metadata and abstract."""
from datetime import datetime, timezone
import re
from typing import Any, Literal

import requests
from pydantic import BaseModel

ENDPOINT = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'


def search_publications(query: str, cursor_mark: str = '*') -> dict:
    """One relevance-sorted page of source-native metadata/abstracts; no automatic date filter."""
    if not query.strip():
        raise ValueError('A literature query is required')
    params = {'query': query.strip(), 'resultType': 'core', 'format': 'json',
              'pageSize': 20, 'cursorMark': cursor_mark}
    result = {'source': 'europe_pmc', 'query_parameters': params,
              'source_url': requests.Request('GET', ENDPOINT, params=params).prepare().url,
              'retrieved_at': datetime.now(timezone.utc).isoformat(), 'status': 'error',
              'records': [], 'total_count': None, 'next_cursor_mark': None,
              'truncated': None, 'error': None}
    try:
        response = requests.get(ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        total = int(payload['hitCount'])
        rows = payload['resultList']['result']
        if not isinstance(rows, list) or len(rows) > 20 or (total > 0 and not rows and cursor_mark == '*'):
            raise ValueError('Malformed publication search response')
        ids = [(row['source'], row['id']) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate publication IDs in page')
        next_cursor = payload.get('nextCursorMark')
        result.update(status='success' if rows else 'not_found', records=rows, total_count=total,
                      next_cursor_mark=next_cursor,
                      truncated=(total > len(rows)) if cursor_mark == '*' else bool(rows and next_cursor and next_cursor != cursor_mark))
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def get_open_access_full_text(pmcid: str) -> dict:
    """Retrieve publicly available article XML only when specifically requested."""
    pmcid = pmcid.strip().upper()
    if not re.fullmatch(r'PMC[1-9][0-9]*', pmcid):
        raise ValueError('Expected a PMCID such as PMC123456')
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
    result = {'source':'europe_pmc', 'record_id':pmcid, 'source_url':url,
              'retrieved_at':datetime.now(timezone.utc).isoformat(),
              'status':'error', 'raw_xml':None, 'error':None}
    try:
        response = requests.get(url, timeout=30)
        if response.status_code in (404, 403):
            result.update(status='unavailable', error='Full text is not available through this endpoint')
            return result
        response.raise_for_status()
        from xml.etree import ElementTree
        try:
            article = ElementTree.fromstring(response.text)
        except ElementTree.ParseError as exc:
            raise ValueError('Invalid full-text XML') from exc
        if article.tag != 'article':
            raise ValueError('Response is not article XML')
        article_ids = [node.text.strip().upper() for node in article.findall('./front/article-meta/article-id')
                       if node.get('pub-id-type') in ('pmc', 'pmcid') and node.text]
        if not any(('PMC' + value if value.isdigit() else value) == pmcid for value in article_ids):
            raise ValueError('Full-text PMCID does not match the requested article')
        result.update(status='success', raw_xml=response.text)
    except (requests.RequestException, ValueError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


class PublicationLookup(BaseModel):
    source: Literal['europe_pmc'] = 'europe_pmc'
    record_id: str
    source_url: str
    retrieved_at: datetime
    status: Literal['success', 'not_found', 'error']
    raw_record: dict[str, Any] | None = None
    error: str | None = None


def get_publication_by_pmid(pmid: str) -> PublicationLookup:
    """Retrieve a PubMed record's core metadata; never claim full text was read."""
    pmid = pmid.strip()
    if not re.fullmatch(r'[1-9][0-9]*', pmid):
        raise ValueError('Expected a numeric PMID')
    params = {'query': f'EXT_ID:{pmid} AND SRC:MED', 'resultType': 'core',
              'format': 'json', 'pageSize': 1}
    url = requests.Request('GET', ENDPOINT, params=params).prepare().url
    metadata = dict(record_id=pmid, source_url=url, retrieved_at=datetime.now(timezone.utc))
    try:
        response = requests.get(ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        count = int(payload['hitCount'])
        rows = payload['resultList']['result']
        if count == 0 and not rows:
            return PublicationLookup(**metadata, status='not_found')
        if count != 1 or len(rows) != 1 or not isinstance(rows[0], dict) or str(rows[0].get('id')) != pmid or rows[0].get('source') != 'MED':
            raise ValueError('Publication response does not match the requested PMID')
        return PublicationLookup(**metadata, status='success', raw_record=rows[0])
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return PublicationLookup(**metadata, status='error', error=f'{type(exc).__name__}: {exc}')
